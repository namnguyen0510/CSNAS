from genotypes import *
from model_generator import *
from visualize import plot
from data_loader import *
import torchvision
from torchvision.models.resnet import resnet50
import torch.nn as nn
from model import NetworkCIFAR as ChildNetwork
from utils import *
from model_generator import *
from visualize import plot



device = torch.device('cuda:0')
data_dir = '/media/namng/Drive_1/cifar10_1%/train'
negative_nb = 1000 # number of negative examples in NCE
lr = 0.001

dataset = JigsawLoader(data_dir)
train_loader = torch.utils.data.DataLoader(dataset, shuffle = True, batch_size = 2, num_workers=4)

# Argument for cell
steps = 4
k = sum(1 for i in range(steps) for n in range(2+i))
num_ops = 8
int_weights = np.repeat(1/num_ops,num_ops)
#weight_mat = np.repeat(int_weights,k).reshape(-1,num_ops)



print("Generation 1")
for t in range(10):
    print(t,'/',10)
    checkpoint_dir = 'checkpoint' + str(t)
    log_filename = 'log' + str(t)
    #normal_gene = generate_cell(num_ops, steps, weight_mat)
    #reduced_gene = generate_cell(num_ops, steps, weight_mat)
    #gene = [normal_gene[1],reduced_gene[1]]
    normal_gene = np.load('normal_cell.npy')
    reduced_gene = np.load('reduced_cell.npy')
    gene = [normal_gene, reduced_gene]
    cell = Network(C = 32, num_classes = 128, layers = 1, criterion = nn.CrossEntropyLoss, generator = gene, steps=4, multiplier=4, stem_multiplier=3)
    plot(cell, 'test')
    child_model = ChildNetwork(C=32, layers = 4, auxiliary = False, genotype = cell.genotype())
    child_model.drop_path_prob = 0

    net = Siamese(child_model).to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    memory = Memory(size = len(dataset), weight= 0.5, device = device)
    memory.initialize(net, train_loader)
    checkpoint = ModelCheckpoint(mode = 'min', directory = checkpoint_dir )
    noise_contrastive_estimator = NoiseContrastiveEstimator(device)
    logger = Logger(log_filename)
    loss_weight = 0.5
    for epoch in range(100):
        print('\nEpoch: {}'.format(epoch))
        memory.update_weighted_count()
        train_loss = AverageMeter('train_loss')
        bar = Progbar(len(train_loader), stateful_metrics=['train_loss', 'valid_loss'])
    
        for step, batch in enumerate(train_loader): 
        # prepare batch
            images = batch['original'].to(device)
            patches = [element.to(device) for element in batch['patches']]
            index = batch['index']
            representations = memory.return_representations(index).to(device).detach()
        # zero grad
            optimizer.zero_grad()
        
        #forward, loss, backward, step
            output = net(images = images, patches = patches, mode = 1)
        
        
            loss_1 = noise_contrastive_estimator(representations, output[1], index, memory, negative_nb = negative_nb)
            loss_2 = noise_contrastive_estimator(representations, output[0], index, memory, negative_nb = negative_nb) 
            loss = loss_weight * loss_1 + (1 - loss_weight) * loss_2
        
            loss.backward()
            optimizer.step()
        
        #update representation memory
            memory.update(index, output[0].detach().cpu().numpy())
        
        # update metric and bar
            train_loss.update(loss.item(), images.shape[0])
            bar.update(step, values=[('train_loss', train_loss.return_avg())])
        logger.update(epoch, train_loss.return_avg())

    #save model if improved
        checkpoint.save_model(net, train_loss.return_avg(), epoch)
        name = 'gene'+str(t)
        np.save(name,np.array([normal_gene[0],reduced_gene[0]]))
