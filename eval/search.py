from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from time import time
from genotypes import *
from model_generator import *
import gc
from ssl_utils import *
import torchvision
from torchvision.models.resnet import resnet50
import torch.nn as nn
from model import NetworkCIFAR as ChildNetwork
from utils import *
from model_generator import *
#from visualize import plot
import time as time
from functools import partial
import pickle
# Setting for search engine
seed = 1
random.seed(seed)
np.random.seed(seed)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device('cuda:0')
# Path to data
data_dir = './ISIC_test'
batch_size = 64*4
negative_nb = 2*(batch_size-1) # number of negative examples in NCE
lr = 1e-3

# Load Data
dataset = JigsawLoader(data_dir)
train_loader = torch.utils.data.DataLoader(dataset, shuffle = True, batch_size = batch_size, num_workers=4)
num_eval = 220
trials = Trials()

# Number of hidden nodes
steps = 5
k = sum(1 for i in range(steps) for n in range(2+i))
num_ops = 8

# Create paramter space for hyperopt
ops_hyperopt = {}
normal_ops = ['normal_ops_' + str(i).zfill(2) for i in range(k)]
for i in range(len(normal_ops)):
    ops_hyperopt.update({normal_ops[i]: hp.choice(normal_ops[i],np.array(range(num_ops), dtype = int))})
reduced_ops = ['reduced_ops_' + str(i).zfill(2) for i in range(k)]
for i in range(len(reduced_ops)):
    ops_hyperopt.update({reduced_ops[i]: hp.choice(reduced_ops[i],np.array(range(num_ops), dtype = int))})

# Search Engine
def objective_function(ops_hyperopt):
    steps = 5
    num_ops = 8
    score = 0
    normal_ops = ['normal_ops_' + str(i).zfill(2) for i in range(k)]
    reduced_ops = ['reduced_ops_' + str(i).zfill(2) for i in range(k)]

    encoded_normal_geno = []
    for i in range(len(normal_ops)):
        encoded_normal_geno.append(ops_hyperopt[normal_ops[i]])

    encoded_reduced_geno = []
    for i in range(len(reduced_ops)):
        encoded_reduced_geno.append(ops_hyperopt[reduced_ops[i]])

    encoded_normal_geno = np.array(encoded_normal_geno)
    encoded_reduced_geno = np.array(encoded_reduced_geno)
    print(encoded_normal_geno,encoded_reduced_geno)
    normal_gene = str2mat(encoded_normal_geno, steps, num_ops)
    reduced_gene = str2mat(encoded_reduced_geno, steps, num_ops)
    gene = [normal_gene[1],reduced_gene[1]]
    cell = Network(C = 32, num_classes = 128, layers = 8,
                   criterion = nn.CrossEntropyLoss, generator = gene, steps=steps, multiplier=4, stem_multiplier=3)
    torch.manual_seed(seed)
    child_model = ChildNetwork(C=32, layers = 8, auxiliary = False, genotype = cell.genotype())
    child_model.drop_path_prob = 0
    net = Siamese(child_model).to(device)
    net = nn.DataParallel(net).cuda()
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    memory = Memory(size = len(dataset), weight= 0.5, device = device, seed = 7)
    memory.initialize(net, train_loader)
    noise_contrastive_estimator = NoiseContrastiveEstimator(device)

    loss_weight = 0.5
    for epoch in range(1):
        print('\nEpoch: {}'.format(epoch))
        memory.update_weighted_count()
        train_loss = AverageMeter('train_loss')
        running_loss = 0.0
        bar = Progbar(len(train_loader), stateful_metrics=['train_loss', 'valid_loss'])

        for step, batch in enumerate(train_loader):
            images = batch['original'].to(device)
            patches = [element.to(device) for element in batch['patches']]
            index = batch['index']
            representations = memory.return_representations(index).to(device).detach()
            optimizer.zero_grad()
            output = net(images = images, patches = patches, mode = 1)
            loss_1 = noise_contrastive_estimator(representations, output[1], index, memory, negative_nb = negative_nb)
            loss_2 = noise_contrastive_estimator(representations, output[0], index, memory, negative_nb = negative_nb)
            loss = loss_weight * loss_1 + (1 - loss_weight) * loss_2
            loss.backward()
            optimizer.step()
            running_loss += loss.item()*images.size(0)
            memory.update(index, output[0].detach().cpu().numpy())
            train_loss.update(loss.item(), images.shape[0])
            bar.update(step, values=[('train_loss', train_loss.return_avg())])

        score = running_loss/len(train_loader)
    print(score)
    del memory
    del net
    del child_model
    del images, patches, output
    del representations
    del loss_1, loss_2, loss
    gc.collect()
    torch.cuda.empty_cache()

    return {'loss': score, 'status': STATUS_OK}

since  = time.time()
algo = partial(tpe.suggest, n_startup_jobs = 20, gamma = 0.2, n_EI_candidates = 20000)
maxevals = num_eval
nevals = 0
failed = 0
while nevals < maxevals:
    best_param = fmin(objective_function, ops_hyperopt, algo=tpe.suggest, max_evals=maxevals + failed,
                  trials=trials, rstate = np.random.RandomState(seed))
    nevals = sum([1 for x in trials.results if x['status']=='ok'])
    failed = maxevals - nevals

pickle.dump(trials, open("tpe_trials.p","wb"))
loss = [x['result']['loss'] for x in trials.trials]
best_param_values = [x for x in best_param.values()]

np.save('BNAS_loss.npy',np.array(loss))
np.save('BNAS_best_cell.npy', np.array(best_param_values))
print("##### Results ###")
print("Score best parameters: ", min(loss))
print("Best Cell: ", best_param_values)
print("Times:" , time.time() - since)
