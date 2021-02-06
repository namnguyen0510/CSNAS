import torch
import sys
import random
import torchvision
from torchvision.datasets import DatasetFolder
from utils import *

class JigsawLoader(DatasetFolder):
    def __init__(self, root_dir):
        super(JigsawLoader, self).__init__(root_dir, pil_loader, extensions=('jpg'))
        self.root_dir = root_dir
        self.color_transform = torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4)
        self.color_transform = torchvision.transforms.ColorJitter()
        self.flips = [torchvision.transforms.RandomHorizontalFlip(), torchvision.transforms.RandomVerticalFlip()]
        self.normalize = torchvision.transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        original = self.loader(path)

        samples = []
        image = torchvision.transforms.Resize((32,32))(original)
        sample = torchvision.transforms.CenterCrop((32,32))(image)
        samples.append(sample)
        for i in range(1):
            sample = torchvision.transforms.RandomCrop((32,32))(image)
            sample = torchvision.transforms.RandomHorizontalFlip(p = 0.5)(sample)
            sample = torchvision.transforms.RandomVerticalFlip(p = 0.5)(sample)
            sample = torchvision.transforms.RandomGrayscale(p=0.5)(sample)
            samples.append(sample)

        samples = [torchvision.transforms.Resize((32,32))(patch) for patch in samples]
        #augmentation collor jitter
        image = self.color_transform(image)
        samples = [self.color_transform(patch) for patch in samples]
        # augmentation - flips
        image = self.flips[0](image)
        image = self.flips[1](image)
        # to tensor
        image = torchvision.transforms.functional.to_tensor(image)
        samples = [torchvision.transforms.functional.to_tensor(patch) for patch in samples]
        # normalize

        image = self.normalize(image)
        samples = [self.normalize(patch) for patch in samples]
        random.shuffle(samples)


        return {'original': image,'patches': samples, 'index' : index}

class Memory(object):
    def __init__(self, device, size = 9000, weight = 0.5, seed = 7):
        self.memory = np.zeros((size, 256))
        self.weighted_sum = np.zeros((size, 256))
        self.weighted_count = 0
        self.weight = weight
        self.device = device
        self.seed = seed
        self.size = size

    def initialize(self, net, train_loader):
        self.update_weighted_count()
        print('Saving representations to memory')
        bar = Progbar(len(train_loader), stateful_metrics=[])
        for step, batch in enumerate(train_loader):
            with torch.no_grad():
                images = batch['original'].to(self.device)
                index = batch['index']
                output = net(images = images, mode = 0)
                self.weighted_sum[index, :] = output.cpu().numpy()
                self.memory[index, :] = self.weighted_sum[index, :]
                bar.update(step, values= [])

    def update(self, index, values):
        self.weighted_sum[index, :] = values + (1 - self.weight) * self.weighted_sum[index, :]
        self.memory[index, :] = self.weighted_sum[index, :]/self.weighted_count
        pass

    def update_weighted_count(self):
        self.weighted_count = 1 + (1 - self.weight) * self.weighted_count

    def return_random(self, size, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        random.seed(self.seed)
        allowed = [x for x in range(index[0])] + [x for x in range(index[0] + 1, self.size )]
        index = random.sample(allowed, size)
        return self.memory[index,:]
    def return_representations(self, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        return torch.Tensor(self.memory[index,:])

class ModelCheckpoint():
    def __init__(self, mode, directory):
        self.directory = directory
        if mode =='min':
            self.best = np.inf
            self.monitor_op = np.less
        elif mode == 'max':
            self.best = 0
            self.monitor_op = np.greater
        else:
            print('\nChose mode \'min\' or \'max\'')
            raise Exception('Mode should be either min or max')
        if os.path.isdir(self.directory):
            shutil.rmtree(self.directory)
            os.mkdir(self.directory)
        else:
            os.mkdir(self.directory)

    def save_model(self, model, current_value, epoch):
        if self.monitor_op(current_value, self.best):
            print('\nSave model, best value {:.3f}, epoch: {}'.format(current_value, epoch))
            self.best = current_value
            torch.save(model.state_dict(), os.path.join(self.directory,'epoch_{}'.format(epoch)))

class NoiseContrastiveEstimator():
    def __init__(self, device):
        self.device = device
    def __call__(self, original_features, path_features, index, memory, negative_nb = 1000):
        loss = 0
        for i in range(original_features.shape[0]):

            temp = 0.07
            cos = torch.nn.CosineSimilarity()
            #cos = torch.nn.KLDivLoss()
            criterion = torch.nn.CrossEntropyLoss()

            negative = memory.return_random(size = negative_nb, index = [index[i]])
            negative = torch.Tensor(negative).to(self.device).detach()

            image_to_modification_similarity = cos(original_features[None, i,:], path_features[None, i,:])/temp
            matrix_of_similarity = cos(path_features[None, i,:], negative) / temp


            similarities = torch.cat((image_to_modification_similarity, matrix_of_similarity))
            loss += criterion(similarities[None,:], torch.tensor([0]).to(self.device))
        return  loss / original_features.shape[0]



# example of calculating the js divergence between two mass functions
from math import log2
from math import sqrt
from numpy import asarray
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


def kl(a,b):
    ma = torch.mean(a)
    mb = torch.mean(b)
    sa = torch.std(a)
    sb = torch.std(b)
    kl = torch.log(sb/sa)+(torch.pow(sa,2)+torch.pow((ma-mb),2))/(2*torch.pow(sb,2))-1/2
    kl = torch.reshape(kl,(-1,))
    return kl

def kl_neg(a,b,device):
    kl_neg = []
    for i in range(b.shape[0]):
        cost = kl(a,b[i])
        kl_neg.append(cost)
    #corr_neg = np.array(corr_neg).astype('float32')
    tensor = torch.cat(kl_neg).type(torch.FloatTensor).to(device)
    return tensor



class JS():
    def __init__(self, device):
        self.device = device
    def __call__(self, original_features, path_features, index, memory, negative_nb = 1000):
        loss = 0
        for i in range(original_features.shape[0]):
            temp =0.07
            criterion = torch.nn.CrossEntropyLoss()
            negative = memory.return_random(size = negative_nb, index = [index[i]])
            negative = torch.Tensor(negative).to(self.device).detach()

            t1 = original_features[None,i,:]
            t2 = path_features[None,i,:]
            t3 = negative

            #similarities = np.concatenate((kl(t1,t2),kl_neg(t2,t3)))
            similarities = torch.cat((kl(t1,t2),kl_neg(t2,t3,self.device)))
            #similarities = similarities.to(self.device)
            loss += criterion(similarities[None,:], torch.tensor([0]).to(self.device))
        return loss / original_features.shape[0]




def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

class Logger:
    def __init__(self, file_name):
        self.file_name = file_name
        index = ['Epoch']
        with open('{}.csv'.format(self.file_name), 'w') as file:
            file.write('Epoch,Loss,Time\n')
    def update(self, epoch, loss):
        now = datetime.datetime.now()
        with open('{}.csv'.format(self.file_name), 'a') as file:
            file.write('{},{:.4f},{}\n'.format(epoch,loss,now))

class AvgrageMeter(object):

  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

class Progbar(object):
    '''
    Taken from:
    https://github.com/keras-team/keras
    '''
    """Displays a progress bar.
    # Arguments
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05,
                 stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        self._values = collections.OrderedDict()
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                       current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                # Stateful metrics output a numeric value.  This representation
                # means "take an average from a single value" but keeps the
                # numeric formatting.
                self._values[k] = [v, 1]
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = ('%d:%02d:%02d' %
                                  (eta // 3600, (eta % 3600) // 60, eta % 60))
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None):
        self.update(self._seen_so_far + n, values)
