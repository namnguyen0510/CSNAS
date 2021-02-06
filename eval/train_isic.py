import os
import sys
import numpy as np
import time
import torch
import utils
import glob
import random
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
from model import NetworkImageNet as Network

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--data', type=str, default='/home/nam/ISIC_2019', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=40*4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=3000, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=58, help='num of init channels')
parser.add_argument('--layers', type=int, default = 24, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=True, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='CNAS_ISIC_s5', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.2, help='label smoothing')
parser.add_argument('--gamma', type=float, default=0.999, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=3, help='epochs between two learning rate decays')
parser.add_argument('--parallel', action='store_true', default=True, help='data parallelism')
args = parser.parse_args()

args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

CLASSES = 8


class CrossEntropyLabelSmooth(nn.Module):

  def __init__(self, num_classes, epsilon):
    super(CrossEntropyLabelSmooth, self).__init__()
    self.num_classes = num_classes
    self.epsilon = epsilon
    self.logsoftmax = nn.LogSoftmax(dim=1)

  def forward(self, inputs, targets):
    log_probs = self.logsoftmax(inputs)
    targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
    targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
    loss = (-targets * log_probs).mean(0).sum()
    return loss


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
  model.drop_path_prob = args.drop_path_prob
  if args.parallel:
    model = nn.DataParallel(model).cuda()

  else:
    model = model.cuda()
  #checkpoint = torch.load('./model_best.pth.tar')
  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))
  loss_weight = np.array([3.04, 1, 3.45, 15.09, 4.51, 43.26, 32.45, 23.18])
  loss_weight = torch.from_numpy(loss_weight).float()
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
  criterion_smooth = criterion_smooth.cuda()

  optimizer = torch.optim.SGD(
    model.parameters(),
    args.learning_rate,
    momentum=args.momentum,
    weight_decay=args.weight_decay
    )
  #model.load_state_dict(checkpoint['state_dict'])
  traindir = os.path.join(args.data, 'train')
  validdir = os.path.join(args.data, 'test')
  normalize = transforms.Normalize(mean=[0.763, 0.545, 0.570], std=[0.141, 0.152, 0.169])
  train_data = dset.ImageFolder(
    traindir,
    transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ColorJitter(
        brightness=0.4,
        contrast=0.4,
        saturation=0.4,
        hue=0.2),
      transforms.ToTensor(),
      normalize,
    ]))
  valid_data = dset.ImageFolder(
    validdir,
    transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      normalize,
    ]))

  num_classes = len(train_data.classes)
  class_weight = np.array([1./213, 1./649, 1./188, 1./43, 1./144, 1./15,1./20,1./28])
  class_weight = torch.from_numpy(class_weight)

  train_targets = [sample[1] for sample in train_data.imgs]
  valid_targets = [sample[1] for sample in valid_data.imgs]

  train_samples_weight = [class_weight[class_id] for class_id in train_targets]
  train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weight, len(train_data))

  valid_samples_weight = [class_weight[class_id] for class_id in valid_targets]
  valid_sampler = torch.utils.data.sampler.WeightedRandomSampler(valid_samples_weight, len(valid_data))


  train_queue = torch.utils.data.DataLoader(
    train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)#, sampler = train_sampler)

  valid_queue = torch.utils.data.DataLoader(
    valid_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)#, sampler = valid_sampler)

  scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.decay_period, gamma=args.gamma)

  best_acc_top1 = 0
  for epoch in range(args.epochs):

    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob *(epoch/ args.epochs)

    train_acc, train_obj = train(train_queue, model, criterion, optimizer)
    logging.info('train_acc %f', train_acc)
    scheduler.step()
    valid_acc_top1, valid_acc_top5, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc_top1 %f', valid_acc_top1)
    logging.info('valid_acc_top5 %f', valid_acc_top5)
    utils.save(model, os.path.join(args.save, 'weights.pt'))

    is_best = False
    if valid_acc_top1 > best_acc_top1:
      best_acc_top1 = valid_acc_top1
      is_best = True

    utils.save_checkpoint({
      'epoch': epoch + 1,
      'state_dict': model.state_dict(),
      'best_acc_top1': best_acc_top1,
      'optimizer' : optimizer.state_dict(),
      }, is_best, args.save)


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    target = target.cuda(async=True)
    input = input.cuda()
    input = Variable(input)
    target = Variable(target)

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
    n = input.size(0)
    objs.update(loss.data.item(), n)
    top1.update(prec1.data.item(), n)
    top5.update(prec5.data.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()
  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = Variable(input).cuda()
      target = Variable(target).cuda(async=True)

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
      n = input.size(0)
      objs.update(loss.data.item(), n)
      top1.update(prec1.data.item(), n)
      top5.update(prec5.data.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, top5.avg, objs.avg


if __name__ == '__main__':
  main()
