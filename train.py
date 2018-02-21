"""
created by: Donghyeon Won
Modified codes from
    http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    https://github.com/pytorch/examples/tree/master/imagenet
"""
from __future__ import print_function
import os
import argparse
import numpy as np
import pandas as pd
import time
import shutil
from PIL import Image
from sklearn.metrics import accuracy_score, mean_squared_error

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.models as models

from util import ProtestDataset, modified_resnet50


# for indexing output of the model
protest_idx = Variable(torch.LongTensor([0]))
violence_idx = Variable(torch.LongTensor([1]))
visattr_idx = Variable(torch.LongTensor(list(range(2,12))))
best_loss = float("inf")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calculate_loss(output, target, criterions, weights = [2,40,1]):
    """calculate loss"""
    # output = output.float()
    N_protest = int(target['protest'].data.sum())
    if N_protest == 0:
        # if no protest image in target
        outputs = [None]
        outputs[0] = output.index_select(1, protest_idx)  # protest output
        targets = [None]
        targets[0] = target['protest'].float()  # protest target
        losses = [weights[i] * criterions[i](outputs[i], targets[i]) for i in range(1)]
        scores = {}
        scores['protest_acc'] = accuracy_score(outputs[0].data.round(), targets[0].data)
        scores['violence_mse'] = 0
        scores['visattr_acc'] = 0
        return losses, scores, N_protest

    protest_mask = target['protest'].byte()
    outputs = [None] * 3
    outputs[0] = output.index_select(1, protest_idx)  # protest output
    outputs[1] = output.index_select(1, violence_idx) # violence output
    outputs[1] = torch.masked_select(outputs[1], protest_mask)
    outputs[2] = output.index_select(1, visattr_idx)  # visual attribute output
    outputs[2] = torch.masked_select(outputs[2], protest_mask).view(N_protest, 10)
    targets = [None] * 3


    targets[0] = target['protest'].float()  # protest target
    targets[1] = target['violence'].float() # violence target
    targets[1] = torch.masked_select(targets[1], protest_mask)
    targets[2] = target['visattr'].float()  # visual attribute target
    targets[2] = torch.masked_select(targets[2], protest_mask).view(N_protest, 10)

    scores = {}
    scores['protest_acc'] = accuracy_score(outputs[0].data.round(), targets[0].data)
    scores['violence_mse'] = mean_squared_error(outputs[1].data, targets[1].data)
    scores['visattr_acc'] =(outputs[2].data.round() == targets[2].data).float().mean(dim = 1).mean() #mean accuracy



    losses = [weights[i] * criterions[i](outputs[i], targets[i]) for i in range(3)]
    # losses[1] = losses[1] / float(N_protest) * args.batch_size
    # losses[2] = losses[2] / float(N_protest) * args.batch_size
    # loss += weights[0] * criterions[0](output_protest, target_protest)
    # loss += weights[1] * criterions[1](output_violence, target_violence)
    # loss += weights[2] * criterions[2](output_visattr, target_visattr)

    return losses, scores, N_protest



def train(train_loader, model, criterions, optimizer, epoch):
    """training the model"""
    # train mode
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_history = AverageMeter()
    protest_acc = AverageMeter()
    violence_mse = AverageMeter()
    visattr_acc = AverageMeter()

    end = time.time()

    for i, sample in enumerate(train_loader):
        # measure data loading batch_time
        input, target = sample['image'], sample['label']
        data_time.update(time.time() - end)

        if args.cuda:
            input = input.cuda()
            for k, v in target.items():
                target[k] = v.cuda()
        target_var = {}
        for k,v in target.items():
            target_var[k] = Variable(v)

        input_var = Variable(input)
        output = model(input_var)

        losses, scores, N_protest = calculate_loss(output, target_var, criterions)

        optimizer.zero_grad()
        loss = 0
        for l in losses:
            loss += l
        loss.backward()
        optimizer.step()

        loss_history.update(loss.data[0], input.size(0))
        protest_acc.update(scores['protest_acc'], input.size(0))
        violence_mse.update(scores['violence_mse'], N_protest)
        visattr_acc.update(scores['visattr_acc'], N_protest)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
             print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Data {data_time.val:.2f} ({data_time.avg:.2f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Protest {protest_acc.val:.2f} ({protest_acc.avg:.2f})\t'
                  'Violence {violence_mse.val:.3f} ({violence_mse.avg:.3f})\t'
                  'Vis Attr {visattr_acc.val:.2f} ({visattr_acc.avg:.2f})'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=loss_history,
                   protest_acc = protest_acc, violence_mse = violence_mse,
                   visattr_acc = visattr_acc))

def validate(val_loader, model, criterions, epoch):
    """training the model"""
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_history = AverageMeter()
    protest_acc = AverageMeter()
    violence_mse = AverageMeter()
    visattr_acc = AverageMeter()

    end = time.time()
    for i, sample in enumerate(val_loader):
        # measure data loading batch_time
        input, target = sample['image'], sample['label']

        if args.cuda:
            input = input.cuda()
            for k, v in target.items():
                target[k] = v.cuda()
        input_var = Variable(input)

        target_var = {}
        for k,v in target.items():
            target_var[k] = Variable(v)

        output = model(input_var)

        losses, scores, N_protest = calculate_loss(output, target_var, criterions)
        loss = 0
        for l in losses:
            loss += l

        loss_history.update(loss.data[0], input.size(0))
        protest_acc.update(scores['protest_acc'], input.size(0))
        violence_mse.update(scores['violence_mse'], N_protest)
        visattr_acc.update(scores['visattr_acc'], N_protest)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
             print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Protest Acc {protest_acc.val:.2f} ({protest_acc.avg:.2f})\t'
                  'Violence MSE {violence_mse.val:.3f} ({violence_mse.avg:.3f})\t'
                  'Vis Attr Acc {visattr_acc.val:.2f} ({visattr_acc.avg:.2f})'
                  .format(
                   epoch, i, len(val_loader), batch_time=batch_time,
                   loss=loss_history, protest_acc = protest_acc,
                   violence_mse = violence_mse, visattr_acc = visattr_acc))
    print(' * Loss {loss.avg:.3f} Protest Acc {protest_acc.avg:.3f} '
          'Violence MSE {violence_mse.avg:.3f} '
          'Vis Attr Acc {visattr_acc.avg:.3f} '
          .format(loss = loss_history, protest_acc = protest_acc,
                  violence_mse = violence_mse, visattr_acc = visattr_acc))
    return loss_history.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def main():
    global best_loss
    data_dir = args.data_dir
    img_dir_train = os.path.join(data_dir, "img/train")
    img_dir_val = os.path.join(data_dir, "img/test") #use test data for validation
    txt_file_train = os.path.join(data_dir, "annot_train.txt")
    txt_file_val = os.path.join(data_dir, "annot_test.txt")

    # load pretrained resnet50 with a modified last fully connected layer
    model = modified_resnet50()

    # we need three different criterion for training
    criterion_protest = nn.BCELoss()
    criterion_violence = nn.MSELoss()
    criterion_visattr = nn.BCELoss()
    criterions = [criterion_protest, criterion_violence, criterion_visattr]



    if args.cuda and not torch.cuda.is_available():
        raise Exception("No GPU Found")
    if args.cuda:
        model = model.cuda()
        criterions = [criterion.cuda() for criterion in criterions]

    optimizer = torch.optim.Adam(
                    model.parameters(), args.lr,
                    )

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = ProtestDataset(
                        txt_file = txt_file_train,
                        img_dir = img_dir_train,
                        transform = transforms.Compose([
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                normalize,
                        ]))
    val_dataset = ProtestDataset(
                    txt_file = txt_file_val,
                    img_dir = img_dir_val,
                    transform = transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        normalize,
                    ]))
    train_loader = DataLoader(
                    train_dataset,
                    num_workers = args.workers,
                    batch_size = args.batch_size,
                    shuffle = True
                    )
    val_loader = DataLoader(
                    val_dataset,
                    num_workers = args.workers,
                    batch_size = args.batch_size)
    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterions, optimizer, epoch)
        loss = validate(val_loader, model, criterions, epoch)
        is_best = loss < best_loss
        if is_best:
            print('best model!!')
        best_loss = min(loss, best_loss)
        save_checkpoint({
            'epoch' : epoch + 1,
            'state_dict' : model.state_dict(),
            'best_loss' : best_loss,
            'optimizer' : optimizer.state_dict(),
        }, is_best)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        type=str,
                        default = "UCLA-protest",
                        help = "directory path to UCLA-protest",
                        )
    parser.add_argument("--cuda",
                        action = "store_true",
                        help = "use cuda?",
                        )
    parser.add_argument("--workers",
                        type = int,
                        default = 4,
                        help = "number of workers",
                        )
    parser.add_argument("--batch_size",
                        type = int,
                        default = 8,
                        help = "batch size",
                        )
    parser.add_argument("--epochs",
                        type = int,
                        default = 30,
                        help = "number of epochs",
                        )
    parser.add_argument("--weight_decay",
                        type = float,
                        default = 1e-4,
                        help = "weight decay",
                        )
    parser.add_argument("--lr",
                        type = float,
                        default = 0.01,
                        help = "learning rate",
                        )
    parser.add_argument("--momentum",
                        type = float,
                        default = 0.9,
                        help = "momentum",
                        )
    parser.add_argument("--print_freq",
                        type = int,
                        default = 10,
                        help = "print frequency",
                        )
    args = parser.parse_args()

    if args.cuda:
        protest_idx = protest_idx.cuda()
        violence_idx = violence_idx.cuda()
        visattr_idx = visattr_idx.cuda()

    main()
