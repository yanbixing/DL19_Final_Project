from __future__ import print_function
from __future__ import division

#import numpy as np
import argparse
#import random
#import shutil
import time
#import warnings


import torch
import torch.nn as nn

import torch.nn.parallel
#import torch.backends.cudnn as cudnn
#import torch.distributed as dist
#import torch.optim as optim

#import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import torchvision
from torchvision import datasets, transforms#, models
#import os
import sys

from torch.autograd import Variable
#import torch.nn.functional as F

########################################################################################
########################################################################################
########################################################################################

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='DL19_FinalProject_PyTorch')

parser.add_argument('--model', type=str, default='densenet',
                    help='type of cnn ("resnet", "alexnet","vgg","squeezenet","densenet","inception")')
# parser.add_argument('--model-folder', type=str, default='/scratch/by783/DL_Final_models/',
#                     help='path to store model files')

# parser.add_argument('--model-file', type=str, default = '190425_raw_vggae_fromscratch_s.pt',
#                     help='path to autoencoder')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--save-folder', type=str, default='/scratch/by783/DL_Final_models/',
                    help='path to save the final model')

parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument("--feature-pinning", type=str, default='False',
                    help="pin all the conv layers.")
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--noise-level', type=float, default=0.3,
                    help='add noise to input')
# no noise added now
parser.add_argument('--dataset-path', type=str, default='/scratch/by783/DL_Final/ssl_data_96',
                    help='path to dataset')

args = parser.parse_args()

########################################################################################

model_name = args.model

# model_load_path = args.model_folder + args.model_file

save_path = args.save_folder + args.save

feature_pinning=str2bool(args.feature_pinning)
num_classes = args.num_classes

num_epochs = args.epochs
loader_batch_size = args.batch_size
loader_image_path = args.dataset_path
noise_level = args.noise_level


########################################################################################
########################################################################################
########################################################################################

class CDAutoEncoder(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.
    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """

    def __init__(self, input_size, output_size, conv_num=1, criterion=nn.MSELoss(), learning_rate=0.01):
        super(CDAutoEncoder, self).__init__()
        if conv_num == 2:
            self.forward_pass = nn.Sequential(
                nn.Conv2d(input_size, output_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_size, output_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            )
            self.backward_pass = nn.Sequential(
                nn.ConvTranspose2d(output_size, output_size, kernel_size=(2, 2), stride=(2, 2)),
                nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(output_size, input_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(input_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )
        if conv_num == 1:
            self.forward_pass = nn.Sequential(
                nn.Conv2d(input_size, output_size, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                nn.BatchNorm2d(output_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            )
            self.backward_pass = nn.Sequential(
                nn.ConvTranspose2d(output_size, input_size, kernel_size=(2, 2), stride=(2, 2)),
                nn.BatchNorm2d(input_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True)
            )

        self.criterion = criterion
        self.optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)

    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()
        # Add noise, but use the original lossless input as the target.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        #         print('forward: x: ',x.shape)
        y = self.forward_pass(x_noisy)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            #             print('forward: x_re: ',x_reconstruct.shape)
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return y.detach()

    def reconstruct(self, x):
        return self.backward_pass(x)


class StackedAutoEncoder(nn.Module):
    r"""
    A stacked autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self, criterion=nn.MSELoss(), learning_rate=0.01):
        super(StackedAutoEncoder, self).__init__()

        self.ae1 = CDAutoEncoder(3, 64, conv_num=1, criterion=criterion, learning_rate=learning_rate)
        self.ae2 = CDAutoEncoder(64, 128, conv_num=1, criterion=criterion, learning_rate=learning_rate)
        self.ae3 = CDAutoEncoder(128, 256, conv_num=2, criterion=criterion, learning_rate=learning_rate)
        self.ae4 = CDAutoEncoder(256, 512, conv_num=2, criterion=criterion, learning_rate=learning_rate)
        self.ae5 = CDAutoEncoder(512, 512, conv_num=2, criterion=criterion, learning_rate=learning_rate)

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)
        a4 = self.ae4(a3)
        a5 = self.ae5(a4)

        if self.training:
            return a5, self.reconstruct(a5)

        else:
            return a5, self.reconstruct(a5)

    def reconstruct(self, x):
        a4_reconstruct = self.ae5.reconstruct(x)
        a3_reconstruct = self.ae4.reconstruct(a4_reconstruct)
        a2_reconstruct = self.ae3.reconstruct(a3_reconstruct)
        a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
        x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
        return x_reconstruct

########################################################################################

def train_model(model, dataloaders, criterion, num_epochs=25):
    since = time.time()
    loss_history = []

    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        model.train()
        running_loss = 0.0
        for inputs, _ in dataloaders['unlabeled']:
            # optimizer.zero_grad()
            inputs = inputs.to(device)
            _, inputs_reconstructed = model(inputs)
            i_o_loss = criterion(inputs, inputs_reconstructed)
            running_loss += i_o_loss.item() * inputs.size(0)

            del i_o_loss, inputs_reconstructed, _



        epoch_loss = running_loss / len(dataloaders['unlabeled'].dataset)
        sys.stdout.write('Training time: {:.0f}s \n'.format(time.time() - since))
        sys.stdout.write('Training loss: {:.4f} \n'.format(epoch_loss))

        model.eval()

        eval_loss = 0.0
        for inputs, _ in dataloaders['train']:
            print('For check')

            inputs = inputs.to(device)

            _, inputs_reconstructed = model(inputs)

            loss = criterion(inputs, inputs_reconstructed)
            eval_loss += loss.item() * inputs.size(0)

            del loss, inputs_reconstructed, _

        epoch_eval_loss = eval_loss / len(dataloaders['train'].dataset)
        sys.stdout.write('Evaluation time: {:.0f}s \n'.format(time.time() - since))
        sys.stdout.write(' Eval loss: {:.4f} \n'.format(epoch_eval_loss))

        loss_history.append((epoch_loss, epoch_eval_loss))

        if epoch_eval_loss < best_loss:
            best_loss = epoch_eval_loss
            # best_model_wts = copy.deepcopy(model.state_dict())
            with open(save_path, 'wb') as f:
                torch.save(model, f)

        with open(save_path + '_loss', 'w') as f:
            for item in loss_history:
                f.write("unlabeled: %s, labeled: %s \n,  " % (item[0], item[1]))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, loss_history

########################################################################################

def image_loader(path, batch_size):
    transform = transforms.Compose(
        [
            #transforms.Resize(input_size),
            #transforms.CenterCrop(input_size),
            # use model fitted with the image size, so no need to resize
            transforms.ToTensor(),
            transforms.Normalize([0.502, 0.474, 0.426], [0.227, 0.222, 0.226])
            # https://pytorch.org/docs/stable/torchvision/transforms.html
            # [mean],[std] for different channels
        ]
    )
    sup_train_data = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised'), transform=transform)
    sup_val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=transform)
    unsup_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transform=transform)
    # source code: https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    # Main idea:
    data_loader_sup_train = torch.utils.data.DataLoader(
        sup_train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    data_loader_sup_val = torch.utils.data.DataLoader(
        sup_val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    data_loader_unsup = torch.utils.data.DataLoader(
        unsup_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    print('sup_train_data.class_to_idx==sup_val_data.class_to_idx: ',
          sup_train_data.class_to_idx == sup_val_data.class_to_idx)

    return data_loader_sup_train, data_loader_sup_val, data_loader_unsup, sup_train_data.class_to_idx

########################################################################################
########################################################################################
########################################################################################



sys.stdout.write("PyTorch Version: {}\n".format(torch.__version__))
sys.stdout.write("Torchvision Version: {}\n ".format(torchvision.__version__))

if torch.cuda.is_available():
    sys.stdout.write('GPU mode \n')
else:
    sys.stdout.write('Warning, CPU mode, pls check\n')


##########################################################################################

####### load data ####

sys.stdout.write('\nBegin to load data...\n')

dataloaders={}

dataloaders['train'], dataloaders['val'], dataloaders['unlabeled'], class_to_idx_dict = image_loader(loader_image_path,loader_batch_size)

####### load model  ####

criterion_ft = nn.MSELoss()

learning_rate_ft = args.lr

model_ft = StackedAutoEncoder(criterion=criterion_ft,learning_rate=learning_rate_ft)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    model_ft = nn.DataParallel(model_ft)

model_ft = model_ft.to(device)

################################

sys.stdout.write('Begin to train...')

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders, criterion_ft, num_epochs=num_epochs)

sys.stdout.write('Finished')





