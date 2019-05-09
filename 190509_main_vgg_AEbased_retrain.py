from __future__ import print_function
from __future__ import division


import argparse
import time
import sys
import copy
# import warnings
# import random
# import shutil
# import numpy as np
# import os


import torch
import torch.nn as nn
import torch.nn.parallel
# import torch.nn.functional as F

# import torch.backends.cudnn as cudnn
# import torch.distributed as dist
# import torch.optim as optim
# import torch.multiprocessing as mp

import torch.utils.data
import torch.utils.data.distributed

import torchvision
from torchvision import datasets, transforms, models


from torch.autograd import Variable

###############################################################################################
################################### Report Environment ########################################
###############################################################################################

sys.stdout.write("PyTorch Version: {}\n".format(torch.__version__))
sys.stdout.write("Torchvision Version: {}\n".format(torchvision.__version__))

if torch.cuda.is_available():
    sys.stdout.write('GPU mode \n')
else:
    sys.stdout.write('Warning, CPU mode, pls check\n')

############################################################################################
######################################     Read Args     ###################################
############################################################################################

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='DL19_FinalProject_PyTorch')

parser.add_argument('--model', type=str, default='vgg',
                    help='type of cnn ("resnet", "alexnet","vgg","squeezenet","densenet","inception")')

parser.add_argument('--model-folder', type=str, default='/beegfs/by783/DL_Final_models/',
                    help='path to store model files')

parser.add_argument('--model-file', type=str, default = '190425_raw_vggae_fromscratch_s.pt',
                    help='path to autoencoder')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--save-folder', type=str, default='/beegfs/by783/DL_Final_models/',
                    help='path to save the final model')

parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')
parser.add_argument("--feature-pinning", type=str, default='True',
                    help="pin all the conv layers.")
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum')
# parser.add_argument('--noise-level', type=float, default=0.3,
#                     help='add noise to input')
# no noise added now
parser.add_argument('--dataset-path', type=str, default='/beegfs/by783/DL_Final/ssl_data_96',
                    help='path to dataset')

args = parser.parse_args()
#args=parser.parse_args("--model vgg --AE-file XXXXXXXX --batch-size 512 --feature-pinning True --save 190505_try2 --epochs 50 --lr 0.001 ".split())
########################################################################################

model_name = args.model

model_load_path = args.model_folder + args.model_file

save_path = args.save_folder + args.save

feature_pinning=str2bool(args.feature_pinning)
num_classes = args.num_classes

num_epochs = args.epochs
loader_batch_size = args.batch_size
loader_image_path = args.dataset_path
# noise_level = args.noise_level

######################################################################################################
############################################  Classes  ################################################
######################################################################################################


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

class AE_Transfered_Network(torch.nn.Module):
    def __init__(self, classifier_type, autoencoder_model, num_classes=1000):
        super(AE_Transfered_Network, self).__init__()
        if classifier_type != 'vgg':
            sys.stdout.write('Dear, we only support vgg now...\n')

        self.features = nn.Sequential(*copy.deepcopy(
            list(autoencoder_model.ae1.forward_pass.children()) +
            list(autoencoder_model.ae2.forward_pass.children()) +
            list(autoencoder_model.ae3.forward_pass.children()) +
            list(autoencoder_model.ae4.forward_pass.children()) +
            list(autoencoder_model.ae5.forward_pass.children())))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))
        self.classifier = nn.Sequential(
            nn.Linear(in_features=4608, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

########################################################################################
##########################################  Functions   ##############################################
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

######################################################################################################
def set_parameter_pin_grad(model, pinning):
    if pinning:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

######################################################################################################

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            # 切换phase重置loss
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            sys.stdout.write('{} Loss: {:.4f} Acc: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))
            sys.stdout.write('training time: {:.0f}s\n'.format( time.time() - since ))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    #best_model_wts = copy.deepcopy(model.state_dict())
                    with open(save_path, 'wb') as f:
                        torch.save(model, f)
                #else:
                    #lr/=4
                # 只有phase为val的acc loss才被加入 val_acc
                val_acc_history.append(epoch_acc)
                with open(save_path+'_val_acc', 'w') as f:
                    for item in val_acc_history:
                        f.write("%s\n" % item)

        print()

    time_elapsed = time.time() - since
    sys.stdout.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    sys.stdout.write('Best val Acc: {:4f}\n'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


########################################################################################
##########################################  Main   ##############################################
########################################################################################

sys.stdout.write('Begin to load data...\n')

dataloaders={}

dataloaders['train'], dataloaders['val'], dataloaders['unlabeled'], class_to_idx_dict = image_loader(loader_image_path,loader_batch_size)

#####################################   load model  ###############################
'''
try:
    model_ae=torch.load(model_load_path).module
except:
    model_ae=torch.load(model_load_path)
'''

#model_ae = StackedAutoEncoder()

#model_ft = AE_Transfered_Network('vgg',model_ae)

try:
    model_ft=torch.load(model_load_path).module
except:
    model_ft=torch.load(model_load_path)

del model_ae #save memory

set_parameter_pin_grad(model_ft,False) # free all layers for fine tuning

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    model_ft = nn.DataParallel(model_ft)

model_ft = model_ft.to(device)

########################################################

sys.stdout.write("Params to learn:\n")
params_to_update = []
for name,param in model_ft.named_parameters():
    if param.requires_grad == True:
        params_to_update.append(param)
        sys.stdout.write("\t{}\n".format(name))

########################################################

##########################################  set training parameters ##########################################

# Observe that all parameters are being optimized
optimizer_ft = torch.optim.SGD(params_to_update, lr=args.lr, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()
############################################ training ###########################
sys.stdout.write('Begin to train...\n')
# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

sys.stdout.write('Finished')