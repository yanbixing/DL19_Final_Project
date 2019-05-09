from __future__ import print_function
from __future__ import division

import numpy as np
import argparse
import random
import shutil
import time
import warnings


import torch
import torch.nn as nn

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim

import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed

import torchvision
from torchvision import datasets, models, transforms
import os
import re
import sys
import copy

##############################################################################
##########################   Function    ##############################
##############################################################################





#################################   set args  #######################


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
#
# parser.add_argument('--model-file', type=str, default = '190425_raw_vggae_fromscratch_s.pt',
#                     help='path to autoencoder')

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
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument("--feature-pinning", type=str, default='False',
                    help="pin all the conv layers.")
parser.add_argument('--noise-level', type=float, default=0.3,
                    help='add noise to input')
# no noise added now
parser.add_argument('--dataset-path', type=str, default='/beegfs/by783/DL_Final/ssl_data_96',
                    help='path to dataset')
args = parser.parse_args()

############################################

model_name = args.model

#model_load_path = args.model_folder + args.model_file

save_path = args.save_folder + args.save

feature_pinning=str2bool(args.feature_pinning)
num_classes = args.num_classes

num_epochs = args.epochs
loader_batch_size = args.batch_size
loader_image_path = args.dataset_path
noise_level = args.noise_level


##################
def image_loader(path, batch_size):
    transform = {
        'train':transforms.Compose([
            # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15),
            # transforms.RandomAffine(15),
            # transforms.RandomResizedCrop(size=224,scale=(0.8, 1.0)),
            # transforms.ToTensor(),
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.502, 0.474, 0.426], [0.227, 0.222, 0.226])
        ]),
        'val':transforms.Compose([
            # transforms.Resize(256),
            # transforms.CenterCrop(224),
            # transforms.ToTensor(),
            # transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.502, 0.474, 0.426], [0.227, 0.222, 0.226])
        ])
    }
    sup_train_data = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised'), transform=transform['train'])
    sup_val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=transform['val'])
    #unsup_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transform=transform)
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
    '''
    data_loader_unsup = torch.utils.data.DataLoader(
        unsup_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    print('sup_train_data.class_to_idx==sup_val_data.class_to_idx: ',
          sup_train_data.class_to_idx == sup_val_data.class_to_idx)
    '''
    return data_loader_sup_train, data_loader_sup_val#, data_loader_unsup, sup_train_data.class_to_idx

########################

def pin_features(model, pinning):
    if pinning:
        for param in model.parameters():
            param.requires_grad = False
        model.classifier.weight.requires_grad = True

    else:
        for param in model.parameters():
            param.requires_grad = True
#############################

def initialize_model(model_name, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None



    if model_name != "vgg":
        sys.stdout.write('We only have vgg now!!!')
    else:
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        #set_parameter_requires_grad(model_ft, feature_extract)

        ##### we want to train all parameters #####

        for model_ft_param in model_ft.parameters():
            model_ft_param.requires_grad = True

        ##### we want to train all parameters #####

        ##########

        model_ft.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))

        model_ft.classifier[0] = nn.Linear(in_features=4608, out_features=4096, bias=True)
        model_ft.classifier[3] = nn.Linear(in_features=4096, out_features=4096, bias=True)
        model_ft.classifier[6] = nn.Linear(in_features=4096, out_features=1000, bias=True)


    return model_ft

################################

def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()
    val_acc_history = []
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            # 切换phase重置loss
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                print('batch_size checker')

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
                        loss = loss1 + 0.4 * loss2
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
            sys.stdout.write('training time: {:.0f}s\n'.format(time.time() - since))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    with open(save_path, 'wb') as f:
                        torch.save(model, f)

                val_acc_history.append(epoch_acc)
                with open(save_path + '_val_acc', 'w') as f:
                    for item in val_acc_history:
                        f.write("%s\n" % item)

        print()

    time_elapsed = time.time() - since
    sys.stdout.write('Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    sys.stdout.write('Best val Acc: {:4f}\n'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history



##############################################################################
##########################   Main program    ##############################
##############################################################################

###################### report environment ###################################


sys.stdout.write("PyTorch Version: {}\n".format(torch.__version__))
sys.stdout.write("Torchvision Version:{}\n ".format(torchvision.__version__))

if torch.cuda.is_available():
    sys.stdout.write('GPU mode\n')
else:
    sys.stdout.write('Warning, CPU mode, pls check')

####### load data, input_size is used ####

dataloaders={}

#dataloaders['train'], dataloaders['val'], data_loader_unsup, class_to_idx_dict = image_loader(loader_image_path,loader_batch_size)

dataloaders['train'], dataloaders['val'] = image_loader(loader_image_path,loader_batch_size)


##########################################

# model_ft = torch.load(model_load_path)

model_ft = initialize_model('vgg')

#pin_features(model_ft, feature_pinning)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('number of GPUS:', torch.cuda.device_count())

if torch.cuda.device_count() > 1:
    model_ft = nn.DataParallel(model_ft)

model_ft = model_ft.to(device)

################ report trained params

params_to_update = model_ft.parameters()
sys.stdout.write("Params to learn:\n")
if feature_pinning:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            sys.stdout.write("\t{}\n".format(name))
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            sys.stdout.write("\t{}\n".format(name))

#################

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()
############################################ training ###########################
sys.stdout.write('Begin to train...\n')
# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

sys.stdout.write('Finished')
