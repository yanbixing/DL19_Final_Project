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
import sys
import copy

#################################   set args  #######################


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
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument("--pretrained", type=str, default='True',
                    help="use pre-trained conv layers.")

args = parser.parse_args()

####################### input params ##################################
save_path='/scratch/by783/DL_Final_models/'+args.save
model_name = args.model
num_epochs = args.epochs
pin_encoder = str2bool(args.pretrained)
loader_batch_size=args.batch_size
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
###################### fixed_params ###################################
num_classes = 1000
loader_image_path='/scratch/by783/DL_Final/ssl_data_96'



sys.stdout.write("PyTorch Version: {}".format(torch.__version__))
sys.stdout.write("Torchvision Version: ".format(torchvision.__version__))

if torch.cuda.is_available():
    sys.stdout.write('GPU mode')
else:
    sys.stdout.write('Warning, CPU mode, pls check')



def image_loader(path, batch_size):
    transform = transforms.Compose(
        [
            #transforms.Resize(input_size),
            #transforms.CenterCrop(input_size),
            # use model fitted with the image size, so no need to resize
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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


# used in initialize model
def set_parameter_pin_grad(model, pinning):
    if pinning:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True


class Model_Based_Autoencoder(torch.nn.Module):
    def __init__(self, model_name, pretrained):
        super(Model_Based_Autoencoder, self).__init__()
        if model_name != 'vgg':
            sys.stdout.write('Dear, we only support vgg now...')

        self.encoder = models.vgg11_bn(pretrained=pretrained).features
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),  # de-conv8
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # de-conv7
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 512, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),  # de-conv6
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # de-conv5
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),  # de-conv4
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),  # de-conv3
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),  # de-conv2
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0)),  # de-conv1
            nn.BatchNorm2d(3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def initialize_model(model_name, pin_encoder, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0



    if model_name != "vgg":
        sys.stdout.write('We only have vgg now!!!')
    else:
        """ VGG11_bn
        """
        model_ft = Model_Based_Autoencoder(model_name, pretrained=use_pretrained)
        set_parameter_pin_grad(model_ft.encoder, pin_encoder)

        input_size = 96

    return model_ft, input_size


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    # unsupervised learning, we do not need train and vals
    since = time.time()
    loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        ################# train the model on unsupervised data ############
        model.train()

        running_loss = 0.0

        for inputs, _ in dataloaders['unlabeled']:
            inputs = inputs.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            running_loss += loss.item() * inputs.size(0)

            loss.backward()
            optimizer.step()

        epoch_loss = running_loss / len(dataloaders['unlabeled'].dataset)
        sys.stdout.write('Training time: {:.0f}s'.format(time.time() - since))
        sys.stdout.write('Training loss: {:.4f}'.format(epoch_loss))
        ################# evaluate the model performance on labeled data ############

        model.eval()

        eval_loss = 0.0
        for inputs, _ in dataloaders['labeled']:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            eval_loss += loss.item() * inputs.size(0)

        epoch_eval_loss = eval_loss / len(dataloaders['labeled'].dataset)
        sys.stdout.write('Evaluation time: {:.0f}s'.format(time.time() - since))
        sys.stdout.write(' Eval loss: {:.4f}'.format(epoch_eval_loss))

        #################

        loss_history.append((epoch_loss, epoch_eval_loss))

        if epoch_eval_loss < best_loss:
            best_loss = epoch_eval_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            with open(save_path, 'wb') as f:
                torch.save(model, f)

        with open(save_path + '_loss', 'w') as f:
            for item in loss_history:
                f.write("unlabeled: %s, labeled: %s \n,  " % (item[0], item[1]))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, loss_history



####### make sure model and input size

model_ft, input_size = initialize_model(model_name, pin_encoder, use_pretrained=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    model_ft = nn.DataParallel(model_ft)

model_ft = model_ft.to(device)

########


####### load data, input_size is used ####

sys.stdout.write('Begin to load data...')

dataloaders={}

dataloaders['labeled'], data_loader_val, dataloaders['unlabeled'], class_to_idx_dict = image_loader(loader_image_path,loader_batch_size)


######

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if pin_encoder:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            sys.stdout.write("\t{}".format(name))
else:
    for name,param in model_ft.named_parameters():
        if param.requires_grad == True:
            sys.stdout.write("\t{}".format(name))

# Observe that all parameters are being optimized
criterion = nn.MSELoss()

learning_rate=0.001
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=learning_rate, weight_decay=1e-5)

# Setup the loss fxn

sys.stdout.write('Begin to train...')

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs)

sys.stdout.write('Finished')


