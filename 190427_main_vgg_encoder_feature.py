from __future__ import print_function
from __future__ import division

import numpy as np
import argparse
import random
import shutil
import time
import warnings
import os
import sys
import copy

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

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

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
parser.add_argument('--folder-path', type=str, default='/scratch/by783/DL_Final_models/',
                    help='path to store model files')
parser.add_argument('--AE-file', type=str, default = '190425_raw_vggae_fromscratch_s.pt',
                    help='path to autoencoder')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--num-classes', type=int, default=1000,
                    help='number of classes')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument('--batch-size', type=int, default=512,
                    help='batch size')
parser.add_argument("--feature-pinning", type=str, default='True',
                    help="pin the feature layers of CNN")
parser.add_argument('--dataset-path', type=str, default='/scratch/by783/DL_Final/ssl_data_96',
                    help='path to dataset')
args = parser.parse_args()

####################### input params ##################################
model_name = args.model
model_folder_path = args.folder_path
save_path = model_folder_path + args.save

AE_load_path = model_folder_path + args.AE_file

feature_pinning=str2bool(args.feature_pinning)
num_classes = args.num_classes

num_epochs = args.epochs
loader_batch_size = args.batch_size
loader_image_path = args.dataset_path


###################### report environment ###################################


sys.stdout.write("PyTorch Version: {}".format(torch.__version__))
sys.stdout.write("Torchvision Version: ".format(torchvision.__version__))

if torch.cuda.is_available():
    sys.stdout.write('GPU mode')
else:
    sys.stdout.write('Warning, CPU mode, pls check')


###################################### functions ###############################
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


# https://stackoverflow.com/questions/37837682/python-class-input-argument/37837766
# https://github.com/awentzonline/pytorch-cns/blob/master/examples/vggmse.py

class Model_Based_Autoencoder(torch.nn.Module):
    def __init__(self, model_name, pretrained):
        super(Model_Based_Autoencoder, self).__init__()
        if model_name != 'vgg':
            sys.stdout.write('Dear, we only support vgg now...\n')

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


# vgg model website: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
class AE_Transfered_Network(torch.nn.Module):
    def __init__(self, classifier_type, autoencoder_model, num_classes=1000):
        super(AE_Transfered_Network, self).__init__()
        if classifier_type != 'vgg':
            sys.stdout.write('Dear, we only support vgg now...\n')

        self.features = copy.deepcopy(autoencoder_model.encoder)
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

def set_parameter_pin_grad(model, pinning):
    if pinning:
        for param in model.parameters():
            param.requires_grad = False
    else:
        for param in model.parameters():
            param.requires_grad = True

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

########################################## set the model ##########################################
sys.stdout.write('Begin to load data...\n')
# make sure model and input size
model_ae = torch.load(AE_load_path)
model_ft = AE_Transfered_Network('vgg',model_ae.module, num_classes=num_classes)
# pin the feature layers ##
set_parameter_pin_grad(model_ft.features,feature_pinning)
# send model to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if torch.cuda.device_count() > 1:
    model_ft = nn.DataParallel(model_ft)

model_ft = model_ft.to(device)

########################################################################################


########################################## load data, input_size is used ##########################################

sys.stdout.write('Begin to load data...\n')

dataloaders={}

dataloaders['train'], dataloaders['val'], dataloaders['unlabeled'], class_to_idx_dict = image_loader(loader_image_path,loader_batch_size)


############################################# report weights to be trained ##########################################

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
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

##########################################  set training parameters ##########################################

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()
############################################ training ###########################
sys.stdout.write('Begin to train...\n')
# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))
um_epochs=num_epochs, is_inception=(model_name=="inception"))

sys.stdout.write('Finished')


