from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import sys
import copy

#################################   set args  #######################
import argparse

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
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--epochs', type=int, default=25,
                    help='upper epoch limit')
parser.add_argument("--pretrained", type=str, default='True',
                    help="use pre-trained conv layers.")
parser.add_argument('--model-folder', type=str, default='/scratch/by783/DL_Final_models/',
                    help='path to store model files')

parser.add_argument('--model-file', type=str, default = '190425_raw_vggae_fromscratch_s.pt',
                    help='path to autoencoder')

args = parser.parse_args()

####################### input params ##################################
save_path='/beegfs/by783/DL_Final_models/'+args.save
model_name = args.model
num_epochs = args.epochs
feature_extract = str2bool(args.pretrained)
# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
###################### fixed_params ###################################

num_classes = 1000
loader_image_path='/beegfs/by783/DL_Final/ssl_data_96'
loader_batch_size=256




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
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False




def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0



    if model_name != "vgg":
        sys.stdout.write('We only have vgg now!!!')
    else:
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)

        ##########

        model_ft.avgpool = nn.AdaptiveAvgPool2d(output_size=(3, 3))

        model_ft.classifier[0] = nn.Linear(in_features=4608, out_features=4096, bias=True)
        model_ft.classifier[3] = nn.Linear(in_features=4096, out_features=4096, bias=True)
        model_ft.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes, bias=True)


        input_size = 96

    return model_ft, input_size



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

            sys.stdout.write('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            sys.stdout.write('training time: {:.0f}s'.format( time.time() - since ))

            # deep copy the model
            if phase == 'val':
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    #best_model_wts = copy.deepcopy(model.state_dict())
                    with open(save_path, 'wb') as f:
                        torch.save(model, f)
                #else:
                    #lr/=4
                val_acc_history.append(epoch_acc)
                with open(save_path+'_val_acc', 'w') as f:
                    for item in val_acc_history:
                        f.write("%s\n" % item)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    return model, val_acc_history



####### make sure model and input size

model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

model_ft = torch.load(args.model_folder+args.model_file)

for param in model_ft.parameters():
    param.requires_grad = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = model_ft.to(device)

########


####### load data, input_size is used ####

sys.stdout.write('Begin to load data...')

dataloaders={}

dataloaders['train'], dataloaders['val'], data_loader_unsup, class_to_idx_dict = image_loader(loader_image_path,loader_batch_size)

######

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
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
optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

sys.stdout.write('Begin to train...')

# Train and evaluate
train_model(model_ft, dataloaders, criterion, optimizer_ft, num_epochs=num_epochs, is_inception=(model_name=="inception"))

sys.stdout.write('Finished')


