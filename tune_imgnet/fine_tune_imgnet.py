import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import numpy as np
import time
import os
import sys
root_path = os.path.join(os.getenv("HOME"), "vqa2019")
sys.path.append(root_path)
import copy
from tuneDataset import TuneDataset
import config

# # for resnet152
# model_save_path = os.path.join(root_path, "models/pretrained/emt_tuned_resnet.pt")

# for densenet121
model_save_path = os.path.join(root_path, "models/pretrained/etm_tuned_densenet.pt")

opt = config.parse_opt()


def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for step, (inputs, labels) in enumerate(tune_loader):
                # inputs = inputs.to(device)
                # labels = labels.to(device)
                inputs = Variable(inputs).cuda().float()
                labels = Variable(labels).cuda().long()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)

            # epoch_loss = running_loss / dataset_sizes[phase]
            # epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_loss = running_loss / len(tune_data)
            # epoch_acc = running_corrects.double() / len(tune_dataset)

            print('{} Loss: {:.4f}: '.format(
                phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



torch.cuda.set_device(opt.TRAIN_GPU_ID)

# # define transforms method
# data_transforms = transforms.Compose([
# 	transforms.Resize(224), 
# 	transforms.ToTensor()
# 	])

# tune_data = TuneDataset(data_transforms)
tune_data = TuneDataset(opt, "train")
tune_loader = data.DataLoader(dataset=tune_data, shuffle=True, batch_size=32)

# model_conv = torchvision.models.resnet152(pretrained=True)
model_conv = torchvision.models.densenet121(pretrained=True)

# fine tune the model
for param in model_conv.parameters():
	param.requires_grad = True

# # fine tune last convolutional layer
# for param in model_conv.layer4[2].conv3.parameters():
# 	param.requires_grad = True

# # for resnet152
# num_ftrs = model_conv.fc.in_features
# model_conv.fc = nn.Linear(num_ftrs, opt.ETM_TOP_NUM)

# for densenet121
model_conv.classifier = nn.Linear(1024, opt.ETM_TOP_NUM)

model_conv.cuda()

# tune_params = list(model_conv.fc.parameters()) + list(model_conv.layer4[2].conv3.parameters())
tune_params = model_conv.parameters()
optimizer = torch.optim.Adam(tune_params, lr=0.0001)
loss_func = nn.CrossEntropyLoss()
decay = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

tuned_model_conv = train_model(model_conv, loss_func, optimizer, decay, num_epochs=100)

# save the tuned model to the models folder
torch.save(tuned_model_conv, model_save_path)