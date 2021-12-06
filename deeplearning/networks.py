import numpy as np

# Pytorch libraries
import torch.nn as nn
import torch.nn.functional as F

class LeNet_5(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(LeNet_5, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        self.fc_input_size = int(self.input_size/4)**2 * 64
        
        self.conv1 = nn.Conv2d(self.in_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm2d(32, track_running_stats=False, affine=False)
        self.mp1= nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm2d(64, track_running_stats=False, affine=False)
        self.mp2= nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(self.fc_input_size, 512)
        self.bn3 = nn.BatchNorm1d(512, track_running_stats=False, affine=False)
        self.fc2 = nn.Linear(512, out_dims)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mp1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp2(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn3(self.fc1(x)))
        x = self.fc2(x)
        return x

class VGG_7(nn.Module):
    def __init__(self, in_dims, in_channels, out_dims=10):
        super(VGG_7, self).__init__()
        self.in_channels = in_channels
        self.input_size = int(np.sqrt(in_dims/in_channels))
        self.fc_input_size = int(self.input_size/8)**2 * 512
        
        self.conv1_1 = nn.Conv2d(self.in_channels, 128, kernel_size=3, padding=1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.conv1_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(128, track_running_stats=False, affine=False)
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        self.conv2_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False)
        self.bn2_2 = nn.BatchNorm2d(256, track_running_stats=False, affine=False)
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.conv3_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False)
        self.bn3_2 = nn.BatchNorm2d(512, track_running_stats=False, affine=False)
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)        

        self.fc1 = nn.Linear(self.fc_input_size, 1024)
        self.bn4 = nn.BatchNorm1d(1024, track_running_stats=False, affine=False)
        self.fc2 = nn.Linear(1024, out_dims)

    # 32C3 - MP2 - 64C3 - Mp2 - 512FC - SM10c
    def forward(self, x):
        x = x.view(x.shape[0], self.in_channels, self.input_size, self.input_size)
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.mp1(x)

        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.mp2(x)

        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.mp3(x)
        
        x = x.view(x.shape[0], -1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.fc2(x)
        return x

def init_weights(module, init_type='kaiming', gain=0.01):
    '''
    initialize network's weights
    init_type: normal | uniform | kaiming  
    '''
    classname = module.__class__.__name__
    if hasattr(module, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
        if init_type == 'normal':
            nn.init.normal_(module.weight.data, 0.0, gain)
        elif init_type == "uniform":
            nn.init.uniform_(module.weight.data)
        elif init_type == 'kaiming':
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')

        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find('BatchNorm') != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)

    elif (classname.find("GroupNorm") != -1 and module.weight is not None):
        nn.init.normal_(module.weight.data, 1.0, gain)
        nn.init.constant_(module.bias.data, 0.0)