# Imports
import torch
from torch import nn  # All neural network modules
from collections import OrderedDict

class sub_cort_model(nn.Module):
    def __init__(self, input_feature, drop_rate):
        super(sub_cort_model, self).__init__()
        # input: N x channels_img x 28 x 28
        # input_feature channels x 28 x 28
        self.input_feature = input_feature
        self.conv_channels = [20, 40, 60, 180]
        self.axial_branch = self.conv_branch(input_feature[0], 3, 2, drop_rate)
        self.sagital_branch = self.conv_branch(input_feature[0], 3, 2, drop_rate)
        self.coronal_branch = self.conv_branch(input_feature[0], 3, 2, drop_rate)
        self.fc1=nn.Linear(self.conv_channels[3]*3,540)
        self.prelu1=nn.PReLU()
        self.fc2=nn.Linear(540, 270)
        self.prelu2=nn.PReLU()
        self.fc3=nn.Linear(270, 15)
        self.prelu3=nn.PReLU()
        self.fc4=nn.Linear(15, 2)


    def forward(self, x,y,z):
        axial=self.axial_branch(x)
        sagital = self.sagital_branch(y)
        coronal = self.coronal_branch(z)
        layer=torch.cat((axial, sagital, coronal),dim=1)
        layer=self.prelu1(self.fc1(layer))#540
        layer=self.prelu2(self.fc2(layer))#270
        layer=self.prelu3(self.fc3(layer))#15
        layer=self.fc4(layer)#2
        return layer
        #return self.branch(x)

    def conv_branch(self, in_channels, kernel, pool_kernel, drop_rate):
        params = OrderedDict([
        ('conv_sub_block1', self.conv_sub_block(in_channels, self.conv_channels[0], kernel)),
        ('conv_sub_block2', self.conv_sub_block(self.conv_channels[0], self.conv_channels[0], kernel),),
        ('maxpol1', nn.MaxPool2d(pool_kernel)),
        ('conv_sub_block3', self.conv_sub_block(self.conv_channels[0], self.conv_channels[1], kernel)),
        ('conv_sub_block4', self.conv_sub_block(self.conv_channels[1], self.conv_channels[1], kernel)),
        ('maxpol2', nn.MaxPool2d(pool_kernel)),
        ('conv_sub_block5', self.conv_sub_block(self.conv_channels[1], self.conv_channels[2], kernel)),
        ('drop_out', nn.Dropout2d(drop_rate)),
        #('fc1_conv', nn.Conv2d(self.conv_channels[2], 1, kernel_size=kernel))
        ('flatten', nn.Flatten()),
        # try to find dynamic way to calculate flatten output size
        ('fc1',nn.Linear(240,self.conv_channels[3])),
        ('prelu',nn.PReLU())
        ])

        return nn.Sequential(params)

            #Flatten(),
            #nn.Linear()
    def conv_sub_block(self, in_channels, out_channels, kernel):
        params = OrderedDict([
            ('conv',nn.Conv2d(
                in_channels,
                out_channels,
                kernel
            ) ),
            ('batch_norm', nn.BatchNorm2d(out_channels) ),
            ('lrelu',nn.LeakyReLU(0.2))
        ])
        return nn.Sequential(params)
'''    

class Flatten(nn.Module):
    def forward(self, input):        
        out_= input.view(-1,1)
        print('input size',input.size(),'output size:',out_.size())
        return out_

 '''
class Linear(nn.Module):
    def __init__(self, out_size):
        super(Linear,self).__init__()
        self.out_size=out_size


    def forward(self, input):
        size= len(input)
        out_=nn.Linear(size,self.out_size)
        print('input size',size,'output size:',out_.out_size)
        return out_

