# Imports
import torch
import torchvision # torch package for vision related things
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import DataLoader  # Gives easier dataset managment by creating mini batches etc.
from tqdm import tqdm  # For nice progress bar!


class CNN(nn.Module):
    def __init__(self, input_feature, drop_rate):
        super(CNN, self).__init__()
        self.conv_sub_block= nn.Sequential(
            nn.Conv2d((3,3))
        )
    def conv_sub_block(self, input_layer, feature_input, name):
        layer=nn.Conv2d(feature_input)
    def forward(self, x):
        pass

def get_branch(prefix, dropout_rate=0.5):
    # input layer
    x_input = Input(shape=(1, 32, 32), dtype='float32', name=prefix+'_input')
    # shallow layers
    x = Conv2D(20, (3, 3), name=prefix+'_conv_1')(x_input)
    x = BatchNormalization(name=prefix+'_batch_norm_1')(x)
    x = PReLU(name=prefix+'_prelu_1')(x)
    x = Conv2D(20, (3, 3), name=prefix+'_conv_2')(x)
    x = BatchNormalization(name=prefix+'_batch_norm_2')(x)
    x = PReLU(name=prefix+'_prelu_2')(x)
    x = MaxPool2D((2, 2), name=prefix+'_max_pool_1')(x)
    x = Conv2D(40, (3, 3), name=prefix+'_conv_3')(x)
    x = BatchNormalization(name=prefix+'_batch_norm_3')(x)
    x = PReLU(name=prefix+'_prelu_3')(x)
    # deep layers
    x = Conv2D(40, (3, 3), name=prefix+'_conv_4')(x)
    x = BatchNormalization(name=prefix+'_batch_norm_4')(x)
    x = PReLU(name=prefix+'_prelu_4')(x)
    x = MaxPool2D((2, 2), name=prefix+'_max_pool_2')(x)
    x = Conv2D(60, (3, 3), name=prefix+'_conv_5')(x)
    x = BatchNormalization(name=prefix+'_batch_norm_5')(x)
    x = PReLU(name=prefix+'_prelu_5')(x)
    x = Dropout(dropout_rate, name=prefix+'_dropout_1')(x)
    x = Flatten(name=prefix+'_flatten')(x)
    x = Dense(180, name=prefix+'_dense_1')(x)
    x_output = PReLU(name=prefix+'_prelu_6')(x)
    return Model(name=prefix+'_branch', inputs=[x_input], outputs=[x_output])
