import torch
import torchvision # torch package for vision related things
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation

def get_loader(phase='train', batch_size=64):
    train = True
    shuffle = True
    if phase != 'train':
        train = False
    if phase == 'test':
        # not shuffle in test data
        shuffle = False

    # preprocess data
    # MNIST dataset
    data_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=train,
                                               transform=transforms.ToTensor(),
                                               download=True)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=data_dataset,
                                               batch_size=batch_size,
                                           shuffle=shuffle)
    return data_loader
