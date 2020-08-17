import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from utils.calc_mean_std import online_mean_and_std


train_mean = (0.440985, 0.390349, 0.438721)
train_std = (0.248148, 0.230837, 0.237781)


def get_train_dataset(data_dir):
    transform = transforms.Compose(
        [transforms.Resize((48, 48)),
         transforms.ToTensor(),
         transforms.Normalize(train_mean, train_std)])

    img_data = torchvision.datasets.ImageFolder(data_dir, transform=transform)
    return img_data


def get_train_and_val_dataset(train_dir, train_ratio):
    resize = transforms.Compose(
        [transforms.Resize((48, 48)),
         transforms.ToTensor()])
    img_data = torchvision.datasets.ImageFolder(train_dir, transform=resize)
    train_size = int(train_ratio * len(img_data))
    validation_size = len(img_data) - train_size
    train_set, validation_set = torch.utils.data.random_split(img_data, [train_size, validation_size])
    mean, std = online_mean_and_std(train_set)
    transform = transforms.Compose(
        [transforms.Resize((48, 48)),
         transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    img_data.transform = transform
    img_data.transforms.transform = transform
    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=50, shuffle=True)
    # validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=50, shuffle=True)
    return train_set, validation_set
