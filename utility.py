import numpy as np
import matplotlib.pyplot as plt
import os, random
import json
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import time

def read(filename):
    with open(filename, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name


def load_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/tes
    
    
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    
    validation_transforms = transforms.Compose([transforms.Resize(256),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406], 
                                                                 [0.229, 0.224, 0.225])])
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)
    
    
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    vloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)
    
    
    return trainloader, testloader, vloader, train_data


def process_image(ima):
   
    
    ima = Image.open(ima)
    ima = ima.resize((256, 256))
    ima = ima.crop((0, 0, 224, 224))
    np_image = np.array(ima) / 255
    mean = np.array([0.485, 0.456, 0.406])
    stdv = np.array([0.229, 0.224, 0.225])
    ima = (np_image - mean) / stdv
    f_image = ima.transpose((2, 1, 0))

    return torch.from_numpy(f_image)



model.class_to_idx = image_datasets[0].class_to_idx


def predict(image_path, model, topk=5):
 
    
    image = process_image(image_path)
    x = image.numpy()

    image = torch.from_numpy(x).float()
    
    image = image.unsqueeze(0)
    image = image.cuda()
    model.to("cuda")

    output = model.forward(image)
    
    ps = torch.exp(output).data
 
    largest = ps.topk(topk)
    prob = largest[0].cpu().numpy()[0]
    idx = largest[1].cpu().numpy()[0]
    classes = []
    idx = largest[1].cpu().numpy()[0]
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    for i in idx:
        classes.append(idx_to_class[i])




    return prob, classes