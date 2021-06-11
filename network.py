import torch
from torchvision import transforms, models
import argparse
import json
from get_input_args_predict import get_input_args
from utility import process_image
import argparse


def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    if architecture =='vgg19':
        model = models.vgg19(pretrained=True)
    elif architecture =='alexnet':
        model = models.alexnet(pretrained = True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
    
    nn.Sequential(nn.Linear(1024,hidden_units),
                              nn.ReLU(),

                              nn.Linear(hidden_units, 102),

                              nn.LogSoftmax(dim=1))

    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint ['class_to_idx']
    
    return model


def predict(image_path, model, category_names, topk, cpu_gpu):
    image = process_image(image_path)
    x = image.numpy()

    image = torch.from_numpy(x).float()
    
    image = image.unsqueeze(0)
    image = image.cuda()
    model.to(device)

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