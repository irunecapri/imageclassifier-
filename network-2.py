import torch
from torchvision import transforms, models
import argparse
import json
from utility import process_image
import argparse
import torch
from torch import nn
    

def get_input_args():
    parser.add_argument('--hidden_units', action='store', help='# of hidden units to add to model', type=int, default=500)

def load_checkpoint(filepath, gpu, architecture):
    if torch.cuda.is_available() and gpu:
        filepath = filepath.cuda()
        checkpoint = torch.load(filepath)
    
    architecture = checkpoint['arch']
    
    if architecture =='vgg19':
        model = models.vgg19(pretrained=True)
    elif architecture =='alexnet':
        model = models.alexnet(pretrained = True)
    elif architecture == 'densenet121':
        model = models.densenet121(pretrained=True)
        
    
    
    
   
   

    model.class_to_idx = checkpoint['class_to_idx'] 
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def predict(image_path, model, top_k, gpu):
    if torch.cuda.is_available() and gpu:
        image_path = image_path.cuda()
 
        model = model.cuda()
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
    for item in classes:
            flower_Names.append(cat_to_name_dict[item])
    
    return probs, classes