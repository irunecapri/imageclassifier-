import torch
import time
import numpy as np
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
from collections import OrderedDict
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
import argparse
import json


from utility import load_data


def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', action='store', help='directory containing images')
    parser.add_argument('--save_dir', action='store', help='save trained checkpoint to this directory' )
    parser.add_argument('--arch', action='store', help='what kind of pretrained architecture to use', default='vgg19')
    parser.add_argument('--gpu', action='store_true', help='use gpu to train model')
    parser.add_argument('--epochs', action='store', help='# of epochs to train', type=int, default=4)
    parser.add_argument('--lr', action='store', help='which learning rate to start with', type=float, default=0.001)
    parser.add_argument('--hidden_units', action='store', help='# of hidden units to add to model', type=int, default=500)
    parser.add_argument('--output_size', action='store', help='# of classes to output', type=int, default=102)
    
    return parser.parse_args()



def main():
  
    in_arg = get_input_args()
    
    start_time = time.time()
    
    trainloader, testloader, vloader, train_data = load_data(in_arg.data_dir)
    
    model = get_model(in_arg.arch)
        
    model = load_model(model, in_arg.arch, in_arg.hidden_units, in_arg.lr, in_arg.gpu)

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=in_arg.lr)
    
    
    train(model, in_arg.epochs, in_arg.lr, criterion, optimizer, trainloader, vloader,in_arg.gpu, start_time)
   
    print(f"Time to train and validate model: {(time.time() - start_time):.3f} seconds")

    save_checkpoint(in_arg.save_dir, model, optimizer, in_arg.epochs, in_arg.arch, image_datasets, in_arg.lr)

def get_model(arch):
    
     if arch == 'vgg19':
        model = models.vgg19(pretrained=True)
     elif arch =='alexnet':
        model = models.alexnet(pretrained = True)
     elif arch == 'densenet121': 
        model = models.densenet121(pretrained = True)
     return model




def load_model(model, arch, hidden_units, lr, gpu):
    if arch == 'vgg19': 
        input_size = 25088
    elif arch == 'alexnet':
        input_size = 9216
    elif arch == 'densenet121':
        input_size = 1024
    output_size = 102
    
    
    for param in model.parameters():
        param.requires_grad = False
    classifier= nn.Sequential(nn.Linear(input_size,hidden_units), nn.ReLU(), nn.Linear(hidden_units, 102), nn.LogSoftmax(dim=1))
    model.classifier = classifier
    criterion = nn.NLLLoss()
    
    return model

def train(model, epochs, lr, criterion, optimizer, trainloader, vloader, gpu, start_time): 
    device = torch.device('cuda' if torch.cuda.is_available() and gpu else 'cpu')   
    model.train()
    epochs= epochs
    steps=0
    running_loss=0
    print_every=20
    print(device)
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps+=1
            inputs, labels =inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            
            if steps % print_every==0:
                test_loss=0
                accuracy=0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in  vloader:
                        inputs, labels=inputs.to(device), labels.to(device)
                        logps =model.forward(inputs)
                        batch_loss=criterion(logps, labels)
                        test_loss+=batch_loss.item()
                        ps=torch.exp(logps)
                        top_p, top_class=ps.topk(1, dim=1)
                        equals = top_class==labels.view(*top_class.shape)
                        accuracy+= torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epochs}.."
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss:{test_loss/len(vloader):.3f}.."
                      f"Validation accuracy: {accuracy/len(vloader):.3f}")  



                
  

            
            
            
def save_checkpoint(save_dir, model, optimizer, epochs, arch, image_datasets, lr):
    model.cpu()
    model.class_to_idx = image_datasets[0].class_to_idx
    checkpoint = {'output_size' : 102,
                  'optimizer': optimizer,
                  'arch': arch,
                  'state_dict': model.state_dict(),
                  'optimizer_state': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx}
   
    torch.save(checkpoint, 'model_checkpoint.pth')
    
       
   


if __name__ == "__main__":
    main()































