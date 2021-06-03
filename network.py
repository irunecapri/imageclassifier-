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


import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
    
    
    
device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=models.densenet121(pretrained=True)
for param in model.parameters():
 
    param.requires_grad = False
 
classifier= nn.Sequential(nn.Linear(1024,500),
                           nn.ReLU(),
                           nn.Linear(500, 102),
                           nn.LogSoftmax(dim=1))

model.classifier = classifier
model
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
model.to(device)



epochs=3
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
                for inputs, labels in testloader:
                    inputs, labels=inputs.to(device), labels.to(device)
                    logps =model.forward(inputs)
                    batch_loss=criterion(logps, labels)
                    test_loss+=batch_loss.item()
                    ps=torch.exp(logps)
                    top_p, top_class=ps.topk(1, dim=1)
                    equals = top_class==labels.view(*top_class.shape)
                    accuracy+= torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.."
                          f"Train loss: {running_loss/print_every:.3f}.."
                          f"Test loss: {test_loss/len(testloader):.3f}.."
                          f"Test accuracy: {accuracy/len(testloader):.3f}")  
        
        running_loss=0
        model.train()
 

correct = 0
total = 0
model.to('cuda')
model.eval()
with torch.no_grad():
    for data in dataloaders[2]: 
        image, label = data
        image, label = image.to('cuda'), label.to('cuda')
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

print('Test Accuracy: %d %%' % (100 * correct / total))





model.class_to_idx = image_datasets[0].class_to_idx

checkpoint = {'input_size' : 1024,
 
    'output_size' : 102,
 

    'optimizer': optimizer,
 
    'arch': "densenet121",
 
    'state_dict': model.state_dict(),
 
    'optimizer_state': optimizer.state_dict(),
 
    'class_to_idx': model.class_to_idx,
    
    

    'classifier': classifier.state_dict()}

              

torch.save(checkpoint, 'model_checkpoint.pth')



def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model=models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1024,500),
                           nn.ReLU(),
                           nn.Linear(500, 102),
                           nn.LogSoftmax(dim=1))
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint ['class_to_idx']
    
    return model
model = load_checkpoint('model_checkpoint.pth')





def process_image(ima):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
   
    
    ima = Image.open(ima)
    ima = ima.resize((256, 256))
    ima = ima.crop((0, 0, 224, 224))
    np_image = np.array(ima) / 255
    mean = np.array([0.485, 0.456, 0.406])
    stdv = np.array([0.229, 0.224, 0.225])
    ima = (np_image - mean) / stdv
    f_image = ima.transpose((2, 1, 0))

    return torch.from_numpy(f_image)




def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    x = np.asarray(image)

    image = x.transpose((1, 2, 0))
    
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (std * image) + mean
    
    
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

img = random.choice(os.listdir('./flowers/test/7/'))
img_path = './flowers/test/7/' + img





model.class_to_idx = image_datasets[0].class_to_idx


def predict(image_path, model, topk=5):
 
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
 
    '''
    
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




def sanity_checking(image_path, model):
    plt.figure(figsize = (7,11))
    ax = plt.subplot(2,1,1)
    name = image_path.split('/')[-2]
    title = cat_to_name[str(name)]
    prob, flower = predict(image_path, model, 5)
    labels = []
    for i in flower:
        labels.append(cat_to_name[i])
    img = process_image(image_path)
    imshow(img, ax, title = title)
    plt.subplot(2,1,2)
    data = pd.DataFrame({'Probabilities': prob, 'Flower Names': labels})
    sb.barplot(x="Probabilities", y="Flower Names", palette=sb.color_palette("tab10"), data=data)
    plt.show()


sanity_checking(img_path, model)
