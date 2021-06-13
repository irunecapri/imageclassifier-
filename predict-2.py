import torch
from torchvision import transforms, models
import argparse
import json
from utility import process_image
import argparse
from network import load_checkpoint
from network import predict
from utility import process_image

def get_input_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', action='store', help='path to image to be classified')
    parser.add_argument('checkpoint', action='store', help='path to stored model')
    parser.add_argument('--top_k', action='store', type=int, default=1, help='how many most probable classes to print out')
    parser.add_argument('--category_names', action='store', help='file which maps classes to names')
    parser.add_argument('--gpu', action='store_true', help='use gpu to infer classes')
    args=parser.parse_args()
    
    return parser.parse_args()

if torch.cuda.is_available() and use_gpu:
 
       image_tensor = image_tensor.cuda()
 
       model = model.cuda()    

    
def main():
    in_arg = get_input_args()    
    
    model = load_checkpoint(in_arg.saved_checkpoint, in_arg.gpu)
    
    processed_image = process_image(in_arg.image_path) 
    
    top_probs, top_labels = predict(in_arg.image_path, model, in_arg.top_k, in_arg.gpu)
    
 
if __name__ == "__main__":
    main()
            
    
    
    
    
    
    
    
    
   