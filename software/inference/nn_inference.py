
import torch
import torchvision
import torchvision.transforms as trans
from torch.utils.data.dataset import Subset
import numpy as np
import NN_model
import dataset_access as dsa
import sys
import os

#------------------------------------------------------------------------------------------------------------

@torch.no_grad()

#--------------------------------------------------------------------------------

def check_inference(num_images) : 

    device = torch.device("cpu")
 
    model = NN_model.My_NN(hidden_dim_1=120, hidden_dim_2=64, output_dim=10)
    model.load_state_dict(torch.load('best_model_state_dict.pt'))

    ds = dsa.fashion_mnist_numpy_dataset('./data/FashionMNIST')

    for i in range(num_images) :
    
         img_idx = np.random.randint(0,ds.targets.shape[0])
         print("Randomly selected Image %d" % img_idx)
         
         img_np = ds.dataset[img_idx] # numpy 2D array
         target = ds.targets[img_idx]
         
         img_tensor_np = np.empty((1,1,28,28), dtype=int)
         img_tensor_np[0,0,:,:] = img_np
         
         img_tensor_torch = torch.from_numpy(img_tensor_np).int()
         img_tensor_torch = img_tensor_torch.to(device)
         
         pred = model(img_tensor_torch.float())
         pred_np = pred.numpy()
         
         max_idx = np.argmax(pred_np)
         
         ds.ascii_art_img_28x28(img_idx)
         
         chk_str = 'CORRECT' if  (max_idx==target) else 'ERROR, should be %s' % ds.class_str[ds.targets[target]]      
         print ("Predicted max_idx=%d : %s %s" % (max_idx,ds.class_str[ds.targets[img_idx]], chk_str)) ;
                  
         ds.matplot_img_28x28(img_idx,2)
         
         
#--------------------------------------------------------------------------------------


if __name__ == '__main__':
   check_inference(20)  
