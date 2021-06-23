
import torch
import torchvision
import torchvision.transforms as trans
from torch.utils.data.dataset import Subset
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------------------

class fashion_mnist_numpy_dataset :

     #----------------------------------------------------------------------------------

     def __init__(self,ds_root_path):
     
       if not os.path.exists(ds_root_path):
              os.makedirs(ds_root_path)
            
       torch_dataset = torchvision.datasets.FashionMNIST(ds_root_path,
                       train=False,download=True,transform=trans.Compose([trans.ToTensor()]))      
       self.dataset = torch_dataset.data.numpy().astype(int)
       self.targets = torch_dataset.targets.numpy().astype(int)
       
       self.class_str = ('target-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle boot')
       self.max_gray_val = 256 # Range is 0-256 (type: uint8 , unsigned char)
     
       # print ("dataset.shape = %s\n" % str(self.dataset.shape)) 
       # print ("targets.shape = %s\n" % str(self.targets.shape)) 

     #----------------------------------------------------------------------------------
 
     def clean_img_csv_folder(self) :
          if os.path.exists('img_csv_dumps'):
              for file in os.scandir('img_csv_dumps'):
                os.remove(file.path)            
            
     #----------------------------------------------------------------------------------
 
     def export_img_csv(self,img_idx) :
          if not os.path.exists('img_csv_dumps'):
              os.makedirs('img_csv_dumps')
              
          np.savetxt(("img_csv_dumps/img_%d.csv" % img_idx) ,self.dataset[img_idx].astype(int), fmt='%4d' , delimiter=",")
          target_file = open(("img_csv_dumps/target_%d.csv"%img_idx),"w") ;
          target_file.write("%d\n" % self.targets[img_idx].astype(int)) 
          target_file.close()
          
     #----------------------------------------------------------------------------------
    
     def ascii_art_img_28x28(self,img_idx) :
     
       print ("\nClass ID : %d : %s" % (self.targets[img_idx],self.class_str[self.targets[img_idx]]))
       gray_scale_vec =  " .:-=+*#%@"
       for i in range(28) :
        for j in range (28) :
           gray_val = self.dataset[(img_idx,i,j)]
           gray_idx = (gray_val * len(gray_scale_vec)) // self.max_gray_val ;
           c = gray_scale_vec[gray_idx]
           sys.stdout.write("%c " % c)
        sys.stdout.write("\n")
     
     #----------------------------------------------------------------------------------
  
     def matplot_img_28x28(self,img_idx, pause_seconds) :
        plt.title("Image %d : %s" % (img_idx,self.class_str[self.targets[img_idx]]))
        fig_data = self.dataset[img_idx]
        plt.imshow(fig_data,cmap='gray',vmin=0,vmax=255)   
        plt.draw()
        plt.pause(pause_seconds)
        plt.clf()
        
#----------------------------------------------------------------------------------

if __name__ == '__main__':

     ds = fashion_mnist_numpy_dataset('../inference/data/FashionMNIST')
     ds.clean_img_csv_folder()     
     plt.rcParams["figure.figsize"] = (3,3)
     plt.ion()
     
     for i in range(10) :
       img_idx = np.random.randint(0,ds.targets.shape[0])
       print("Randomly selected Image %d" % img_idx)
       ds.ascii_art_img_28x28(img_idx)
       ds.matplot_img_28x28(img_idx,1)
       ds.export_img_csv(img_idx)



#----------------------------------------------------------------------------------
     
     
