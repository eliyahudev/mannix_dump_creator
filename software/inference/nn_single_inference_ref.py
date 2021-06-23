
import torch
import torchvision
import torchvision.transforms as trans
from torch.utils.data.dataset import Subset
import numpy as np
import NN_model
import dataset_access as dsa
import sys
import os

#--------------------------------------------------------------------------------

def dump_np_to_csv_multi2D(np_arr,path_prefix,format) :
  #Recursive functions to dump a numpy array with dimensions higher than 2 down to multiple 2D arrays
  
  if (len(np_arr.shape)<=2) :
    np.savetxt(path_prefix+'.csv',np_arr, fmt=format , delimiter=",")
  else : 
    for i in range(np_arr.shape[0]) :
        dump_np_to_csv_multi2D(np_arr[i],"%s_%d"%(path_prefix,i),format)
  

#---------------------------------------------------------------------------------

@torch.no_grad()

#--------------------------------------------------------------------------------

def run_main() : 

    device = torch.device("cpu")
 
    model = NN_model.My_NN(hidden_dim_1=120, hidden_dim_2=64, output_dim=10)
    model.load_state_dict(torch.load('best_model_state_dict.pt'))

    model.set_gen_np_ref = True
    
    ds = dsa.fashion_mnist_numpy_dataset('./data/FashionMNIST')

    # img_idx = np.random.randint(0,ds.targets.shape[0])
    img_idx = 9558
    
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
      
    if not os.path.exists('csv_single_ref_dumps/orig'):
        os.makedirs('csv_single_ref_dumps/orig')

    dump_np_to_csv_multi2D(( model.conv1_out_np        ), "csv_single_ref_dumps/orig/conv1_out",        '%4.2f')    
    dump_np_to_csv_multi2D(( model.conv1_relu_out_np   ), "csv_single_ref_dumps/orig/conv1_relu_out",   '%4.2f')
    dump_np_to_csv_multi2D(( model.conv1_pool2d_out_np ), "csv_single_ref_dumps/orig/conv1_pool2d_out", '%4.2f')
    dump_np_to_csv_multi2D(( model.conv2_out_np        ), "csv_single_ref_dumps/orig/conv2_out",        '%4.2f')      
    dump_np_to_csv_multi2D(( model.conv2_relu_out_np   ), "csv_single_ref_dumps/orig/conv2_relu_out",   '%4.2f')
    dump_np_to_csv_multi2D(( model.conv2_pool2d_out_np ), "csv_single_ref_dumps/orig/conv2_pool2d_out", '%4.2f')
    dump_np_to_csv_multi2D(( model.fc1_relu_out_np     ), "csv_single_ref_dumps/orig/fc1_relu_out",     '%4.2f')
    dump_np_to_csv_multi2D(( model.fc2_out_np          ), "csv_single_ref_dumps/orig/fc2_out",          '%4.2f')
    dump_np_to_csv_multi2D(( model.fc2_out_relu_np     ), "csv_single_ref_dumps/orig/fc2_out_relu",     '%4.2f')
    dump_np_to_csv_multi2D(( model.fc3_out_np          ), "csv_single_ref_dumps/orig/fc3_out",          '%4.2f')
    
      
#--------------------------------------------------------------------------------------


if __name__ == '__main__':
   run_main()   
