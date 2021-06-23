import torch.nn as nn
import torch.nn.functional as F

class My_NN(nn.Module):
    def __init__(self, hidden_dim_1, hidden_dim_2, output_dim):

        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)

        FC_input_dim = 12 * 4 * 4  # this is the size of the image after pass through the convolution layer
       
        self.fc1 = nn.Linear(in_features=FC_input_dim, out_features=hidden_dim_1)
        self.fc2 = nn.Linear(in_features=hidden_dim_1, out_features=hidden_dim_2)
        self.out = nn.Linear(in_features=hidden_dim_2, out_features=output_dim)

        # Reference intermediate tensors      
        self.set_gen_np_ref       = False
        self.conv1_out_np         = None        
        self.conv1_relu_out_np    = None
        self.conv1_pool2d_out_np  = None
        self.conv2_out_np         = None        
        self.conv2_relu_out_np    = None
        self.conv2_pool2d_out_np  = None
        self.fc1_out_np           = None
        self.fc1_relu_out_np      = None
        self.fc2_out_np           = None
        self.fc2_out_relu_np      = None
        self.fc3_out_np           = None
        
    def forward(self, input_vector):
        # convolution phase
        conv1_out = self.conv1(input_vector)
        conv1_relu_out = F.relu(conv1_out)
        conv1_pool2d_out = F.max_pool2d(conv1_relu_out, kernel_size=2, stride=2)

        conv2_out = self.conv2(conv1_pool2d_out)
        conv2_relu_out = F.relu(conv2_out)
        conv2_pool2d_out = F.max_pool2d(conv2_relu_out, kernel_size=2, stride=2)

        # here we reshape the data matrix [batch, channel, height, width] to one long vector  to be suitable for the FC phase        
        fc1_in = conv2_pool2d_out.reshape(-1, 12 * 4 * 4)  # 12 - num of channel, 4*4 - the matrix after conv phase
        
        # fc1
        #fc1_in = conv_out.reshape(-1, 12*4*4)
        fc1_out = self.fc1(fc1_in)
        fc1_relu_out = F.relu(fc1_out)

        # fc2
        fc2_out = self.fc2(fc1_relu_out)
        fc2_out_relu = F.relu(fc2_out)
        model_out = self.out(fc2_out_relu)
        
        if self.set_gen_np_ref :

            self.conv1_out_np         =  conv1_out.numpy()                
            self.conv1_relu_out_np    =  conv1_relu_out.numpy()           
            self.conv1_pool2d_out_np  =  conv1_pool2d_out.numpy()
            self.conv2_out_np         =  conv2_out.numpy()             
            self.conv2_relu_out_np    =  conv2_relu_out.numpy()      
            self.conv2_pool2d_out_np  =  conv2_pool2d_out.numpy()
            self.fc1_out_np           =  fc1_out.numpy()       
            self.fc1_relu_out_np      =  fc1_relu_out.numpy()    
            self.fc2_out_np           =  fc2_out.numpy()      
            self.fc2_out_relu_np      =  fc2_out_relu.numpy()
            self.fc3_out_np           =  model_out.numpy()        
                    
        return model_out
        
   