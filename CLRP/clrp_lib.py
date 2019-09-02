
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import copy
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialise and return the relevance value
R = None
Nega_flag = None

def initialise_rel(R_, Nega_flag_):
    global R, Nega_flag
    Nega_flag = Nega_flag_
    R = R_
    
def return_rel():
    global R
    return R


# hook function for Linear module
def linear_forward_hook(self, input, output):    
    self.input = input[0]
    
def linear_backward_hook(self, grad_input, grad_output):
    global R, Nega_flag
    
    W = self.parameters().__next__().clone()
    if Nega_flag == True:
        W *= torch.Tensor([-1]).to(device);
        Nega_flag = False
    
    V = torch.max(torch.zeros(W.t().shape).to(device), W.t())
    Z = torch.mm(self.input,V) + torch.Tensor([1e-9]).to(device)
    S = R/Z
    C = torch.mm(S,V.t())
    R = self.input*C


    
# hook function for Conv module and Alpha_Beta rules for first Convolutional layer
def conv_forward_hook(self, input, output):    
    self.input = input[0]

def Conv_single(input_, self_, R_in):    
    with torch.enable_grad():
        
        input = Variable(input_, requires_grad=True)

        V = torch.max(torch.zeros(self_.weight.shape).to(device), self_.weight)

        Z = F.conv2d(input, V, None, self_.stride,
                            self_.padding, self_.dilation, self_.groups) + torch.Tensor([1e-9]).to(device)
        S = R_in/Z

        Z.backward(S)
        return input * input.grad
    
    
def conv_backward_hook(self, grad_input, grad_output):
    global R
    R = Conv_single(self.input, self, R)


def FirstConv_backward_hook(self, grad_input, grad_output):
    with torch.enable_grad():
        global R

        VL = torch.max(torch.zeros(self.weight.shape).to(device), self.weight)

        VH = torch.min(torch.zeros(self.weight.shape).to(device), self.weight)

        Input = Variable(self.input.data, requires_grad=True).to(device)

        Low = Variable((torch.ones(self.input.shape) * torch.Tensor([-2.1179])).to(device), requires_grad=True)

        High = Variable((torch.ones(self.input.shape) * torch.Tensor([2.64])).to(device), requires_grad=True)

        Z1 = F.conv2d(Input, self.weight, None, self.stride,
                                self.padding, self.dilation, self.groups)

        Z2 = F.conv2d(Low, VL, None, self.stride,
                               self.padding, self.dilation, self.groups)

        Z3 = F.conv2d(High, VH, None, self.stride,
                                self.padding, self.dilation, self.groups)

        Z = Z1 - Z2 - Z3 + torch.Tensor([1e-9]).to(device); S = R/Z

        Z3.backward(S); Z1.backward(S); Z2.backward(S); 

        R = Input*Input.grad - Low*Low.grad - High*High.grad



# hook function for Max Pooling module
def maxpool_forward_hook(self, input, output):
    self.input = input[0]
    self.output_shape = output.shape
    

def maxPool_relprop(input, size=3, stride=1, padding=0, R_temp = None):
    out = F.max_pool2d(input, kernel_size=size, stride=1, padding=padding)
    mb,nx,wy,hy = out.shape
    
    R_ = torch.zeros(out.shape).to(device)
    R_[:,:,::stride,::stride] = R_temp
    
    mb,nx,wx,hx = input.shape
    Re = torch.zeros((mb,nx,wx+2*padding,hx+2*padding)).to(device)

    index = torch.zeros(mb, size*size, nx, wy, hy).to(device)

    for i in range(size):
            for j in range(size):
                        index[:,i*size+j] = torch.eq(input[:,:,i:i+wy,j:j+hy], out).to(device, dtype=torch.float32)
                        
    redisR = (1.0/index.sum(dim=1)*R_).to(device)
    
    for i in range(size):
            for j in range(size):
                        Re[:,:,i:i+wy,j:j+hy] += index[:, i*size+j] * redisR           
    return Re[:, :, padding : wx+padding, padding : hx+padding]



def maxpool_backward_hook(self, grad_input, grad_output):
    global R
    
    if len(self.output_shape) == 4 and R.dim() == 2: R = R.reshape(self.output_shape)
    
    R = maxPool_relprop(self.input, size=self.kernel_size, stride=self.stride, padding=self.padding, R_temp = R)
             
 


