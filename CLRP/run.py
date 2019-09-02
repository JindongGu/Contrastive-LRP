'''
This is implementation of Contrastive Layerwise Relevance Propagation (CLRP) in the paper:

Gu, Jindong, Yinchong Yang, and Volker Tresp."Understanding Individual Decisions of CNNs via Contrastive Backpropagation."
Asian Conference on Computer Vision. Springer, Cham, 2018.

'''

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import torchvision.transforms as transforms

import os
import sys
from PIL import Image
from clrp_lib import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# register forward and backward hook functions in Conv, MaxPooling, Linear layers of the loaded model
def register_hooks(model=None):
    model.features[0].register_forward_hook(conv_forward_hook)
    model.features[0].register_backward_hook(FirstConv_backward_hook)

    for i in [2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]:
        model.features[i].register_forward_hook(conv_forward_hook)
        model.features[i].register_backward_hook(conv_backward_hook)

    for i in [4, 9, 16, 23, 30]:
        model.features[i].register_forward_hook(maxpool_forward_hook)
        model.features[i].register_backward_hook(maxpool_backward_hook)

    for i in [0, 3, 6]:
        model.classifier[i].register_forward_hook(linear_forward_hook)
        model.classifier[i].register_backward_hook(linear_backward_hook)
        
    return model

# to create a saliency map of the target signal or the contrastive signal
def clrp(out, target_class, contrastive_signal = False, CLRP = 'type1'):
    
    # initialize the relevance, i.e., the logit score
    Nega_flag_ = False
    if contrastive_signal == False:
        R = torch.zeros(out.shape).to(device)
        R[0, target_class] = out[0, target_class].detach()
    else:
        if CLRP == 'type1': R = out.detach(); R[0, target_class] = 0
        elif CLRP == 'type2':
            R = torch.zeros(out.shape).to(device); R[0, target_class] = out[0, target_class].detach(); Nega_flag_ = True

    # (Nega_flag_= False) for CLRP1 and (Nega_flag_= True) for CLRP2
    initialise_rel(torch.abs(R), Nega_flag_ = Nega_flag_)

    # propagate the relevance score back, which is computed in registered hook functions
    out.backward(out, retain_graph=True)
    return return_rel().cpu().data.numpy()


# visualize the saliency maps (LRP and CLRP)
def visualize(sm_lrp, sm_contrast, name):
    print(sm_lrp.shape, sm_contrast.shape)
    sm_lrp_ = (sm_lrp[0]/sm_lrp.max()).transpose((1, 2, 0))
    sm_lrp_ = np.maximum(0, sm_lrp_)
    Image.fromarray((sm_lrp_*255).astype(np.uint8), mode='RGB').save('SMs/'+ name + '_LRP.png')

    sm_lrp_ = sm_lrp/sm_lrp.sum()
    sm_contrast = sm_contrast/sm_contrast.sum()
    sm_clrp = sm_lrp_ - sm_contrast

    sm_clrp_ = (sm_clrp[0]/sm_clrp.max()).transpose((1, 2, 0))
    sm_clrp_ = np.maximum(0, sm_clrp_)
    Image.fromarray((sm_clrp_*255).astype(np.uint8), mode='RGB').save('SMs/'+ name + '_CLRP.png')


## load pre-trained VGG16 model and register hook functions in the loaded model
model = models.vgg16(pretrained=True)
model = register_hooks(model=model)
model = model.eval()
model = model.to(device)

## create a preprocessor
preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

## model prediction and Gradient backpropagation
images = ['ze1', 'ze2', 'ze3', 'ze4']


for i in range(len(images)):
    
    # laod and preprocess the image
    img = Image.open('imgs/'+images[i] + '.jpg')
    img_tensor = preprocess(img).to(device)
    img_variable = Variable(img_tensor.unsqueeze_(0), requires_grad=True)

    # classify the loaded image
    out = model(img_variable)

    # specify a target class to create saliency maps
    # 340, 386 correspond to Zebra and Africa_elephant in ImageNet dataset, respectively.
    target_class = 386

    sm_lrp = clrp(out, target_class, contrastive_signal = False)
    sm_contrast = clrp(out, target_class, contrastive_signal = True, CLRP = 'type1')
     
    #visualize the relevance score
    visualize(sm_lrp, sm_contrast, images[i] + '_' + str(target_class))


    






