import matplotlib.pyplot as plt
import os
import torch
import torchvision.transforms as transforms

os.environ['KMP_DUPLICATE_LIB_OK']='True'
mean=torch.Tensor([[[0.485, 0.456, 0.406]]]).view(-1,1,1)
std=torch.Tensor([[[0.229, 0.224, 0.225]]]).view(-1,1,1)
t=transforms.ToPILImage(mode=None)
def show(n_way,n_shot,n_query,x,x_i):
    if n_way>10:
        n_way=10
    plt.figure()
    for k in range(n_way):
        for i in range(n_shot):
            plt.subplot(n_way,n_shot+n_query,k*(n_shot+n_query)+i+1)
            pic = x[k*n_shot+i].mul_(std).add_(mean)
            plt.imshow(t(pic))
            plt.axis('off')
        for j in range(n_query):
            plt.subplot(n_way, n_shot + n_query, k*(n_shot+n_query)+n_shot+j+1)
            pic = x_i[k*n_query+j].mul_(std).add_(mean)
            plt.imshow(t(pic))
            plt.axis('off')
    plt.show()

def showsq(n_way,n_shot,n_query,x):
    if n_way>10:
        n_way=10
    plt.figure()
    for k in range(n_way):
        for i in range(n_shot+n_query):
            plt.subplot(n_way,n_shot+n_query,k*(n_shot+n_query)+i+1)
            pic = x[k*(n_shot+n_query)+i].mul_(std).add_(mean)
            plt.imshow(t(pic))
            plt.axis('off')
    plt.show()

def showbatch(bs,x1,x2):
    if bs>10:
        bs=10
    plt.figure()
    for k in range(bs):
        plt.subplot(bs,2,k*2+1)
        pic = x1[k].mul_(std).add_(mean)
        plt.imshow(t(pic))
        plt.subplot(bs,2,k*2+2)
        pic = x2[k].mul_(std).add_(mean)
        plt.imshow(t(pic))
        plt.axis('off')
    plt.show()









