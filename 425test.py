# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 22:24:40 2019

@author: nana
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 21:43:28 2019

@author: nana
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 13:56:25 2018

@author: nana
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 17:15:00 2018

@author: nana
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:09:43 2018

@author: nana
"""
#做完agc的数据

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
import scipy.io as sio
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=10, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-2, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--mode", type=str, default="S", help='with known noise level (S) or blind training (B)')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
opt = parser.parse_args()

def main():
    ## Load dataset
    print('Loading dataset ...\n')
    dataset_train = sio.loadmat(r'E:/data/class2/traindata_wan.mat')
    #dataset_train = sio.loadmat(r'E:/DnCNN-PyTorch-master/200data.mat')
    train3dimen = dataset_train['B']
    train3dimen = train3dimen.astype(np.float32)
    #traindata1 = train3dimen.reshape(20480, 100)
    #trainsize = train3dimen.shape[1]*train3dimen.shape[2]
    #for i in range(0,100):
    traindata = np.expand_dims(train3dimen[:,:,:].copy(), 1)
    #traindata = traindata.reshape(1, 20480, 100)
    dataset_noise = sio.loadmat(r'E:/data/class2/trainnoise_wan.mat')
    #dataset_noise = sio.loadmat(r'E:/DnCNN-PyTorch-master/200noise.mat')
    noise3dimen = dataset_noise['Bnoise']
    noise3dimen = noise3dimen.astype(np.float32)
    #noisedata1 = noise3dimen.reshape(20480, 100)
    noisedata = np.expand_dims(noise3dimen[:,:,:].copy(), 1)
    #for j in range(0, 100): 
        #train2dimen = train3dimen[i]
    #loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    #print("# of training samples: %d\n" % int(len(dataset_train)))
    ## Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)#返回标量，loss.sum()
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    #noiseL_B=[0,55] # ingnored when opt.mode=='S'
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
           current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        #print('learning rate %f' % current_lr)
        # train
        #for j in range(0, 100):
            # training step
        model.train()
        model.zero_grad()
        optimizer.zero_grad()
        img_train = traindata[:,:,:,:]
        noise = noisedata[:,:,:,:]
        imgn_train = img_train + noise
        '''
        np.save("imgn_train.npy",imgn_train)
        mat = np.load("imgn_train.npy")
        sio.savemat('imgn_train.mat', {'imgn_train': mat})
        '''
        img_train = torch.Tensor(img_train)
        img_train = Variable(img_train) 
        imgn_train = torch.Tensor(imgn_train)
        imgn_train = Variable(imgn_train)
        img_train = img_train.cuda()
        imgn_train = imgn_train.cuda()
        '''
        imgn_train1 = imgn_train1.cpu()
        imgn_train1 = imgn_train1.data.numpy()
        np.save("imgn_train1.npy",imgn_train1)
        mat1 = np.load("imgn_train1.npy")
        sio.savemat('imgn_train1.mat', {'imgn_train1': mat1})
        '''
        #print (imgn_train.shape)
        noise = torch.Tensor(noise)
        noise = Variable(noise)
        noise = noise.cuda()
        '''
        noise = noise.cpu()
        noise = noise.data.numpy()
        np.save("out_noise.npy",noise)
        mat = np.load("out_noise.npy")
        sio.savemat('outnoise.mat', {'outnoise': mat})
        '''
        out_train = model(imgn_train)
        #print (out_train)
        loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
        print (loss)
        loss.backward()
        optimizer.step()
        # results
        model.eval()
        imgn_train = imgn_train.cpu()
        out_train = out_train.cpu()
        out_effective = torch.clamp(imgn_train-out_train, 0., 1.)
        out_effective = out_effective.cpu()
        out_effective = out_effective.data.numpy()
        np.save("gra_effective.npy",out_effective)
        mat1 = np.load("gra_effective.npy")
        sio.savemat(r'E:/data/class2/xs_gra_effective_518.mat', {'gra_effective': mat1})
        #out_train = out_train.cpu()
        out_train = out_train.data.numpy()
        np.save("gra_train.npy",out_train)
        mat = np.load("gra_train.npy")
        sio.savemat(r'E:/data/class2/xs_gra_train_518.mat', {'gra_train': mat}) 
    '''
    noise = noise.cpu()
    noise = noise.data.numpy()
    np.save("out_noise.npy",noise)
    mat = np.load("out_noise.npy")
    sio.savemat('outnoise.mat', {'outnoise': mat})
    '''
        #out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        #psnr_train = batch_PSNR(out_train, img_train, 5)
        #print("[epoch %d][%d/100] loss: %.4f PSNR_train: %.4f" %
             #(epoch+1, i+1, loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
        #if step % 10 == 0:
            # Log the scalar values
            #writer.add_scalar('loss', loss.item(), step)
            #writer.add_scalar('PSNR on training data', psnr_train, step)
        #step += 1
        ## the end of each epoch
    #model.eval()
    #val
    '''
    dataset_val = sio.loadmat(r'E:/data/matlab/val_datawithnoise.mat')
    imgn_val = dataset_val['DX']
    imgn_val = np.expand_dims(imgn_val[:,:,:].copy(), 1)
    imgn_val = torch.Tensor(imgn_val)
    imgn_val = Variable(imgn_val)
    imgn_val = imgn_val.cuda()
    out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
    #print (out_val)
    out_val = out_val.cpu()
    out_val = out_val.data.numpy()
    np.save("out_val.npy",out_val)
    mat3 = np.load("out_val.npy")
    sio.savemat('out_val.mat', {'out_val': mat3})
    '''
    '''
        # validate
        psnr_val = 0
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
            imgn_val = img_val + noise
            img_val, imgn_val = Variable(img_val.cuda()), Variable(imgn_val.cuda())
            out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
            psnr_val += batch_PSNR(out_val, img_val, 1.)
        psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
          # log the images
        out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))
if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
'''
main()  