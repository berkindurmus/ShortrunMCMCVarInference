# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 11:55:53 2020

@author: Berkin


"""
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.base import BaseEstimator, ClassifierMixin
import torch as t,torch.nn as nn
import torchvision as tv , torchvision.transforms as tr
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch.nn.functional as H
from tqdm import tqdm
import pickle


sgma=0.3
seed = 1
im_sz = 32  
n_ch = 1 # Input channel size 
m=64# Sample Size
n_f = 64 
K=25
n_s=300000 # number of steps
ngf = 1
nz=100# Number of latent variables

s_langevin=0.0001 # s
#t.manual_seed(seed)
#if t.cuda.is_available() :
 # t.cuda.manual_seed_all(seed)
  
device = t.device('cuda' if t.cuda.is_available() else 'cpu' )
class est(BaseEstimator, ClassifierMixin):  
    
    def __init__(self,a,b):
        self.a = a
        self.b=b
        self.scr=0

    def fit(self, z_smp,p_dat):
        
        t_z_smp=t.from_numpy(z_smp).to(device)
        t_p_dat=t.from_numpy(p_dat).to(device)
        z_k_m,z_k,_,_=langevin_dyn(t_z_smp,t_p_dat,self.a,self.b,25,s_noise(t_z_smp))
        self.scr=t.pow(z_k-z_k_m, 2).sum()
        
        return (self.scr.cpu().detach().cpu().numpy())

    def score(self,z_smp,p_dat):
        
        t_z_smp=t.from_numpy(z_smp).to(device)
        t_p_dat=t.from_numpy(p_dat).to(device)
        z_k_m,z_k,_,_=langevin_dyn(t_z_smp,t_p_dat,self.a,self.b,25,s_noise(t_z_smp))
        res=t.pow(z_k-z_k_m, 2).sum()
        
        return(-res.detach().cpu().numpy()) 
    
def s_noise(x):
  return t.normal(mean=0, std=1,size=(1, nz,1,1)).to(device)

def langevin_dyn(z_1,p_dat,a,b,K,noise_s):

    dlog=[]
    loss4=t.nn.MSELoss()
    z_1_aut=t.autograd.Variable(z_1, requires_grad =True )
    z0_initial=z_1_aut.clone()
    for i in range(K):
    
        pred=f(z_1_aut)
        loss = 1/(2*sgma**2) * t.pow(p_dat- pred, 2).sum() 
        loss /= (p_dat.size(0))
        f_prime=t.autograd.grad(loss,[z_1_aut],retain_graph=True,create_graph=True )
        dlog.append(f_prime[0].clone().detach().cpu().numpy())
        z_1_aut.data=z_1_aut.data-a*(z_1_aut.data+f_prime[0])+b *s_noise(z_1_aut)
        if(i==K-5):
            y=z_1_aut.clone()
            a=0.01
            b=0.01
    dynamics_dis=loss4(z0_initial,z_1_aut)
    return y.detach(),z_1_aut.detach(),dynamics_dis.detach().cpu().numpy(),dlog

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()

class G (nn.Module) :
    def __init__(self):
        super(G, self).__init__()
        self.convt1 = nn.ConvTranspose2d(nz, 256, kernel_size=4, stride=1, padding=0)
        self.convt2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.convt4 = nn.ConvTranspose2d(64, 3, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.leakyrelu = nn.LeakyReLU()
        self.tanh = nn.Tanh()

    def forward(self, z):
        out = self.convt1(z)
        out = self.leakyrelu(out)
        out = self.convt2(out)
        out = self.leakyrelu(out)
        out = self.convt3(out)
        out = self.leakyrelu(out)
        out = self.convt4(out)
        out = self.tanh(out)
        return out


rss=[]
######################################################
#transform =tr.Compose( [ tr.Resize (im_sz) , tr.ToTensor() , tr.Normalize (( .5  ) , (.5 )) ] )
#p_d = t.stack ( [ x [0] for x in tv.datasets.CelebA(root='./data',download=True,transform= transform ) ] ).to(device )
#p_d=p_d[:1000,:,:]
#p_d_labels=((t.nn.functional.one_hot(tv.datasets.MNIST(root='./data',download=True,transform= transform ).train_labels).reshape(60000,10,1,1)).float()).to(device)
path = 'data/celeba/celeba_40000_32.pickle'

with open(path, mode='rb') as f:
    ds = pickle.load(f)
    
#normalize = lambda x: ((x / 255.) * 2.) - 1.
#for i in range(len(ds)):
    #ds[i]=normalize(ds[i])
#normalize = lambda x: ((x / 255.) * 2.) - 1.

p_d = t.stack([x for x in ds]).to(device)

#temp_z_ = t.randn(m, 100)
#fixed_z_ = temp_z_
#fixed_z_ = (fixed_z_.view(-1, 100, 1, 1)).to(device)

def sample_p_d ( ) :
  p_d_i =t.LongTensor(m).random_( 0 ,p_d.shape[0])
  return (p_d [p_d_i ] ).detach()

z_0 = lambda : t.normal(mean=0, std=1,size=(m, nz,1,1)).to(device)
f =G().to(device)

optimizerG = t.optim.Adam(f.parameters(), lr=1e-4, betas=[.9, .999])


cur_a=0.1
cur_b=0.1
list_a=[]
list_b=[]
dynamic_dis_l=[]
dlog_l=[]

#############################################################
for i in tqdm(range (n_s)):
    
    d_sample=sample_p_d()
    z_sample=z_0() 
    z_sample,_,dynamics_dis,dlog=langevin_dyn(z_sample,d_sample,cur_a,cur_b,25,s_noise(z_sample))
    dynamic_dis_l.append(dynamics_dis)
    dlog_l.append(dlog)
    pred= f(z_sample.data)
    loss = 1/(2*sgma**2) * t.pow(d_sample - pred, 2).sum() 
    loss /= (d_sample.size(0))
    optimizerG.zero_grad()
    loss.backward() 
    
    #t.nn.utils.clip_grad_norm_(f.parameters(), 1)
    
    optimizerG.step()
    
        
    #Update s step size
    if(i%1000==0 ):
        a =np.linspace(0.05,0.1,10)
        b= np.linspace(0.05,0.1,10)
        
        my_param_grid=dict(a=a, b=b)
        cv = [(slice(None), slice(None))]
        est_mdl=est(cur_a,cur_b)
        gs = GridSearchCV(estimator=est_mdl, param_grid=my_param_grid,verbose=100,cv=cv,n_jobs=6)
        grid_results = gs.fit(z_sample.cpu().numpy(),d_sample.cpu().numpy())
        print("Value of a and b changed to" +str(gs.best_params_))
        cur_a=gs.best_params_['a']
        cur_b=gs.best_params_['b']
    
    
    # Plot current Estimations
    if(i%1000==0):
         # Show Progress
         list_a.append(cur_a)
         list_b.append(cur_b)
         print(str((100*i)/n_s)+' % done')
         
         #Estimate the image
         a=(t.clamp(((f(z_sample[0].reshape(1,nz,1,1)).reshape(3,im_sz,im_sz).cpu().detach()).permute(1,2,0)),-1.,1.)+1)/2.
        
        # Plot estimation and ground truth
         fig, (ax1, ax2,ax3,ax4) = plt.subplots(1, 4)
         ax1.imshow(a)
         a2=(t.clamp(((d_sample[0].cpu().detach().reshape(3,im_sz,im_sz)).permute(1,2,0)),-1.,1.)+1)/2.
         ax2.imshow(a2)
         a3=(t.clamp(((f(z_sample[1].reshape(1,nz,1,1)).reshape(3,im_sz,im_sz).cpu().detach()).permute(1,2,0)),-1.,1.)+1)/2.
         ax3.imshow(a3)
         a4=(t.clamp(((d_sample[1].cpu().detach().reshape(3,im_sz,im_sz)).permute(1,2,0)),-1.,1.)+1)/2.
         ax4.imshow(a4)
         plt.show()
        
         print('Calculated loss = '+str(loss.data.cpu().numpy()))
         loss=0
    
         
    
