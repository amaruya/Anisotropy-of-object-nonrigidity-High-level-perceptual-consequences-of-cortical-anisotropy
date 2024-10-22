#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot tuning curves for ME models in Figure 5A and B


"""


import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
global max_abs
import torch
import os 
current_folder = os.path.dirname(os.path.abspath(__file__))
path=current_folder+'/Toolbox/Data/'
toolbox_path=current_folder+'/Toolbox/'
import sys
sys.path.append(toolbox_path)
from pyr3D_Gabor import pyr3D_Gabor


def Make_sine_wave(Dir=45,Sz=64,T_sz=32,sp_frq=4,tmp_frq=5,Vis=1):
    def compute_envelope(freq, ratio):
        return np.inf if freq == 0 else (1.0/freq)*ratio
    direction=Dir
    width=Sz
    height=Sz
    duration=T_sz
    stimulus_fps=10
    centerx=.5
    centery=.5
    centert=.5
    spatial_freq=np.array([sp_frq])
    temporal_freq=np.array([tmp_frq])
    spatial_env=min(compute_envelope(1, .1),.5)
    temp_env=min(compute_envelope(1, .1),.5)
    
    x=np.linspace(0,1,width, endpoint=True)
    y=x
    t=np.linspace(0,1,duration, endpoint=True)
    Y,T,X=np.meshgrid(np.flip(y),t,x)
    
    centerx=.5
    centery=.5
    centert=.5
    
    filter_temporal_width = int(stimulus_fps*(2/3.))
    
    #% Gabor filter
    fh = -spatial_freq*np.cos(direction/180.*np.pi)
    fv = -spatial_freq*np.sin(direction/180.*np.pi)
    # normalize temporal frequency to wavelet size
    ft = np.real(temporal_freq*(filter_temporal_width/float(stimulus_fps)))
    
    gab_cos=np.cos(fh*(X-centerx)*2*np.pi+fv*(Y-centery)*2*np.pi+ft*(T-centert)*2*np.pi)*1/(np.sqrt(2*np.pi**3)*spatial_env*spatial_env*temp_env)*np.exp(-((X-centerx)**2+(Y-centery)**2)/(2*spatial_env**2))
    #    
    sine_wave=gab_cos
    # Show the video 
    if Vis==1:
        for tt in range(duration):
            plt.figure()
            plt.imshow(sine_wave[tt,:,:],cmap='gray',vmax=np.max(sine_wave),vmin=np.min(sine_wave))
            plt.axis('off')
    return sine_wave


dtype=torch.float32

#%%        
num_orientations=16
pyr=pyr3D_Gabor(UorA='A_pyr')


Directions=np.arange(-180,180,360/num_orientations)
angles=np.arange(-180,180,1)
Responses=[]
for aa in range(len(angles)):
    luminance_images=Make_sine_wave(Dir=angles[aa],Sz=128*2,T_sz=128,sp_frq=36,tmp_frq=36,Vis=0)
    luminance_images=torch.tensor(luminance_images).unsqueeze(0).to(torch.float32)
    pyr_coeffs=pyr.forward(luminance_images)
    Responses_o=[]
    # Each orientation response
    for dd in range(len(Directions)):
        Ks=[(0,ss,dd)  for ss in range(3)]
        res=np.zeros((1,128,128*2,128*2))
        for ii in range(len(Ks)):
            res+=pyr_coeffs[Ks[ii]].abs().numpy()**2
        Responses_o.append(np.mean(res))
    Responses.append(Responses_o)
    
#%%
Responses=np.array(Responses)
file=path+'anisotropy_pyr_num_cells_dir.npy'
num_cells=np.load(file)
plt.figure()
cmap=plt.cm.jet(np.linspace(0,1,num_orientations))
for ii in range(0,num_orientations):
    idx=np.argwhere((Directions[ii]-Directions)==0)[1:]
    
    plt.plot(angles,Responses[:,ii]*num_cells[ii],color=cmap[ii])
    plt.xticks(np.arange(-180,180,45),fontsize=15)
    plt.xlabel('Direction (Deg)',fontsize=20)
    plt.ylabel(r'$n_i$$m_i$',fontsize=20)
    plt.title('ME anisotropy',fontsize=30)


#%%        
num_orientations=16
pyr=pyr3D_Gabor(UorA='U_pyr')

Directions=np.arange(-180,180,360/num_orientations)

Responses=[]
for aa in range(len(angles)):
    luminance_images=Make_sine_wave(Dir=angles[aa],Sz=128*2,T_sz=128,sp_frq=36,tmp_frq=36,Vis=0)
    luminance_images=torch.tensor(luminance_images).unsqueeze(0).to(torch.float32)
    pyr_coeffs=pyr.forward(luminance_images)
    Responses_o=[]
    # Each orientation response
    for dd in range(len(Directions)):
        Ks=[(0,ss,dd)  for ss in range(3)]
        res=np.zeros((1,128,128*2,128*2))
        for ii in range(len(Ks)):
            res+=pyr_coeffs[Ks[ii]].abs().numpy()**2
        Responses_o.append(np.mean(res))
    Responses.append(Responses_o)
#%
Responses=np.array(Responses)
    
plt.figure()

cmap=plt.cm.jet(np.linspace(0,1,num_orientations))
for ii in range(0,num_orientations):
    idx=np.argwhere((Directions[ii]-Directions)==0)[1:]
    
    plt.plot(angles,Responses[:,ii]*np.mean(num_cells),color=cmap[ii])
    plt.xticks(np.arange(-180,180,45),fontsize=15)
    plt.xlabel('Direction (Deg)',fontsize=20)
    plt.ylabel(r'$\bar{n}\bar{m_i}$',fontsize=20)
    plt.title('ME tuning isotropy',fontsize=30)