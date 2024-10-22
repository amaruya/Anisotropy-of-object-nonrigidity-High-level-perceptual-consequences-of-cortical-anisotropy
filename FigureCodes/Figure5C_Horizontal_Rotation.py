#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 13:49:44 2024

@author: akihitomaruya
"""

"""
This is a code to compute the optic flow for Figure 5C for the horizontal rotation. It also computes the optic flow for different stretches and 
the differential invariants for Figure 6. 
"""


import matplotlib.colors
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import torch
import os

current_folder = os.path.dirname(os.path.abspath(__file__))


toolbox_path=current_folder+'/Toolbox/'
import sys
sys.path.append(toolbox_path)

from Vis_vec_field import *
from Compute_optic_flow_from_3D_gabor import *
def rotate(x,y,rot=np.pi/2):
    new_x=np.cos(rot)*x-np.sin(rot)*y
    new_y=np.sin(rot)*x+y*np.cos(rot)
    return new_x,new_y

Stretches=np.linspace(1,1.5,11)
#%% Make a stimulus
for ss in range(len(Stretches)):
    
   #%%
    sz_x=128*2
    sz_y=128*2
    duration=128
    num_points=500
    Omega=np.linspace(0,180,duration,endpoint=False)*np.pi/180
    phi=30*np.pi/180
    stretch_factor_X=Stretches[ss]
    stretch_factor_Y=1
    theta=np.linspace(0,2*np.pi,num_points)
    X=np.cos(phi)*np.cos(theta)
    Y=np.sin(phi)*np.cos(theta)
    Z=np.sin(theta)
    pos1=(np.outer(np.cos(Omega),X)+np.outer(np.sin(Omega),Z))*stretch_factor_Y/(stretch_factor_X*stretch_factor_Y)#*stretch_factor_X/np.sqrt(stretch_factor_X**2+stretch_factor_Y**2)
    pos2=np.outer(np.ones_like(Omega),Y)*stretch_factor_X/(stretch_factor_X*stretch_factor_Y)#*stretch_factor_Y/np.sqrt(stretch_factor_X**2+stretch_factor_Y**2)
    
    vec1=(-np.outer(np.sin(Omega),X)+np.outer(np.cos(Omega),Z))*stretch_factor_Y/(stretch_factor_X*stretch_factor_Y)
    vec2=np.zeros_like(vec1)*stretch_factor_X/(stretch_factor_X*stretch_factor_Y)
    
    
   
    Video,ex_idx=make_rotating_stim(pos1,pos2,vec1,-vec2,name=f'Horizontal_rt_im_stretch_{stretch_factor_X}',sz_x=sz_x,sz_y=sz_y,scale=.01)
    
    pos1w=(np.outer(np.cos(Omega)**2+np.sin(Omega)**2,np.cos(theta))+(1-np.cos(phi))*np.outer(np.sin(Omega*2)/2,np.sin(theta)))*stretch_factor_Y/(stretch_factor_X*stretch_factor_Y)
    pos2w=(np.sin(phi)*(np.outer(np.cos(Omega),np.cos(theta))-np.outer(np.sin(Omega),np.sin(theta))))*stretch_factor_X/(stretch_factor_X*stretch_factor_Y)
    vec1w=(1-np.cos(phi))*(2*np.outer(np.sin(Omega)*np.cos(Omega),np.cos(theta))+np.outer(np.cos(2*Omega),np.sin(theta)))*stretch_factor_Y/(stretch_factor_X*stretch_factor_Y)
    vec2w=-np.sin(phi)*(np.outer(np.sin(Omega),np.cos(theta))+np.outer(np.cos(Omega),np.sin(theta)))*stretch_factor_X/(stretch_factor_X*stretch_factor_Y)
    
    
    
    
    #%% get the gradient field 
    curl1=-(-np.outer(np.cos(Omega)*np.cos(phi),np.sin(theta))+np.outer(np.sin(Omega),np.cos(theta)))*stretch_factor_Y/(stretch_factor_X*stretch_factor_Y)
    curl2=-(-np.outer(np.sin(phi)*np.ones_like(Omega),np.sin(theta)))*stretch_factor_X/(stretch_factor_X*stretch_factor_Y)
    
    div1=np.outer(np.sin(phi)*np.ones_like(Omega),np.sin(theta))*stretch_factor_Y/(stretch_factor_X*stretch_factor_Y)
    div2=(-np.outer(np.cos(Omega)*np.cos(phi),np.sin(theta))+np.outer(np.sin(Omega),np.cos(theta)))*stretch_factor_X/(stretch_factor_X*stretch_factor_Y)
    
    
    def1_1=-curl1
    def1_2=curl2
    
    def2_1=-div1
    def2_2=div2
    
    
    
    curlw1=-(-np.outer(np.cos(Omega)**2+np.sin(Omega)**2,np.sin(theta))+(1-np.cos(phi))/2*np.outer(np.sin(2*Omega),np.cos(theta)))*stretch_factor_Y/(stretch_factor_X*stretch_factor_Y)
    curlw2=-(np.sin(phi)*(-np.outer(np.cos(Omega),np.sin(theta))-np.outer(np.sin(Omega),np.cos(theta))))*stretch_factor_X/(stretch_factor_X*stretch_factor_Y)
    
    divw1=-np.sin(phi)*(-np.outer(np.cos(Omega),np.sin(theta))-np.outer(np.sin(Omega),np.cos(theta)))*stretch_factor_Y/(stretch_factor_X*stretch_factor_Y)
    divw2=-(np.outer(np.cos(Omega)**2*np.cos(phi)+np.sin(Omega)**2,np.sin(theta))-(1-np.cos(phi))/2*np.outer(np.sin(2*Omega),np.cos(theta)))*stretch_factor_X/(stretch_factor_X*stretch_factor_Y)
    
    defw1_1=-curlw1
    defw1_2=curlw2
    
    defw2_1=-divw1
    defw2_2=divw2
    
    
    #%% Get line integral for both cases
    #%% Line integral
    int_div=np.sum(div1*vec1+div2*vec2,axis=1)
    int_curl=np.sum(curl1*vec1+curl2*vec2,axis=1)
    int_def1=np.sum(def1_1*vec1+def1_2*vec2,axis=1)
    int_def2=np.sum(def2_1*vec1+def2_2*vec2,axis=1)
    int_def=np.sqrt(int_def1**2+int_def2**2)
    norm=np.max((np.abs(int_def),np.abs(int_curl),np.abs(int_div)))
    int_div=int_div/norm
    int_curl=int_curl/norm
    int_def=int_def/norm
    
    #%%
    
    dtype=torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    VideoR=torch.tensor(Video).reshape(1,duration,sz_y,sz_x).type(dtype) 
    
    pyr_flow=Heeger_pyr_flow(UorA='U_pyr',
                    sname=current_folder+f'/Images/Steer_pyr_opticflow_H_ortho_im_stretch_{stretch_factor_X}',
                    vfile=current_folder+'/Videos/',
                    name=f'Optic_flow__H_ortho_im_stretch_{stretch_factor_X}',scale=.5,fps=10,smoothness=False,alpha=100,ex_idx=ex_idx,num_scales=2)
    
    
    u_hatU,v_hatU,U_o,V_o=pyr_flow.forward(VideoR)
    
    #%%
    
    U=u_hatU[ex_idx[0,:],ex_idx[1,:],ex_idx[2,:]].reshape(duration,-1)
    V=-v_hatU[ex_idx[0,:],ex_idx[1,:],ex_idx[2,:]].reshape(duration,-1)
   
    
    
    num_cut=5
    int_div_u=np.sum(div1*U+div2*V,axis=1)[num_cut:-num_cut]
    int_curl_u=np.sum(curl1*U+curl2*V,axis=1)[num_cut:-num_cut]
    int_def1_u=np.sum(def1_1*U+def1_2*V,axis=1)[num_cut:-num_cut]
    int_def2_u=np.sum(def2_1*U+def2_2*V,axis=1)[num_cut:-num_cut]
    int_def_u=np.sqrt(int_def1_u**2+int_def2_u**2)
    norm=np.max((np.abs(int_def_u).max(),np.abs(int_curl_u).max(),np.abs(int_div_u).max()))
    int_div_u=int_div_u/norm
    int_curl_u=int_curl_u/norm
    int_def_u=int_def_u/norm
    
    
    
    
    #%% Apply Anisotropy
    
    
    #normal=np.load(file)
    file=toolbox_path+'/Data/'+'anisotropy_pyr_num_cells_dir.npy'
    num_cells=np.load(file)
    num_cells=num_cells/num_cells.mean()
    
    
    #%%
    dtype=torch.float32
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    
    
    pyr_flow=Heeger_pyr_flow(UorA='A_pyr',
                    sname=current_folder+f'/Images/Steer_pyr_opticflow_H_aniso_ortho_im_stretch_{stretch_factor_X}',
                    vfile=current_folder+'/Videos/',
                    name=f'Optic_flow_aniso_H_ortho_im_stretch_{stretch_factor_X}',scale=.5,fps=10,smoothness=False,alpha=100,ex_idx=ex_idx,num_scales=2,num_cells=num_cells)
    
    
    u_hat,v_hat,U_o,V_o=pyr_flow.forward(VideoR)
    
    
    #%%
    U=u_hat[ex_idx[0,:],ex_idx[1,:],ex_idx[2,:]].reshape(duration,-1)
    V=-v_hat[ex_idx[0,:],ex_idx[1,:],ex_idx[2,:]].reshape(duration,-1)
   
    
    num_cut=5
    int_div_a=np.sum(div1*U+div2*V,axis=1)[num_cut:-num_cut]
    int_curl_a=np.sum(curl1*U+curl2*V,axis=1)[num_cut:-num_cut]
    int_def1_a=np.sum(def1_1*U+def1_2*V,axis=1)[num_cut:-num_cut]
    int_def2_a=np.sum(def2_1*U+def2_2*V,axis=1)[num_cut:-num_cut]
    int_def_a=np.sqrt(int_def1_a**2+int_def2_a**2)
    
    
    
    norm=np.max((np.abs(int_def_a).max(),np.abs(int_curl_a).max(),np.abs(int_div_a).max()))
    
    int_div_a=int_div_a/norm
    int_curl_a=int_curl_a/norm
    int_def_a=int_def_a/norm
    
   
