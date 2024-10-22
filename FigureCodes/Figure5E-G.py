#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is the code for Figure 5E-G, which can compute the physical motion and visualize it as a function of wobbling weight k, and 
compute the cosine similarity for iso- and anisotropic cortecies optic flows.
If you need to visualize the physical motion, set vis=1 in template_modelV and H.


"""

import matplotlib.colors
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import torch
import os 
current_folder = os.path.dirname(os.path.abspath(__file__))
path=current_folder+'/Toolbox/Data/'
toolbox_path=current_folder+'/Toolbox/'
import sys
sys.path.append(toolbox_path)
from Vis_vec_field import *
from Compute_optic_flow_from_pyr_with_smoothness_constraint import *

def rotate(x,y,rot=-np.pi/2):
    new_x=np.cos(rot)*x-np.sin(rot)*y
    new_y=np.sin(rot)*x+y*np.cos(rot)
    return new_x,new_y

sz_x=128*2
sz_y=128*2
duration=128
num_points=500
Omega=np.linspace(0,180,duration,endpoint=False)
phi=30
theta=np.linspace(0,2*np.pi,num_points)
num_cut=10

#Pose of Object
R_y=lambda Omega:[ np.array([[np.cos(Omega[tt]* np.pi/180),0,np.sin(Omega[tt]* np.pi/180)],
                            [0,1,0],
                            [-np.sin(Omega[tt]* np.pi/180),0,np.cos(Omega[tt]* np.pi/180)]]).reshape(1,3,3) for tt in range(len(Omega))]
#Elevation
R_z=lambda PhiE: np.array([[np.cos(-PhiE* np.pi/180),-np.sin(-PhiE* np.pi/180),0],
                           [np.sin(-PhiE* np.pi/180),np.cos(-PhiE* np.pi/180),0],
                            [0,0,1]]).reshape(1,3,3)
# Camera Elevation
R_x=lambda PhiC:np.array([[1,0,0],
                          [0,np.cos(PhiC* np.pi/180),-np.sin(PhiC* np.pi/180)],
                          [0,np.sin(PhiC* np.pi/180),np.cos(PhiC* np.pi/180)]]).reshape(1,3,3)

S_x=lambda w: np.array([[0,0,w],[0,0,0],[-w,0,0]]).reshape(1,3,3)

S_y=lambda w: np.array([[0,0,w],[0,0,0],[-w,0,0]]).reshape(1,3,3)

S_z =lambda w: np.array([[0,-w,0],[w,0,0],[0,0,0]]).reshape(1,3,3)

#%% Horizontal Template Model
X=np.cos(theta)
Y=np.zeros_like(X)
Z=np.sin(theta)

def template_modelH(k,vis=0,name='H_rot'):
    
    Vel_mat=(S_y(1)@np.concatenate(R_y(Omega),axis=0)@R_z(phi)@np.concatenate(R_y(-Omega*k),axis=0)+np.concatenate(R_y(Omega),axis=0)@R_z(phi)@S_y(-k)@np.concatenate(R_y(-Omega*k),axis=0)
             )@np.concatenate(R_y(Omega*k),axis=0)
    Pos_mat=np.concatenate(R_y(Omega),axis=0)@R_z(phi)@np.concatenate(R_y(-Omega*k),axis=0)@np.concatenate(R_y(Omega*k),axis=0)
    
    pos1W=np.vstack([Pos_mat[tt,0,0]*X+Pos_mat[tt,0,1]*Y+Pos_mat[tt,0,2]*Z for tt in range(duration)])
    pos2W=np.vstack([Pos_mat[tt,1,0]*X+Pos_mat[tt,1,1]*Y+Pos_mat[tt,1,2]*Z for tt in range(duration)])
    
    vec1W=np.vstack([Vel_mat[tt,0,0]*X+Vel_mat[tt,0,1]*Y+Vel_mat[tt,0,2]*Z for tt in range(duration)])
    vec2W=np.vstack([Vel_mat[tt,1,0]*X+Vel_mat[tt,1,1]*Y+Vel_mat[tt,1,2]*Z for tt in range(duration)])
    if vis==1:
        Video,ex_idx=make_rotating_stim(pos1W,pos2W,vec1W,-vec2W,name=name,sz_x=sz_x,sz_y=sz_y,Vis=1)
    return vec1W[num_cut:-num_cut,:],-vec2W[num_cut:-num_cut,:]

def template_modelV(k,vis=0,name='V_rot'):
    
    Vel_mat=(S_y(1)@np.concatenate(R_y(Omega),axis=0)@R_z(phi)@np.concatenate(R_y(-Omega*k),axis=0)+np.concatenate(R_y(Omega),axis=0)@R_z(phi)@S_y(-k)@np.concatenate(R_y(-Omega*k),axis=0)
             )@np.concatenate(R_y(Omega*k),axis=0)
    Pos_mat=np.concatenate(R_y(Omega),axis=0)@R_z(phi)@np.concatenate(R_y(-Omega*k),axis=0)@np.concatenate(R_y(Omega*k),axis=0)
    
    pos1W=np.vstack([Pos_mat[tt,0,0]*X+Pos_mat[tt,0,1]*Y+Pos_mat[tt,0,2]*Z for tt in range(duration)])
    pos2W=np.vstack([Pos_mat[tt,1,0]*X+Pos_mat[tt,1,1]*Y+Pos_mat[tt,1,2]*Z for tt in range(duration)])
    
    vec1W=np.vstack([Vel_mat[tt,0,0]*X+Vel_mat[tt,0,1]*Y+Vel_mat[tt,0,2]*Z for tt in range(duration)])
    vec2W=np.vstack([Vel_mat[tt,1,0]*X+Vel_mat[tt,1,1]*Y+Vel_mat[tt,1,2]*Z for tt in range(duration)])
    
    pos1W,pos2W=rotate(pos1W,pos2W)
    vec1W,vec2W=rotate(vec1W,vec2W)
    if vis==1:
        Video,ex_idx=make_rotating_stim(pos1W,pos2W,vec1W,-vec2W,name=name,sz_x=sz_x,sz_y=sz_y,Vis=1)
    return vec1W[num_cut:-num_cut,:],-vec2W[num_cut:-num_cut,:]



#%% load estimate


# Horizontal
# Uniform
u_hatU_H=np.load(path+'u_hatU_H.npy')[num_cut:-num_cut,:]
v_hatU_H=np.load(path+'v_hatU_H.npy')[num_cut:-num_cut,:]
idxUH=np.sqrt(u_hatU_H**2+v_hatU_H**2)>0
unit_u_hatU_H=u_hatU_H[idxUH]/np.sqrt(u_hatU_H[idxUH]**2+v_hatU_H[idxUH]**2)
unit_v_hatU_H=v_hatU_H[idxUH]/np.sqrt(u_hatU_H[idxUH]**2+v_hatU_H[idxUH]**2)
# Aniso
u_hatA_H=np.load(path+'u_hatA_H.npy')[num_cut:-num_cut,:]
v_hatA_H=np.load(path+'v_hatA_H.npy')[num_cut:-num_cut,:]
idxAH=np.sqrt(u_hatA_H**2+v_hatA_H**2)>0
unit_u_hatA_H=u_hatA_H[idxAH]/np.sqrt(u_hatA_H[idxAH]**2+v_hatA_H[idxAH]**2)
unit_v_hatA_H=v_hatA_H[idxAH]/np.sqrt(u_hatA_H[idxAH]**2+v_hatA_H[idxAH]**2)

# Vertical 
# Uniform
u_hatU_V=np.load(path+'u_hatU_V.npy')[num_cut:-num_cut,:]
v_hatU_V=np.load(path+'v_hatU_V.npy')[num_cut:-num_cut,:]
idxUV=np.sqrt(u_hatU_V**2+v_hatU_V**2)>0
unit_u_hatU_V=u_hatU_V[idxUV]/np.sqrt(u_hatU_V[idxUV]**2+v_hatU_V[idxUV]**2)
unit_v_hatU_V=v_hatU_V[idxUV]/np.sqrt(u_hatU_V[idxUV]**2+v_hatU_V[idxUV]**2)
# Aniso
u_hatA_V=np.load(path+'u_hatA_V.npy')[num_cut:-num_cut,:]
v_hatA_V=np.load(path+'v_hatA_V.npy')[num_cut:-num_cut,:]
idxAV=np.sqrt(u_hatA_V**2+v_hatA_V**2)>0
unit_u_hatA_V=u_hatA_V[idxAV]/np.sqrt(u_hatA_V[idxAV]**2+v_hatA_V[idxAV]**2)
unit_v_hatA_V=v_hatA_V[idxAV]/np.sqrt(u_hatA_V[idxAV]**2+v_hatA_V[idxAV]**2)





# examine 1000 ks
num_k=1000
ks=np.linspace(0,1,num_k)
Hcos_sim_meanU=[]
Hcos_sim_stdU=[]
Hcos_sim_meanA=[]
Hcos_sim_stdA=[]

Vcos_sim_meanU=[]
Vcos_sim_stdU=[]
Vcos_sim_meanA=[]
Vcos_sim_stdA=[]

for kk in range(len(ks)):
    # Horizontal
    vec1_k,vec2_k=template_modelH(k=ks[kk])
    unit_vec1_kU=vec1_k[idxUH]/np.sqrt(vec1_k[idxUH]**2+vec2_k[idxUH]**2)
    unit_vec2_kU=vec2_k[idxUH]/np.sqrt(vec1_k[idxUH]**2+vec2_k[idxUH]**2)
    Hcos_sim_meanU.append(np.mean(unit_u_hatU_H*unit_vec1_kU+unit_v_hatU_H*unit_vec2_kU))
    Hcos_sim_stdU.append(np.std(unit_u_hatU_H*unit_vec1_kU+unit_v_hatU_H*unit_vec2_kU))
    
    
    unit_vec1_kA=vec1_k[idxAH]/np.sqrt(vec1_k[idxAH]**2+vec2_k[idxAH]**2)
    unit_vec2_kA=vec2_k[idxAH]/np.sqrt(vec1_k[idxAH]**2+vec2_k[idxAH]**2)
    Hcos_sim_meanA.append(np.mean(unit_u_hatA_H*unit_vec1_kA+unit_v_hatA_H*unit_vec2_kA))
    Hcos_sim_stdA.append(np.std(unit_u_hatA_H*unit_vec1_kA+unit_v_hatA_H*unit_vec2_kA))
    
    # Vertical
    vec1_k,vec2_k=template_modelV(k=ks[kk])
    unit_vec1_kU=vec1_k[idxUV]/np.sqrt(vec1_k[idxUV]**2+vec2_k[idxUV]**2)
    unit_vec2_kU=vec2_k[idxUV]/np.sqrt(vec1_k[idxUV]**2+vec2_k[idxUV]**2)
    Vcos_sim_meanU.append(np.mean(unit_u_hatU_V*unit_vec1_kU+unit_v_hatU_V*unit_vec2_kU))
    Vcos_sim_stdU.append(np.std(unit_u_hatU_V*unit_vec1_kU+unit_v_hatU_V*unit_vec2_kU))
    
    
    unit_vec1_kA=vec1_k[idxAV]/np.sqrt(vec1_k[idxAV]**2+vec2_k[idxAV]**2)
    unit_vec2_kA=vec2_k[idxAV]/np.sqrt(vec1_k[idxAV]**2+vec2_k[idxAV]**2)
    Vcos_sim_meanA.append(np.mean(unit_u_hatA_V*unit_vec1_kA+unit_v_hatA_V*unit_vec2_kA))
    Vcos_sim_stdA.append(np.std(unit_u_hatA_V*unit_vec1_kA+unit_v_hatA_V*unit_vec2_kA))
    
Hcos_sim_meanU=np.hstack(Hcos_sim_meanU)
Hcos_sim_stdU=np.hstack(Hcos_sim_stdU)
Hcos_sim_meanA=np.hstack(Hcos_sim_meanA)
Hcos_sim_stdA=np.hstack(Hcos_sim_stdA)

Vcos_sim_meanU=np.hstack(Vcos_sim_meanU)
Vcos_sim_stdU=np.hstack(Vcos_sim_stdU)
Vcos_sim_meanA=np.hstack(Vcos_sim_meanA)
Vcos_sim_stdA=np.hstack(Vcos_sim_stdA)






#%%

# plot Uniform 
plt.figure(figsize=(10,10))
plt.plot(ks,Hcos_sim_meanU,'b-',label='H Mean',linewidth=5)
plt.fill_between(ks, Hcos_sim_meanU-Hcos_sim_stdU/2,Hcos_sim_meanU+Hcos_sim_stdU/2,alpha=0.2 , color='lightblue',label='H std')
plt.plot(ks,Vcos_sim_meanU,'r-',label='V Mean',linewidth=5)
plt.fill_between(ks, Vcos_sim_meanU-Vcos_sim_stdU/2,Vcos_sim_meanU+Vcos_sim_stdU/2,alpha=0.2 , color='deeppink',label='V std')
plt.axline((ks[np.argmax(Hcos_sim_meanU)], 0), (ks[np.argmax(Hcos_sim_meanU)], 1), linestyle='--', color='b',label='H Max')
plt.axline((ks[np.argmax(Vcos_sim_meanU)], 0), (ks[np.argmax(Vcos_sim_meanU)], 1), linestyle='--', color='r',label='V Max')

plt.xticks(np.arange(0,1.1,.2),fontsize=25)
plt.yticks(np.arange(0,1.1,.5),fontsize=25)
plt.xlabel(r'$k$',fontsize=30)
plt.ylabel('Cosine similarity',fontsize=30)
plt.legend(loc='lower center',fontsize=22)
plt.title('Isotropic Cortex',fontsize=35)

plt.show()
    
    
# plot Aniso
plt.figure(figsize=(10,10))
plt.plot(ks,Hcos_sim_meanA,'b-',label='H Mean',linewidth=5)
plt.fill_between(ks, Hcos_sim_meanA-Hcos_sim_stdA/2,Hcos_sim_meanA+Hcos_sim_stdA/2,alpha=0.2 , color='lightblue',label='H std')
plt.plot(ks,Vcos_sim_meanA,'r-',label='V Mean',linewidth=5)
plt.fill_between(ks, Vcos_sim_meanA-Vcos_sim_stdA/2,Vcos_sim_meanA+Vcos_sim_stdA/2,alpha=0.2 , color='deeppink',label='V std')
plt.axline((ks[np.argmax(Hcos_sim_meanA)], 0), (ks[np.argmax(Hcos_sim_meanA)], 1), linestyle='--', color='b',label='H Max')
plt.axline((ks[np.argmax(Vcos_sim_meanA)], 0), (ks[np.argmax(Vcos_sim_meanA)], 1), linestyle='--', color='r',label='V Max')

plt.xticks(np.arange(0,1.1,.2),fontsize=25)
plt.yticks(np.arange(0,1.1,.5),fontsize=25)
plt.xlabel(r'$k$',fontsize=30)
plt.ylabel('Cosine similarity',fontsize=30)
plt.legend(loc='lower center', fontsize=22, bbox_to_anchor=(0.6, 0))
plt.title('Anisotropic Cortex',fontsize=35)

plt.show()

