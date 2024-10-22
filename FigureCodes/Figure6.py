#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


This is a code to compute the figure 6. 
First three are for physical differential invariants. 
Next two are for the best fit physical differential invariants to the ME optic flow invariants.
The last two are for the best k across different amount of stretches.


"""


import matplotlib.colors
import numpy as np
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
import torch
current_folder = os.path.dirname(os.path.abspath(__file__))
path=current_folder+'/Toolbox/Data/'
toolbox_path=current_folder+'/Toolbox/'
import sys
sys.path.append(toolbox_path)
from Vis_vec_field import *

def rotate(x,y,rot=np.pi/2):
    new_x=np.cos(rot)*x-np.sin(rot)*y
    new_y=np.sin(rot)*x+y*np.cos(rot)
    return new_x,new_y
#%% Make a stimulus
sz_x=128*2
sz_y=128*2
duration=128
num_points=50
Omega=np.linspace(0,180,duration,endpoint=False)*np.pi/180
phi=30*np.pi/180
theta=np.linspace(0,2*np.pi,num_points)
X=np.cos(phi)*np.cos(theta)
Y=np.sin(phi)*np.cos(theta)
Z=np.sin(theta)
ks=np.arange(0,1.1,.05)#np.array([0,1])#np.hstack((np.arange(0,1.1,.2),np.array([.5])))#np.array([.5])#np.arange(0,1.1,.2)
rot=0#np.pi/2
DIV=[]
CURL=[]
DEF=[]
Vis=0
for ii in range(len(ks)):
    pos1w_=np.outer(np.cos(Omega)*np.cos(ks[ii]*Omega)+np.sin(Omega)*np.sin(ks[ii]*Omega),np.cos(theta))+np.outer(np.sin(Omega)*np.cos(ks[ii]*Omega)-np.sin(ks[ii]*Omega)*np.cos(Omega)*np.cos(phi),np.sin(theta))
    pos2w_=np.sin(phi)*(np.outer(np.cos(ks[ii]*Omega),np.cos(theta))-np.outer(np.sin(ks[ii]*Omega),np.sin(theta)))
    
    pos1w=pos1w_*np.cos(rot)-pos2w_*np.sin(rot)
    pos2w=pos1w_*np.sin(rot)+pos2w_*np.cos(rot)
    vec1w_=np.outer(-np.sin(Omega)*np.cos(ks[ii]*Omega)*np.cos(phi)-ks[ii]*np.cos(Omega)*np.sin(ks[ii]*Omega)*np.cos(phi)+np.cos(Omega)*np.sin(ks[ii]*Omega)+ks[ii]*np.sin(Omega)*np.cos(ks[ii]*Omega),np.cos(theta))\
        +np.outer(np.cos(Omega)*np.cos(ks[ii]*Omega)-ks[ii]*np.sin(Omega)*np.sin(ks[ii]*Omega)-ks[ii]*np.cos(ks[ii]*Omega)*np.cos(Omega)*np.cos(phi)+np.sin(ks[ii]*Omega)*np.sin(Omega)*np.cos(phi),np.sin(theta))
    vec2w_=-np.sin(phi)*(np.outer(np.sin(ks[ii]*Omega),np.cos(theta))+np.outer(np.cos(ks[ii]*Omega),np.sin(theta)))*ks[ii]
    vec1w=vec1w_*np.cos(rot)-vec2w_*np.sin(rot)
    vec2w=vec1w_*np.sin(rot)+vec2w_*np.cos(rot)
    
        
    
    
    curlw1=-(-np.outer(np.cos(Omega)*np.cos(ks[ii]*Omega)+np.sin(Omega)*np.sin(ks[ii]*Omega),np.sin(theta))+np.outer(np.sin(Omega)*np.cos(ks[ii]*Omega)-np.sin(ks[ii]*Omega)*np.cos(Omega)*np.cos(phi),np.cos(theta)))
    curlw2=-(np.sin(phi)*(-np.outer(np.cos(ks[ii]*Omega),np.sin(theta))-np.outer(np.sin(ks[ii]*Omega),np.cos(theta))))
    
    
    divw1=curlw2
    divw2=- curlw1

    defw1_1=curlw1
    defw1_2=-curlw2

    defw2_1=divw1
    defw2_2=-divw2
    if Vis==1:
        name=f'opt_vertically_wobbling_fields_with_k:{ks[ii]}'
        vis_vec(pos1w,pos2w,vec1w,vec2w,name,scale=1)
        name=f'Curl_fields_wobble_k:{ks[ii]}'
        vis_vec(pos1w,pos2w,curlw1,curlw2,name,scale=2)
        
        name=f'Div_fields_wobble_k:{ks[ii]}'
        vis_vec(pos1w,pos2w,divw1,divw2,name,scale=2)
        
        name=f'Def1_field_wobble:{ks[ii]}'
        vis_vec(pos1w,pos2w,defw1_1,defw1_2,name,scale=2)
        
        name=f'Def2_field_wobble:{ks[ii]}'
        vis_vec(pos1w,pos2w,defw2_1,defw2_2,name,scale=2)
    
    int_divw=np.sum(divw1*vec1w+divw2*vec2w,axis=1)
    int_curlw=np.sum(curlw1*vec1w+curlw2*vec2w,axis=1)
    int_def1w=np.sum(defw1_1*vec1w+defw1_2*vec2w,axis=1)
    int_def2w=np.sum(defw2_1*vec1w+defw2_2*vec2w,axis=1)
    int_defw=np.sqrt(int_def1w**2+int_def2w**2)
    norm=np.max((np.abs(int_defw),np.abs(int_curlw),np.abs(int_divw)))
    int_divw=int_divw/norm
    int_curlw=int_curlw/norm
    int_defw=int_defw/norm
    
    DIV.append(int_divw)
    CURL.append(int_curlw)
    DEF.append(int_defw)
DIV=np.vstack(DIV)
CURL=np.vstack(CURL)
DEF=np.vstack(DEF)
#%% Plot


num_plots = np.shape(DIV)[0]

cm_subsection = np.linspace(0, 1, num_plots) 

colors = [ cm.jet(x) for x in cm_subsection ]
fig, ax = plt.subplots()
for ii in range(num_plots):
    plt.plot(Omega*180/np.pi,DIV[ii,:],color=colors[ii])
plt.xticks(np.arange(0,181,90))

plt.xlabel('Motion phase (deg)')
plt.ylabel('Divergence')
plt.ylim([-1.1,1.1])
plt.title('Transition from wobbling to rotation')
cbar = plt.colorbar(cm.ScalarMappable(cmap=cm.jet), ax=ax)
cbar.set_label(r'$k$')
tick_labels=['0 (rotation)','','1 (wobbling)']
cbar.set_ticks([0, .5,1])
cbar.set_ticklabels(tick_labels)


fig, ax = plt.subplots()
for ii in range(num_plots):
    plt.plot(Omega*180/np.pi,CURL[ii,:],color=colors[ii])
plt.xticks(np.arange(0,181,90))

plt.xlabel('Motion phase (deg)')
plt.ylabel('Curl')
plt.ylim([-1.1,1.1])
plt.title('Transition from wobbling to rotation')
cbar = plt.colorbar(cm.ScalarMappable(cmap=cm.jet), ax=ax)
cbar.set_label(r'$k$')
tick_labels=['0 (rotation)','','1 (wobbling)']
cbar.set_ticks([0, .5,1])
cbar.set_ticklabels(tick_labels)


fig, ax = plt.subplots()
for ii in range(num_plots):
    plt.plot(Omega*180/np.pi,DEF[ii,:],color=colors[ii])
plt.xticks(np.arange(0,181,90))

plt.xlabel('Motion phase (deg)')
plt.ylabel('Deformation')
plt.ylim([-1.1,1.1])
plt.title('Transition from wobbling to rotation')
cbar = plt.colorbar(cm.ScalarMappable(cmap=cm.jet), ax=ax)
cbar.set_label(r'$k$')
tick_labels=['0 (rotation)','','1 (wobbling)']
cbar.set_ticks([0, .5,1])
cbar.set_ticklabels(tick_labels)




sz_x=128*2
sz_y=128*2
duration=128
num_points=500
OmegaA=np.linspace(0,180,duration,endpoint=False)*np.pi/180
Omega=OmegaA[5:-5]
phi=30*np.pi/180
theta=np.linspace(0,2*np.pi,num_points)
X=np.cos(phi)*np.cos(theta)
Y=np.sin(phi)*np.cos(theta)
Z=np.sin(theta)

# Define the objective function to minimize
def objectiveH(params,DIV,CURL,DEF,stretch=1):
    k = params
    stretch_factor_X=stretch
    stretch_factor_Y=1
    st_x=stretch_factor_X/(stretch_factor_X*stretch_factor_Y)
    st_y=stretch_factor_Y/(stretch_factor_X*stretch_factor_Y)
    curlw1=-(-np.outer(np.cos(Omega)*np.cos(k*Omega)+np.sin(Omega)*np.sin(k*Omega),np.sin(theta))+np.outer(np.sin(Omega)*np.cos(k*Omega)-np.sin(k*Omega)*np.cos(Omega)*np.cos(phi),np.cos(theta)))*st_x
    curlw2=-(np.sin(phi)*(-np.outer(np.cos(k*Omega),np.sin(theta))-np.outer(np.sin(k*Omega),np.cos(theta))))*st_y
    vec1w=(np.outer(-np.sin(Omega)*np.cos(k*Omega)*np.cos(phi)-k*np.cos(Omega)*np.sin(k*Omega)*np.cos(phi)+np.cos(Omega)*np.sin(k*Omega)+k*np.sin(Omega)*np.cos(k*Omega),np.cos(theta))\
        +np.outer(np.cos(Omega)*np.cos(k*Omega)-k*np.sin(Omega)*np.sin(k*Omega)-k*np.cos(k*Omega)*np.cos(Omega)*np.cos(phi)+np.sin(k*Omega)*np.sin(Omega)*np.cos(phi),np.sin(theta)))*st_x
    vec2w=-np.sin(phi)*(np.outer(np.sin(k*Omega),np.cos(theta))+np.outer(np.cos(k*Omega),np.sin(theta)))*k*st_y
    
    divw1=curlw2
    divw2=- curlw1

    defw1_1=curlw1
    defw1_2=-curlw2

    defw2_1=divw1
    defw2_2=-divw2
    
    
    int_divw=np.sum(divw1*vec1w+divw2*vec2w,axis=1)
    int_curlw=np.sum(curlw1*vec1w+curlw2*vec2w,axis=1)
    int_def1w=np.sum(defw1_1*vec1w+defw1_2*vec2w,axis=1)
    int_def2w=np.sum(defw2_1*vec1w+defw2_2*vec2w,axis=1)
    int_defw=np.sqrt(int_def1w**2+int_def2w**2)
    norm=np.max((np.abs(int_defw),np.abs(int_curlw),np.abs(int_divw)))
    int_divw=int_divw/norm
    int_curlw=int_curlw/norm
    int_defw=int_defw/norm
    
    diff=np.sum((CURL-int_curlw)**2)+np.sum((DIV-int_divw)**2)+np.sum((DEF-int_defw)**2)
    
    return diff

def objectiveV(params,DIV,CURL,DEF,stretch=1):
    k = params
    stretch_factor_X=1
    stretch_factor_Y=stretch
    st_x=stretch_factor_X/(stretch_factor_X*stretch_factor_Y)
    st_y=stretch_factor_Y/(stretch_factor_X*stretch_factor_Y)
    curlw1=-(-np.outer(np.cos(Omega)*np.cos(k*Omega)+np.sin(Omega)*np.sin(k*Omega),np.sin(theta))+np.outer(np.sin(Omega)*np.cos(k*Omega)-np.sin(k*Omega)*np.cos(Omega)*np.cos(phi),np.cos(theta)))*st_x
    curlw2=-(np.sin(phi)*(-np.outer(np.cos(k*Omega),np.sin(theta))-np.outer(np.sin(k*Omega),np.cos(theta))))*st_y
    vec1w=(np.outer(-np.sin(Omega)*np.cos(k*Omega)*np.cos(phi)-k*np.cos(Omega)*np.sin(k*Omega)*np.cos(phi)+np.cos(Omega)*np.sin(k*Omega)+k*np.sin(Omega)*np.cos(k*Omega),np.cos(theta))\
        +np.outer(np.cos(Omega)*np.cos(k*Omega)-k*np.sin(Omega)*np.sin(k*Omega)-k*np.cos(k*Omega)*np.cos(Omega)*np.cos(phi)+np.sin(k*Omega)*np.sin(Omega)*np.cos(phi),np.sin(theta)))*st_x
    vec2w=-np.sin(phi)*(np.outer(np.sin(k*Omega),np.cos(theta))+np.outer(np.cos(k*Omega),np.sin(theta)))*k*st_y
    
    divw1=curlw2
    divw2=- curlw1

    defw1_1=curlw1
    defw1_2=-curlw2

    defw2_1=divw1
    defw2_2=-divw2
    
    
    int_divw=np.sum(divw1*vec1w+divw2*vec2w,axis=1)
    int_curlw=np.sum(curlw1*vec1w+curlw2*vec2w,axis=1)
    int_def1w=np.sum(defw1_1*vec1w+defw1_2*vec2w,axis=1)
    int_def2w=np.sum(defw2_1*vec1w+defw2_2*vec2w,axis=1)
    int_defw=np.sqrt(int_def1w**2+int_def2w**2)
    norm=np.max((np.abs(int_defw),np.abs(int_curlw),np.abs(int_divw)))
    int_divw=int_divw/norm
    int_curlw=int_curlw/norm
    int_defw=int_defw/norm
    
    diff=np.sum((CURL-int_curlw)**2)+np.sum((DIV-int_divw)**2)+np.sum((DEF-int_defw)**2)
    
    return diff
def get_best_gradients(k):
    curlw1=-(-np.outer(np.cos(OmegaA)*np.cos(k*OmegaA)+np.sin(OmegaA)*np.sin(k*OmegaA),np.sin(theta))+np.outer(np.sin(OmegaA)*np.cos(k*OmegaA)-np.sin(k*OmegaA)*np.cos(OmegaA)*np.cos(phi),np.cos(theta)))
    curlw2=-(np.sin(phi)*(-np.outer(np.cos(k*OmegaA),np.sin(theta))-np.outer(np.sin(k*OmegaA),np.cos(theta))))
    vec1w=np.outer(-np.sin(OmegaA)*np.cos(k*OmegaA)*np.cos(phi)-k*np.cos(OmegaA)*np.sin(k*OmegaA)*np.cos(phi)+np.cos(OmegaA)*np.sin(k*OmegaA)+k*np.sin(OmegaA)*np.cos(k*OmegaA),np.cos(theta))\
        +np.outer(np.cos(OmegaA)*np.cos(k*OmegaA)-k*np.sin(OmegaA)*np.sin(k*OmegaA)-k*np.cos(k*OmegaA)*np.cos(OmegaA)*np.cos(phi)+np.sin(k*OmegaA)*np.sin(OmegaA)*np.cos(phi),np.sin(theta))
    vec2w=-np.sin(phi)*(np.outer(np.sin(k*OmegaA),np.cos(theta))+np.outer(np.cos(k*OmegaA),np.sin(theta)))*k
    
    divw1=curlw2
    divw2=- curlw1

    defw1_1=curlw1
    defw1_2=-curlw2

    defw2_1=divw1
    defw2_2=-divw2
    
    
    int_divw=np.sum(divw1*vec1w+divw2*vec2w,axis=1)
    int_curlw=np.sum(curlw1*vec1w+curlw2*vec2w,axis=1)
    int_def1w=np.sum(defw1_1*vec1w+defw1_2*vec2w,axis=1)
    int_def2w=np.sum(defw2_1*vec1w+defw2_2*vec2w,axis=1)
    int_defw=np.sqrt(int_def1w**2+int_def2w**2)
    norm=np.max((np.abs(int_defw),np.abs(int_curlw),np.abs(int_divw)))
    int_divw=int_divw/norm
    int_curlw=int_curlw/norm
    int_defw=int_defw/norm
    return int_divw,int_curlw,int_defw

#%% Load data
Stretches=np.linspace(1,1.5,11)
H_U_ks=[]
H_A_ks=[]
V_U_ks=[]
V_A_ks=[]

for ss in range(len(Stretches)):
    
    stretch_factor_X=Stretches[ss]
    DIV_V_U=np.load(path+f'V_div_uniform_im_stretch_{stretch_factor_X}.npy')
    CURL_V_U=np.load(path+f'V_curl_uniform_im_stretch_{stretch_factor_X}.npy')
    DEF_V_U=np.load(path+f'V_def_uniform_im_stretch_{stretch_factor_X}.npy')
    
    
    DIV_V_A=np.load(path+f'V_div_aniso_im_stretch_{stretch_factor_X}.npy')
    CURL_V_A=np.load(path+f'V_curl_aniso_im_stretch_{stretch_factor_X}.npy')
    DEF_V_A=np.load(path+f'V_def_aniso_im_stretch_{stretch_factor_X}.npy')
    
    DIV_H_U=np.load(path+f'H_div_uniform_im_stretch_{stretch_factor_X}.npy')
    CURL_H_U=np.load(path+f'H_curl_uniform_im_stretch_{stretch_factor_X}.npy')
    DEF_H_U=np.load(path+f'H_def_uniform_im_stretch_{stretch_factor_X}.npy')
    
    
    DIV_H_A=np.load(path+f'H_div_aniso_im_stretch_{stretch_factor_X}.npy')
    CURL_H_A=np.load(path+f'H_curl_aniso_im_stretch_{stretch_factor_X}.npy')
    DEF_H_A=np.load(path+f'H_def_aniso_im_stretch_{stretch_factor_X}.npy')
    
    if ss==0:
        DIV_V_U1=np.load(path+f'V_div_uniform_im_stretch_{stretch_factor_X}.npy')
        CURL_V_U1=np.load(path+f'V_curl_uniform_im_stretch_{stretch_factor_X}.npy')
        DEF_V_U1=np.load(path+f'V_def_uniform_im_stretch_{stretch_factor_X}.npy')
        
        
        DIV_V_A1=np.load(path+f'V_div_aniso_im_stretch_{stretch_factor_X}.npy')
        CURL_V_A1=np.load(path+f'V_curl_aniso_im_stretch_{stretch_factor_X}.npy')
        DEF_V_A1=np.load(path+f'V_def_aniso_im_stretch_{stretch_factor_X}.npy')
        
        DIV_H_U1=np.load(path+f'H_div_uniform_im_stretch_{stretch_factor_X}.npy')
        CURL_H_U1=np.load(path+f'H_curl_uniform_im_stretch_{stretch_factor_X}.npy')
        DEF_H_U1=np.load(path+f'H_def_uniform_im_stretch_{stretch_factor_X}.npy')
        
        
        DIV_H_A1=np.load(path+f'H_div_aniso_im_stretch_{stretch_factor_X}.npy')
        CURL_H_A1=np.load(path+f'H_curl_aniso_im_stretch_{stretch_factor_X}.npy')
        DEF_H_A1=np.load(path+f'H_def_aniso_im_stretch_{stretch_factor_X}.npy')
    
    
    
    # Initial guess for parameters
    initial_params = .5
    
    # Horizontal uniform 
    # Minimize the objective function
    result = minimize(objectiveH, initial_params, args=(DIV_H_U,CURL_H_U,  DEF_H_U,stretch_factor_X), method='BFGS')
    
    # Extract the optimized parameters
    optimized_params_H_U = result.x
    
    H_U_ks.append(optimized_params_H_U)
    Fit_DIV_H_U,Fit_CURL_H_U,Fit_DEF_H_U=get_best_gradients(optimized_params_H_U)
    
    
    
    
    
    # Horizontal aniso
    # Minimize the objective function
    result = minimize(objectiveH, initial_params, args=(DIV_H_A,CURL_H_A,  DEF_H_A,stretch_factor_X), method='BFGS')
    
    # Extract the optimized parameters
    optimized_params_H_A = result.x
    H_A_ks.append(optimized_params_H_A)
    Fit_DIV_H_A,Fit_CURL_H_A,Fit_DEF_H_A=get_best_gradients(optimized_params_H_A)
    
    
   
    
    # Vertical uniform 
    # Minimize the objective function
    result = minimize(objectiveV, initial_params, args=(DIV_V_U,CURL_V_U,  DEF_V_U,stretch_factor_X), method='BFGS')
    
    # Extract the optimized parameters
    optimized_params_V_U = result.x
    V_U_ks.append(optimized_params_V_U)
    Fit_DIV_V_U,Fit_CURL_V_U,Fit_DEF_V_U=get_best_gradients(optimized_params_V_U)
    
   
    
    
    # Horizontal aniso
    # Minimize the objective function
    result = minimize(objectiveV, initial_params, args=(DIV_V_A,CURL_V_A,  DEF_V_A,stretch_factor_X), method='BFGS')
    
    # Extract the optimized parameters
    optimized_params_V_A = result.x
    V_A_ks.append(optimized_params_V_A)
    Fit_DIV_V_A,Fit_CURL_V_A,Fit_DEF_V_A=get_best_gradients(optimized_params_V_A)
    
    
    
   
#%% Plot them as a function of amount of stretch
# Uniform
H_U_ks=np.hstack(H_U_ks)
H_A_ks=np.hstack(H_A_ks)
V_U_ks=np.hstack(V_U_ks)
V_A_ks=np.hstack(V_A_ks)


#%% Figure 6 
# plot isotropic cortex V
plt.figure(figsize=(18,9))
plt.subplot(1,2,1)
plt.plot(Omega*180/np.pi,DIV_V_U1,'r*')
div,curl,Def=get_best_gradients(V_U_ks[0])
plt.plot(OmegaA*180/np.pi,div,'r--')
plt.plot(Omega*180/np.pi,CURL_V_U1,'g*')
plt.plot(OmegaA*180/np.pi,curl,'g--')
plt.plot(Omega*180/np.pi,DEF_V_U1,'b*')
plt.plot(OmegaA*180/np.pi,Def,'--b')
plt.xticks([0,180],fontsize=15)
plt.yticks([-1,1],fontsize=15)
plt.xlabel('Motion Phase (Deg)',fontsize=30)
plt.ylabel('Gradients',fontsize=30)
plt.title('Vertical',fontsize=40)
plt.text(15,-0.8,r'$k= $'+f'{round(V_U_ks[0],2)}',fontsize=15)
# plot isotropic cortex H
plt.subplot(1,2,2)
plt.plot(Omega*180/np.pi,DIV_H_U1,'r*')
div,curl,Def=get_best_gradients(H_U_ks[0])
plt.plot(OmegaA*180/np.pi,div,'r--')
plt.plot(Omega*180/np.pi,CURL_H_U1,'g*')
plt.plot(OmegaA*180/np.pi,curl,'g--')
plt.plot(Omega*180/np.pi,DEF_H_U1,'b*')
plt.plot(OmegaA*180/np.pi,Def,'--b')
plt.xticks([0,180],fontsize=15)
plt.yticks([-1,1],fontsize=15)
plt.xlabel('Motion Phase (Deg)',fontsize=30)
plt.ylabel('Gradients',fontsize=30)
plt.title('Horizontal',fontsize=40)
plt.suptitle('Isotropic tuning',fontsize=40)
plt.text(15,-0.8,r'$k= $'+f'{round(H_U_ks[0],2)}',fontsize=15)


# plot anisotropic cortex V
plt.figure(figsize=(18,9))
plt.subplot(1,2,1)
plt.plot(Omega*180/np.pi,DIV_V_A1,'r*')
div,curl,Def=get_best_gradients(V_A_ks[0])
plt.plot(OmegaA*180/np.pi,div,'r--')
plt.plot(Omega*180/np.pi,CURL_V_A1,'g*')
plt.plot(OmegaA*180/np.pi,curl,'g--')
plt.plot(Omega*180/np.pi,DEF_V_A1,'b*')
plt.plot(OmegaA*180/np.pi,Def,'--b')
plt.xticks([0,180],fontsize=15)
plt.yticks([-1,1],fontsize=15)
plt.xlabel('Motion Phase (Deg)',fontsize=30)
plt.ylabel('Gradients',fontsize=30)
plt.title('Vertical',fontsize=40)
plt.text(15,-0.8,r'$k= $'+f'{round(V_A_ks[0],2)}',fontsize=15)
# plot anisotropic cortex H
plt.subplot(1,2,2)
plt.plot(Omega*180/np.pi,DIV_H_A1,'r*')
div,curl,Def=get_best_gradients(H_A_ks[0])
plt.plot(OmegaA*180/np.pi,div,'r--')
plt.plot(Omega*180/np.pi,CURL_H_A1,'g*')
plt.plot(OmegaA*180/np.pi,curl,'g--')
plt.plot(Omega*180/np.pi,DEF_H_A1,'b*')
plt.plot(OmegaA*180/np.pi,Def,'--b')
plt.xticks([0,180],fontsize=15)
plt.yticks([-1,1],fontsize=15)
plt.xlabel('Motion Phase (Deg)',fontsize=30)
plt.ylabel('Gradients',fontsize=30)
plt.title('Horizontal',fontsize=40)
plt.suptitle('Anisotropic tuning',fontsize=40)
plt.text(15,-0.8,r'$k= $'+f'{round(H_A_ks[0],2)}',fontsize=15)

#%% Model fit
import numpy as np
from scipy.optimize import curve_fit
def polynomial_func(x, *coefficients):
    return sum(coef * x**i for i, coef in enumerate(coefficients))
degree=3
# Initial guess for the polynomial coefficients
initial_guess = np.ones(degree + 1)

# Fit the polynomial function to the data


plt.figure()
plt.plot(Stretches,H_U_ks,'or')
poptH_U, _ = curve_fit(polynomial_func, Stretches, H_U_ks, p0=initial_guess)
x_values = np.linspace(1, 1.5, 100)
predicted_values = polynomial_func(x_values, *poptH_U)
plt.plot(x_values,predicted_values,'r-',label='H-rotation')

plt.plot(Stretches,V_U_ks,'*b')
poptV_U, _ = curve_fit(polynomial_func, Stretches, V_U_ks, p0=initial_guess)
x_values = np.linspace(1, 1.5, 100)
predicted_values = polynomial_func(x_values, *poptV_U)
plt.plot(x_values,predicted_values,'b-',label='V-rotation')

plt.yticks([0,0.5,1],fontsize=20)
plt.xticks([1,1.25,1.5],fontsize=20)
plt.ylabel(r'Best $k$',fontsize=30)
plt.xlabel('Amount of H-stretch',fontsize=30)
plt.title('Isotropic Cortex',fontsize=35)
plt.legend(fontsize=20)

plt.figure()
plt.plot(Stretches,H_A_ks,'or')
poptH_A, _ = curve_fit(polynomial_func, Stretches, H_A_ks, p0=initial_guess)
x_values = np.linspace(1, 1.5, 100)
predicted_values = polynomial_func(x_values, *poptH_A)
plt.plot(x_values,predicted_values,'r-',label='H-rotation')

plt.plot(Stretches,V_A_ks,'*b')
poptV_A, _ = curve_fit(polynomial_func, Stretches, V_A_ks, p0=initial_guess)
x_values = np.linspace(1, 1.5, 100)
predicted_values = polynomial_func(x_values, *poptV_A)
plt.plot(x_values,predicted_values,'b-',label='V-rotation')

plt.yticks([0,0.5,1],fontsize=20)
plt.xticks([1,1.25,1.5],fontsize=20)
plt.ylabel(r'Best $k$',fontsize=30)
plt.xlabel('Amount of H-stretch',fontsize=30)
plt.title('Anisotropic Cortex',fontsize=35)
plt.legend(fontsize=20)


