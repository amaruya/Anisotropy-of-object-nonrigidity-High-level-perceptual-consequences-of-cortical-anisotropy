#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 13:49:50 2024

@author: akihitomaruya

The code will generate videos that are used in the experiment: two rigidly connected rings that rotate either horizontally or vertically 
with different amount of stretch. 

"""

import matplotlib.colors
import matplotlib.cm as cm
import numpy as np
import matplotlib.pyplot as plt
import numpy.matlib
import moviepy.video.io.ImageSequenceClip
from scipy.ndimage.filters import gaussian_filter
import os
# Get the current working directory
current_folder = os.path.dirname(os.path.abspath(__file__))
#%% Int

fps=60
distance=100; # distance in cm 
fc=distance# focal lenght of camera
dc=distance #distamce from stick to focal point of camera
im_sz=378
uu=np.linspace(-1.5, 1.5,im_sz)
vv=np.linspace(-1.5,1.5,im_sz)
duration=190
im_rots=np.array([0,90])
A=np.linspace(1,1.5,12)
lw=1
types=['Image stretch','Physical stretch']

for rr in range(len(im_rots)):
    
    for aa in range(len(A)):
        for st in range(len(types)):
            im_rot=im_rots[rr]
            st_type=types[st]
            if (st_type=='Image stretch') & (im_rot==0):
                a1=1 #width
                b1=1 #height
                im_st_w1=A[aa]
                im_st_h1=1
            elif (st_type=='Image stretch') & (im_rot==90):
                a1=1 #width
                b1=1 #height
                im_st_w1=1
                im_st_h1=A[aa]
            elif (st_type=='Physical stretch') & (im_rot==0):
                a1=A[aa] #width
                b1=1 #height
                im_st_w1=1
                im_st_h1=1
            elif (st_type=='Physical stretch') & (im_rot==90):
                a1=1 #width
                b1=A[aa] #height
                im_st_w1=1
                im_st_h1=1
                
            
            im_st_w=im_st_w1/np.sqrt(im_st_h1*im_st_w1)
            im_st_h=im_st_h1/np.sqrt(im_st_h1*im_st_w1)
            a=a1/np.sqrt(a1*b1)
            b=b1/np.sqrt(a1*b1)
        
            sfile = os.path.join(current_folder, 'Images')
            vfile = os.path.join(current_folder, 'Videos')
            name=f'{aa}_Ellipse_h_{a}_w_{b}_im_stretch_h_{im_st_h}_w_{im_st_w}_rot_{im_rot}_num_im_{duration}'
            
            try:
                os.mkdir(sfile+'/'+name)
            except:
                print('File exists')
            
            try:
                os.mkdir(vfile+'/'+'Stretch')
            except:
                print('Vfile exists')
            
            im_rot=im_rot*np.pi/180
            phi=30*np.pi/180
            theta=np.arange(0,np.pi*2,.01)
            Omega=np.linspace(0,2*np.pi,duration,endpoint=False)
            for tt in range(len(Omega)): 
                im=np.ones((im_sz,im_sz))*.5
                OmegaT=Omega[tt]
                
                
                
                X=a*np.cos(OmegaT)*np.cos(theta)-b*np.sin(OmegaT)*np.sin(theta)*np.cos(phi)
                Y=b*np.sin(theta)*np.sin(phi)-np.sin(phi)*b
                Z=-a*np.sin(OmegaT)*np.cos(theta)-b*np.cos(OmegaT)*np.sin(theta)*np.cos(phi)
                
                u1=X*fc/(dc-Z)
                v1=Y*fc/(dc-Z)
                
                u1r=u1*np.cos(im_rot)-v1*np.sin(im_rot)
                v1r=u1*np.sin(im_rot)+v1*np.cos(im_rot)
                u1r=im_st_w*u1r
                v1r=im_st_h*v1r
                X=a*np.cos(OmegaT)*np.cos(theta)-b*np.sin(OmegaT)*np.sin(theta)*np.cos(-phi)
                Y=b*np.sin(theta)*np.sin(-phi)+np.sin(phi)*b
                Z=-a*np.sin(OmegaT)*np.cos(theta)-b*np.cos(OmegaT)*np.sin(theta)*np.cos(-phi)
                
                u2=X*fc/(dc-Z)
                v2=Y*fc/(dc-Z)
                
                u2r=u2*np.cos(im_rot)-v2*np.sin(im_rot)
                v2r=u2*np.sin(im_rot)+v2*np.cos(im_rot)
                u2r=im_st_w*u2r
                v2r=im_st_h*v2r
                
                
                #%%
                ind_u1=np.array([np.argmin((u1r[ii]-uu)**2) for ii in range(len(u1r))])
                ind_v1=np.array([np.argmin((v1r[ii]-vv)**2) for ii in range(len(u1r))])
                
                ind_u2=np.array([np.argmin((u2r[ii]-uu)**2) for ii in range(len(u2r))])
                ind_v2=np.array([np.argmin((v2r[ii]-vv)**2) for ii in range(len(u2r))])
                
                im[ind_v1,ind_u1]=1
                im[ind_v2,ind_u2]=1
                
                
                
                im=gaussian_filter(im,sigma=lw)
                im[im>.5]=1
                im=gaussian_filter(im,sigma=2)
                im[im<=.5]=.5
                
                px = 1/plt.rcParams['figure.dpi']
                fig=plt.figure(figsize=(im_sz*px, im_sz*px),frameon=False)
                ax = plt.Axes(fig, [0., 0., 1., 1.])
                ax.set_axis_off()
                fig.add_axes(ax)
                plt.imshow(im,cmap='gray',vmax=1,vmin=0)
                plt.axis('off')
                if tt<10:
                    sname=sfile+'/'+name+'/'+f'im_0000{tt}.jpg'
                elif tt>99 and tt<1000:
                    sname=sfile+'/'+name+'/'+f'im_00{tt}.jpg'
                elif tt>999:
                    sname=sfile+'/'+name+'/'+f'im_0{tt}.jpg'
                elif tt>9999:
                    sname=sfile+'/'+name+'/'+f'im_{tt}.jpg'
                else:
                    sname=sfile+'/'+name+'/'+f'im_000{tt}.jpg'
                fig.savefig(sname,facecolor=fig.get_facecolor())
                plt.close('all')    
            
            
            video_name=vfile+'/'+'Stretch'+'/'+name+'.mp4'
            fps=fps
            
            image_files2 = [sfile+'/'+name+'/'+img for img in sorted(os.listdir(sfile+'/'+name)) if img.endswith(".jpg")]
            clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files2, fps=fps)
            clip.write_videofile(video_name) 
            
            
            
            
            
            
            
            
