#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 3. Cortical anisotropy and shape anisotropy


"""

from matplotlib.pyplot import cm
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import vonmises
import os 
current_folder = os.path.dirname(os.path.abspath(__file__))
def angle_estimate(Ns,pfo,K1s,thetac,thetas,k=0.125):
    vec1=np.zeros(2)
    vec2=np.zeros(2)
    for ii in range(len(pfo)): 
        n=Ns[ii].numpy()
        # Central
       
        E_gc=n*get_response(pfo[ii],thetac,K1s[ii],thetas)
        mu=(pfo[ii].numpy())
        
        u=np.array([np.cos(mu),np.sin(mu)])
        vec1+=u*E_gc*n
        
        # Surround
       
        E_gcs= n*get_response(pfo[ii],thetas,K1s[ii],thetac)
        mu=(pfo[ii])
        u=np.array([np.cos(mu),np.sin(mu)])
        vec2+=u*E_gcs
    return np.arctan2(vec1[1],vec1[0])*180/np.pi,np.arctan2(vec2[1],vec2[0])*180/np.pi,np.abs(np.arctan2(vec1[1],vec1[0])*180/np.pi-np.arctan2(vec2[1],vec2[0])*180/np.pi)

def get_response(pfo,thetac,K1s,thetas,k=0.125):
    lc=vonmises.pdf(pfo,K1s,thetac)
    ls=vonmises.pdf(pfo,K1s,thetas)
    l=np.sqrt(lc**2+ls**2+k)
    return lc/l

    
# Load the tuning width (Kappa) and the number of cells 
file_path=current_folder+'/'+'data/'
Ns=(torch.load(file_path+'d.pt')[:8].reshape(-1)+torch.load(file_path+'d.pt')[8:16].reshape(-1))/2
K1s=(torch.load(file_path+'K1s.pt')[:8].reshape(-1)+torch.load(file_path+'K1s.pt')[8:16].reshape(-1))/2


#%% Figure  3A
#Anisotropic
Ns=torch.hstack((Ns,Ns))
K1s=torch.hstack((K1s,K1s))
# # Make orientation tuning curve for 16 orientations
# preferred orientation
pfo=torch.arange(-np.pi,np.pi,np.pi/8)
color = cm.rainbow(np.linspace(0, 1, len(pfo)))
x=torch.arange(-np.pi,torch.pi,.0001)
plt.figure()
plt.rcParams['font.size'] = 20
for ii in range(len(pfo)):
    tuning=vonmises.pdf(pfo[ii], K1s[ii], x)
    plt.plot(x*180/torch.pi,tuning*Ns[ii].numpy(),color=color[ii])
plt.title('Anisotropic V1')
plt.ylabel(r'$n_if(\theta|\mu_i,k_i)$')
plt.xlabel('Stimulus orientation (degrees)')
plt.xticks(np.arange(-180,181,90))

#% Isotropic


color = cm.rainbow(np.linspace(0, 1, len(pfo)))
x=torch.arange(-np.pi,torch.pi,.0001)
plt.figure()
plt.rcParams['font.size'] = 20
for ii in range(len(pfo)):
    tuning=vonmises.pdf(pfo[ii], K1s.mean(), x)
    plt.plot(x*180/torch.pi,tuning*Ns.mean().numpy(),color=color[ii])
plt.title('Isotropic V1')
plt.ylabel(r'$\bar{n}f(\theta|\mu_i,\bar{k})$')
plt.xlabel('Stimulus orientation (degrees)')
plt.xticks(np.arange(-180,181,90))
#%% Figure 3B Left
# Plot the number of cells
plt.figure()
plt.plot(np.arange(-180,180,22.5),Ns,'-ro')
plt.ylabel('# of cells',fontsize=20)
plt.xlabel('Preferred Orientation (degs)',fontsize=20)
plt.xticks(np.arange(-180,182,90))
plt.ylim([200,500])
# Plot the tuning widths
plt.plot(np.arange(-180,180,22.5),Ns.mean()*torch.ones_like(Ns),'-ko')
plt.ylabel('# of cells',fontsize=20)
plt.xlabel('Preferred Orientation (degs)',fontsize=20)
plt.xticks(np.arange(-180,182,90))
plt.ylim([200,500])
tuning_widths=np.array([28.138,
29.793,
34.759,
38.069,
35.586,
40.552,
33.931,
30.207,
28.552,
32.276,
34.759,
38.483,
35.586,
37.655,
33.931,
32.276,
])

t_width_ave=(tuning_widths[:8]+tuning_widths[8:])/2
t_width_ave=np.hstack((t_width_ave,t_width_ave))
plt.figure()
plt.rcParams['font.size'] = 20
plt.plot(np.arange(-180,180,22.5),t_width_ave,'-ro')
plt.ylabel('Tuning width',fontsize=20)
plt.xlabel('Preferred Orientation (degs)',fontsize=20)
plt.xticks(np.arange(-180,182,90))
plt.ylim([20,40])
plt.plot(np.arange(-180,180,22.5),t_width_ave.mean()*torch.ones_like(Ns),'-ko')
plt.ylabel('Tuning width',fontsize=20)
plt.xlabel('Preferred Orientation (degs)',fontsize=20)
plt.xticks(np.arange(-180,182,90))
plt.ylim([20,40])


#%% Figure 3C
Egs1=[]
Egs2=[]

k=0.125
Ns=Ns#/Ns.max()*2
#Ns=torch.ones_like(Ns)
#K1s=torch.ones_like(K1s)
theta=np.arange(5,90-6,1)#*torch.pi/180
phys_angV=[]
phys_angH=[]
AngH=[]
AngV=[]
Theta_pH=[]
Theta_qH=[]
Theta_pV=[]
Theta_qV=[]

for ii in range(len(theta)):
    c=theta[ii]
    thetacV=90-c
    thetasV=-thetacV
    Diff_v=np.abs(thetacV-thetasV)
    thetacH=c
    thetasH=180-c
    Diff_h=np.abs(thetacH-thetasH)
    theta_pH,theta_qH,angV=angle_estimate(Ns, pfo, K1s, thetacH*np.pi/180, thetasH*np.pi/180,k)
    theta_pV,theta_qV,angH=angle_estimate(Ns, pfo, K1s, thetacV*np.pi/180, thetasV*np.pi/180,k)
    Theta_pH.append(theta_pH)
    Theta_qH.append(theta_qH)
    Theta_pV.append(theta_pV)
    Theta_qV.append(theta_qV)
    phys_angV.append(Diff_v)
    phys_angH.append(Diff_h)
    AngH.append(angH)
    AngV.append(angV)
AngH=np.hstack(AngH)#*180/torch.pi
AngV=np.hstack(AngV)#*180/torch.pi

AngH_A=AngH
AngV_A=AngV


plt.figure()
plt.plot(np.hstack(phys_angH),AngH-AngV,'r-')
#plt.plot(np.hstack(phys_angV),AngV,'r-')
plt.xlabel('Physical angle (deg)', fontsize=20)
plt.ylabel(r'$\gamma_H-\gamma_V$', fontsize=20)
#plt.axhline(0,linestyle='--',color='k')
plt.xticks(np.arange(0,182,45))


#% Uniform
Egs1=[]
Egs2=[]

k=0.125
Ns=Ns#/Ns.max()*2
#Ns=torch.ones_like(Ns)
#K1s=torch.ones_like(K1s)
theta=np.arange(5,90-6,1)#*torch.pi/180
phys_angV=[]
phys_angH=[]
AngH=[]
AngV=[]
Theta_pH=[]
Theta_qH=[]
Theta_pV=[]
Theta_qV=[]

for ii in range(len(theta)):
    c=theta[ii]
    thetacV=90-c
    thetasV=-thetacV
    Diff_v=np.abs(thetacV-thetasV)
    thetacH=c
    thetasH=180-c
    Diff_h=np.abs(thetacH-thetasH)
    theta_pH,theta_qH,angV=angle_estimate(torch.ones_like(Ns), pfo, torch.ones_like(K1s), thetacH*np.pi/180, thetasH*np.pi/180,k)
    theta_pV,theta_qV,angH=angle_estimate(torch.ones_like(Ns), pfo, torch.ones_like(K1s), thetacV*np.pi/180, thetasV*np.pi/180,k)
    Theta_pH.append(theta_pH)
    Theta_qH.append(theta_qH)
    Theta_pV.append(theta_pV)
    Theta_qV.append(theta_qV)
    phys_angV.append(Diff_v)
    phys_angH.append(Diff_h)
    AngH.append(angH)
    AngV.append(angV)
AngH_U=np.hstack(AngH)#*180/torch.pi
AngV_U=np.hstack(AngV)#*180/torch.pi


plt.plot(np.hstack(phys_angH),AngH_U-AngV_U,'k-')
#plt.plot(np.hstack(phys_angV),AngV,'r-')
plt.xlabel('Physical angle (deg)', fontsize=20)
plt.ylabel(r'$\hat{\gamma_H}-\hat{\gamma_V}$', fontsize=20)
#plt.axhline(0,linestyle='--',color='k')
plt.xticks(np.arange(0,182,45))
plt.legend(['Anisotropic', 'Isotropic'], loc='best', bbox_to_anchor=(.7, .7),fontsize=8.5)
plt.tight_layout()


#%% Figure 3B right
# Plot physical VS estimated
# plot Rh_hat vs physical 
plt.figure(figsize=(5,5))

plt.plot(np.hstack(phys_angH),AngH_A,'r-')
plt.plot(np.hstack(phys_angH),AngH_U,'k-')
plt.xlabel('Physical angle (deg)', fontsize=20)
plt.ylabel(r'$\hat{\gamma_H}$', fontsize=20)
#plt.axhline(0,linestyle='--',color='k')
plt.xticks(np.arange(0,180+45,45))
plt.yticks(np.arange(0,180+45,45))

plt.legend(['Anisotropic', 'Isotropic'], loc='best', bbox_to_anchor=(.4, .8),fontsize=8.5)
plt.tight_layout()
plt.axis('square')
plt.figure(figsize=(5,5))
plt.plot(np.hstack(phys_angV),AngV_A,'r-')
plt.plot(np.hstack(phys_angV),AngV_U,'k-')
plt.xlabel('Physical angle (deg)', fontsize=20)
plt.ylabel(r'$\hat{\gamma_V}$', fontsize=20)
#plt.axhline(0,linestyle='--',color='k')
plt.xticks(np.arange(0,180+45,45))
plt.yticks(np.arange(0,180+45,45))

plt.legend(['Anisotropic', 'Isotropic'], loc='best', bbox_to_anchor=(.4, .8),fontsize=8.5)
plt.tight_layout()
plt.axis('square')


