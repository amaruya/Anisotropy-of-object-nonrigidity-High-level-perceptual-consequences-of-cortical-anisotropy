#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the indivusal result of the experiments here for figure S1-2
"""

import pandas as pd 
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
current_folder = os.path.dirname(os.path.abspath(__file__))

path=current_folder+'/data/'
max_path=current_folder+'/data/'
Names=['Aki','Sabina','Larisa','Ashwin']
interval=[[(0,59)],[(0,27),(27,39)],[(0,45)],[(0,18),(18,39)]]
data_pd=pd.DataFrame()
Max_res=[]
for nn in range(len(Names)):
    Name=Names[nn]
    Max_res.append(np.load(max_path+Name+'_data.npy'))
    for kk in range(len(interval[nn])):
        data_path=path+Name+f'/{Name}{interval[nn][kk][0]}_to_{interval[nn][kk][1]}_data.pkl'
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        stretch=np.linspace(1,1.5,12)
        Stretch_im_ad=[]
        Stretch_phys_ad=[]
        for ii in range(0,len(data['Stretch image adjust']),3):
            Stretch_im_ad.append(stretch[data['Stretch image adjust'][ii:ii+3].astype(int)])
            Stretch_phys_ad.append(stretch[data['Stretch phys adjust'][ii:ii+3].astype(int)])
        Stretch_im_ad=np.vstack(Stretch_im_ad)
        Stretch_phys_ad=np.vstack(Stretch_phys_ad)
        # make a dataframe
        
        data['Stretch image adjust']=np.mean(Stretch_im_ad,1)
        data['Stretch phys adjust']=np.mean(Stretch_phys_ad,1)
        data['Pattern']=data['Pattern'][:len(data['Stretch image adjust'])]
        data['Test']=data['Test'][:len(data['Stretch image adjust'])]
        pre_data=pd.DataFrame(data)
        if len(pre_data)>40:
            pre_data=pre_data.iloc[:40]
        data_pd_=pre_data
        data_pd_['ID']=nn
        data_pd=pd.concat([data_pd,data_pd_])


#%% Organize the data in accordance with whether test image is horizontal or vertical 

data_pd_all=data_pd

binwidth=.01
for oo in range(4):

    data_pd=data_pd_all[data_pd_all['ID']==oo]
    plt.rcParams.update({'font.size': 25})
    Test=data_pd['Test']
    # Compute average stretch for each condition
    # When test is Horizontal rotation and stretch the vertically rotating ring horizontally 
    # Image stretch
    H_Each_mean_im=data_pd['Stretch image adjust'][Test==0].values
    H_mean_im=np.mean(H_Each_mean_im)
    # Physical stretch
    H_Each_mean_phys=data_pd['Stretch phys adjust'][Test==0].values
    H_mean_phys=np.mean(H_Each_mean_phys)
    
    
    plt.figure(figsize=(24, 12))
    plt.subplot(1,2,1)
    # Plotting histograms with striped patterns
    plt.hist(H_Each_mean_im, color='r', bins=np.arange(min(H_Each_mean_im), max(H_Each_mean_im) + binwidth, binwidth), hatch='/', alpha=0.5)
    plt.hist(H_Each_mean_phys, color='b', bins=np.arange(min(H_Each_mean_phys), max(H_Each_mean_phys) + binwidth, binwidth), hatch='\\', alpha=0.5)
    
    # Plotting vertical lines for mean values
    plt.axvline(H_mean_im, color='r', linestyle='--')
    plt.axvline(H_mean_phys, color='b', linestyle='--')
    
    plt.xlim(1, 1.6)
    plt.xlabel('H-axis stretch',fontsize=30)
    plt.ylabel('#',fontsize=30)
    plt.title('Matching horizontally rotating shapes',fontsize=35)
    plt.legend(['Mean (image)', 'Mean (physical)', 'Image stretch', 'Physical stretch'])
    #plt.suptitle(f'Oberver {oo}',fontsize=35)
    # When test is Vertical rotation and stretch the horizontally rotating ring horizontally 
    # Image stretch
    V_Each_mean_im = data_pd['Stretch image adjust'][Test==1].values
    V_mean_im = np.mean(V_Each_mean_im)
    
    # Physical stretch
    V_Each_mean_phys = data_pd['Stretch phys adjust'][Test==1].values
    V_mean_phys = np.mean(V_Each_mean_phys)
    
    plt.subplot(1,2,2)
    # Plotting histograms with striped patterns
    plt.hist(V_Each_mean_im, color='r', bins=np.arange(min(V_Each_mean_im), max(V_Each_mean_im) + binwidth, binwidth), hatch='/', alpha=0.5)
    plt.hist(V_Each_mean_phys, color='b', bins=np.arange(min(V_Each_mean_phys), max(V_Each_mean_phys) + binwidth, binwidth), hatch='\\', alpha=0.5)
    
    # Plotting vertical lines for mean values
    plt.axvline(V_mean_im, color='r', linestyle='--')
    plt.axvline(V_mean_phys, color='b', linestyle='--')
    
    plt.xlim(1, 1.6)
    plt.xlabel('H-axis stretch',fontsize=30)
    plt.ylabel('#',fontsize=30)
    plt.legend(['Mean (image)', 'Mean (physical)', 'Image stretch', 'Physical stretch'])
    plt.title('Matching vertically rotating shapes',fontsize=35)
    plt.suptitle(f'Oberver {oo+1}',fontsize=35)
    plt.show()
    
    
    
    #%% plot rigid non-rigid judgement for shape matched condition
    H_VS_V=data_pd['H VS V']
 
    #%% Conduct boot-strapping method to see the statistical significance
    from scipy.stats import ttest_ind
    from scipy.stats import ttest_1samp
    Original=H_VS_V.values
    ISH=1-data_pd[Test==1]['Non-rigidity image'].values
    PSH=1-data_pd[Test==1]['Non-rigidity phys'].values
    ISV=data_pd['Non-rigidity image'][Test==0].values
    PSV=data_pd['Non-rigidity phys'][Test==0].values
    # Conduct bootstrapping
    # Number of bootstrap samples
    num_samples = 1000
    
    # Function to calculate the statistic of interest (e.g., mean, median)
    def statistic(data):
        return np.mean(data)
    
    # Bootstrap function
    def bootstrap(data, num_samples, statistic):
        bootstrap_samples = []
        n = 1000
        for _ in range(num_samples):
            # Resample with replacement
            bootstrap_sample = np.random.choice(data, size=n, replace=True)
            # Calculate the statistic of interest for the bootstrap sample
            bootstrap_statistic = statistic(bootstrap_sample)
            bootstrap_samples.append(bootstrap_statistic)
        return bootstrap_samples
    
    Original_b=bootstrap(Original, num_samples=1000, statistic=statistic)
    ISH_b=bootstrap(ISH, num_samples=1000, statistic=statistic)
    PSH_b=bootstrap(PSH, num_samples=1000, statistic=statistic)
    ISV_b=bootstrap(ISV, num_samples=1000, statistic=statistic)
    PSV_b=bootstrap(PSV, num_samples=1000, statistic=statistic)
    Max_res=np.hstack(Max_res)
    Max_b=bootstrap(Max_res, num_samples=1000, statistic=statistic)
    # Perform one-sample t-tests
    t_stat_Original, p_val_Original = ttest_1samp(Original_b, 0.5)
    t_stat_ISH, p_val_ISH = ttest_1samp(ISH_b, 0.5)
    t_stat_PSH, p_val_PSH = ttest_1samp(PSH_b, 0.5)
    t_stat_ISV, p_val_ISV = ttest_1samp(ISV_b, 0.5)
    t_stat_PSV, p_val_PSV = ttest_1samp(PSV_b, 0.5)
    t_stat_Max, p_val_Max = ttest_1samp(PSV_b, 0.5)
    # Calculate mean and variance for each sample
    means = [np.mean(arr) for arr in [Original_b,  ISH_b,PSH_b ,ISV_b,PSV_b, Max_b]]
    variances = [np.var(arr) for arr in [Original_b,  ISH_b,PSH_b ,ISV_b,PSV_b, Max_b]]
    std_errors = [np.std(arr, ddof=1) / np.sqrt(num_samples) for arr in [Original_b, ISH_b, PSH_b ,ISV_b,PSV_b, Max_b]]

    # Combine results for plotting
    labels = ['Original', 'H', 'H', 'V', 'V','Max']
    
    p_values = [p_val_Original,p_val_ISH,p_val_PSH, p_val_ISV,p_val_PSV,p_val_Max]
    # Plotting
    x = np.arange(len(labels))
    width =0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 12))
    
    # Plot bars for means
    color=['k','r','b','r','b','r']
    
    for ii in range(len(means)):
        ax1.bar(x[ii] , means[ii], width, color=color[ii] )
    ax1.set_ylabel('Proportion of V-rotation more non-rigid', fontsize=35)
    
    # Plot error bars for variances
    for ii in range(len(x)):
        ax1.errorbar(x[ii], means[ii], std_errors[ii],color='k', capsize=5)
    
    # Plot significance markers
    for i, p_val in enumerate(p_values):
        if p_val < 0.05:
            ax1.text(x[i], means[i] + 0.01, '*', ha='center', va='bottom', color='k', fontsize=20)
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=35)
    ax1.set_yticks([0,0.5,1])
    ax1.set_title(f'Observer {oo+1}', fontsize=35)
    plt.show()
    
    
    
    
    
    
    
    
