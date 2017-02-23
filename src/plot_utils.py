#!/usr/bin/python

# CerBERUS - Cerebellum-Basal ganglia-CortEx Research Unified System.
# Copyright (C) 2016 Francesco Mannella <francesco.mannella@gmail.com> 
# and Daniele Caligiore <daniele.caligiore@gmail.com>
#
# This file is part of CerBERUS.
#
# CerBERUS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CerBERUS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CerBERUS.  If not, see <http://www.gnu.org/licenses/>.

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_means(sim) :
   
    areas = sim.SELECTED_META_AREAS
    meansdf = sim.meandf 
    des_means = sim.targets.iloc[:,2:]
    des_means = des_means.as_matrix() 
    

    fig = plt.figure(figsize=(16, 10))

    if np.any(meansdf.TIC=='NOTIC') :
        for i,area in zip(xrange(len(areas)), areas):

            select = (meansdf.META_AREA==area) & (meansdf.TIC=='NOTIC')
            ax = fig.add_subplot(5,2,(i*2 +1))
            y = meansdf[select].iloc[:,5:-1].as_matrix().T
            ax.plot(y, lw=2)   
            x = np.linspace(0,sim.INNER_WINDOW-1, sim.MEASURE_INTERVALS )
            ax.plot(x, des_means[i*2,:], c="red")

    if np.any(meansdf.TIC=='TIC') :
        for i,area in zip(xrange(len(areas)), areas):

            select = (meansdf.META_AREA==area) & (meansdf.TIC=='TIC')
            ax = fig.add_subplot(5,2,(i*2+2))
            y = meansdf[select].iloc[:,5:-1].as_matrix().T
            ax.plot(y, lw=2)   
            x = np.linspace(0,sim.INNER_WINDOW-1, sim.MEASURE_INTERVALS )
            ax.plot(x, des_means[((i*2)+1),:], c="red")

     

# we define a general plot function for a layer 
def plot_layer_gs(data,row,xaxis=True, label="layer",N=3) :
    
    gs = gridspec.GridSpec(rows, 20 )
    
    # paramenters 
    colors = ['r','g','b']    # color types
    xlims =[250,stime+50]    # limits on the x axis
    ylims = [-0.1,4.1]    # limits on the y axis   
    linewidth = 2    # width of the plot line
    plotalpha = 0.5    # alpha color of the line 
    fillalpha = 0.01    # alpha color of the filled area  
        

    plt.subplot(gs[row,:2])    # plot within the subplot window
    plt.axis("off")
    plt.text(.1,.2,label,fontsize=11)

    plt.subplot(gs[row,2:])    # plot within the subplot window

    # for each channel
    x = arange(stime)
    cols=iter(cm.rainbow(np.linspace(0,1,N)))
    for i in xrange(N) :
        color = next(cols)
        y = data[i,:]    
        # plot the line
        plt.plot(y, color=color, linewidth=linewidth, alpha=plotalpha)
        # fill the area below the line
        plt.fill_between(x, y, 0, facecolor=color, alpha=fillalpha)

    plt.xlim(xlims)
    plt.ylabel(label)
    plt.yticks([0,1])
    if not xaxis :
        plt.xticks([])


def plot_raw_layers(sim) : 
    
    # open the plot window (sizes in figsize are in inches)
    f = plt.figure("BASAL GANGLIA-CORTEX SIMULATION EFP",figsize=([10,6]),facecolor='white') 
     

    rows = 6    # we have 6 rows in our figure 
    row = 0    # start the row counter
    for area in np.hstack(("DA", sim.SELECTED_META_AREAS )) :

        data = sim.raw_df[sim.raw_df.AREA==area].iloc[:,1:].as_matrix().ravel()
        plot_row_gs(data, row, xaxis=False, label=area)
        row +=1

def plot_ctx_channels(sim) : 
    
    # open the plot window (sizes in figsize are in inches)
    f = plt.figure("CHANNELS",figsize=([10,6]),facecolor='white') 
     

    rows = 10     
    cols = (sim.TRIALS-1)/10 +1    
    stime = sim.INNER_WINDOW
    gs = gridspec.GridSpec(rows, cols)
    
    xlims =[-50,stime+50]     
    ylims = [-0.5,4.5] 
    linewidth = 2.0    # width of the plot line
    plotalpha = 0.8    # alpha color of the line 
    fillalpha = 0.1    # alpha color of the filled area       
      
    for win in range(sim.TRIALS-1) :
        win_data = sim.df[(sim.df.AREA=="CTX") & (sim.df.WIN==win) ].iloc[:,5:-1].as_matrix()
        
        ax = f.add_subplot(gs[int(win%10),int(win/10)])    
        ax.plot(win_data.T, linewidth=linewidth, alpha=plotalpha)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_axis_off()
    
# we define a general plot function for a single row     
def plot_row_gs(data, row, xaxis=True, label="layer", color="black") :

    stime = data.shape[0]

    gs = gridspec.GridSpec(6, 20 )
    
    xlims =[250,stime+50] # limits on the x axis 
    
    mmax = 1 
    mmin =  .5
    ylims = [-0.5,4.5] # limits on the y axis   
    
    linewidth = 2.0    # width of the plot line
    plotalpha = 0.8    # alpha color of the line 
    fillalpha = 0.1    # alpha color of the filled area       
    
    plt.subplot(gs[row,:2])    # plot within the subplot window
    plt.axis("off")
    plt.text(.1,.2,label,fontsize=11)

    plt.subplot(gs[row,2:])    # plot within the subplot window
    
    y = data    
    x = np.arange(stime)
    
    # plot the line
    plt.plot(y, color=color, linewidth=linewidth, alpha=plotalpha)
    
    plt.axis("off")
    plt.xlim(xlims)
    plt.ylim(ylims)
    plt.ylabel(label)
    plt.yticks([])
    plt.xticks([])
