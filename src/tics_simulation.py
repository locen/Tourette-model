#!/usr/bin/env python

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

import os
import numpy as np
from numpy.random import RandomState as RNG
import pandas as pd
from bg import BG
from bg import Layer
from cerebellum import Cerebellum
from onset import OnsetLayer
from parameter_manager import ParameterManager as PM

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
    
def get_timewindows(timeseries, timewindow_number, diameter=201, dispace=0) :
    '''
    Build a dataframe of timewindows (rows)  from a timeseries
    
    
    @param      timeseries          The sequence of timesteps of the simulation
    @param      timewindow_number   The number of segments within the timeseries
    @param      diameter            The length in timesteps of each timewindow at 
                                    the center of a segment
    
    return
                res                 A 'timewindow_number'X'diameter' matrix.
                                    each row is a time window
    '''
    
    radius = diameter/2

    orig_time_series = np.squeeze(timeseries.copy())
    stime = len(orig_time_series)
    timewindows_store = np.array([])
    timewindow_size = stime/timewindow_number

    for curr_timewindow in xrange(1,timewindow_number) :
        start = curr_timewindow*timewindow_size-radius + dispace
        end = curr_timewindow*timewindow_size+radius+1 + dispace
        win = orig_time_series[start:end]
        timewindows_store = np.hstack([ timewindows_store, win ])
    res = timewindows_store.reshape(timewindow_number-1,diameter)

    return res 

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------

def get_maxamp(timeseries, timewindow_number, diameter=201, displace = 0 ) :
    
    '''
    Get the maximum amplitudes in the timewindows of a timeseries
    
    
    @param      timeseries          The sequence of timesteps of the simulation
    @param      timewindow_number   The number of segments within the timeseries
    @param      diameter            The length in timesteps of each timewindow at 
                                    the center of a segment
    return
                res                 A 'timewindow_number'x2 matrix.
                                    The first column is the position of the timewindow,
                                    the second column id the corresponding amplitude.
                                    Rows are reverse-ordered based on the amplitude 
    '''  


    timewindows_store = get_timewindows(timeseries,timewindow_number,diameter, displace)

    res = timewindows_store.max(1) - timewindows_store.min(1)
    res = np.vstack([range(timewindow_number-1),res])
    res = res[:,np.argsort(res[1,:])].T

    return res

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------

def get_tic_timewindows(time_series, timewindow_number, area_idx, channel_idx, 
        tic_group_idx, notic_group_idx, seed, diameter=201 ) : 
    '''
    Build a dataframe of timewindows (rows)  from a timeseries
    
    
    @param      timeseries          The sequence of timesteps of the simulation
    @param      timewindow_number   The number of segments within the timeseries
    @param      diameter            The length in timesteps of each timewindow at 
                                    the center of a segment
    @param      rea_idx             index of the area [0-8] described by the timeseries
    @param      channel_idx         index of the channel (0,1,2)
    @param      tic_group_idx       indices of the tic timewindows
    @param      notic_group_idx     indices of the notic timewindows
    @param      seed                seed of simulation 
    
    return
                res                 A 'timewindow_number'X(SEED+AREA+TIC+WIN+CHANNEL+TIMESTEPS) matrix.
                                    each row has a SEED_INDEX, AREA_INDEX, TIC_VALUE, WINDOW_INDEX, 
                                    CHANNEL_INDEX, TIMEWINDOW_ROW
    #
    '''
   
    stime = len(time_series)
    
    timewindows_store = get_timewindows(time_series, timewindow_number );

    TIC_FACTOR = np.arange(timewindow_number-1)
    TIC_FACTOR[tic_group_idx] = 1
    TIC_FACTOR[notic_group_idx] = 0
    TIC_FACTOR = TIC_FACTOR.reshape(timewindow_number-1,1)
    WIN_FACTOR = np.arange(timewindow_number-1).reshape(timewindow_number-1,1)
    AREA_FACTOR = np.ones([timewindow_number-1,1])*area_idx
    CHANNEL_FACTOR = np.ones([timewindow_number-1,1])*channel_idx
    SEED_FACTOR = np.ones([timewindow_number-1,1])*seed

    timewindows_dataframe = np.hstack([ 
        SEED_FACTOR,
        AREA_FACTOR, 
        TIC_FACTOR,
        WIN_FACTOR,
        CHANNEL_FACTOR,
        timewindows_store])
    
    return timewindows_dataframe

# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------

class LesionTypes:
    NOLESION = 0
    CRB = 1
    CRB_BGTHA = 2
    CRB_CRBTHA = 3
    STNCRB = 4

class ScoreTypes:
    ANALYTIC = 0
    SIMPLE = 1

class TicsSimulation():
    
    AREAS =  ["PUTrd1", "PUTrd2", "STN", "GPi", "GPe","TH", "CTX", "CRB"] 
    META_AREAS = ["PUT", "STN", "GPi", "GPe", "TH", "CTX", "CRB"]
    SELECTED_META_AREAS = ["PUT",  "GPi", "GPe", "CTX", "CRB"] 
    MEASURE_INTERVALS = 101
    INNER_WINDOW = 201
    SAVE_POTENTIALS = False
    SCORE = ScoreTypes.SIMPLE

    def __init__(self, rng=RNG(1), pm=None, SEED=None,
            TRIALS=10, LESION=LesionTypes.NOLESION, DA_VALUE=1, 
            RANDOM_EXTERNAL=False,
            CTPS=0.00000, CNPS=0.2000, CNNS=0.03125, DATS=0.00000, DANS=0.15625, RAPS=0.00000,
            CTPL=0.03500, CNPL=0.18625, CNNL=0.75000, DATL=0.10000, DANL=0.07500, RAPL=0.02500,
            CTPO=2.00000, CNPO=1.80000, CNNO=3.00000, DATO=0.10000, DANO=6.00000, RAPO=2.00000,
            CTPA=4.00000, CNPA=1.00000, CNNA=0.00000, DATA=0.00000, DANA=0.07800, RAPA=3.00000
            ):
       
        """
        @param  rng                 random number generator
        @param  pm                  parameter_manager
        @param  SEED                seed of the random number generator
        @param  TRIALS              number of trials
        @param  LESION              type of lesion
        @param  DA_VALUE            amplitude of dopamine activation
        @param  RANDOM_EXTERNAL     external stimulus is random
        @param  CTPS                starting point of the excitatory cortical stimulus during a TIC window
        @param  CNPS                starting point of the excitatory cortical stimulus during a NOTIC window
        @param  CNNS                starting point of the inhibitory cortical stimulus during a NOTIC window
        @param  RAPS                starting point of the inhibitory cortical stimulus during a RANDOM window
        @param  DATS                starting point of the dopaminergic stimulus during a TIC window
        @param  DANS                starting point of the dopaminergic stimulus during a NOTIC window
        @param  CTPL                length of the excitatory cortical stimulus during a TIC window
        @param  CNPL                length of the excitatory cortical stimulus during a NOTIC window
        @param  CNNL                length of the inhibitory cortical stimulus during a NOTIC window   
        @param  DATL                length of the dopaminergic stimulus during a TIC window
        @param  DANL                length of the dopaminergic stimulus during a NOTIC window
        @param  RAPL                length of the excitatory cortical stimulus during a RANDOM window
        @param  CTPO                slope of the excitatory cortical stimulus during a TIC window
        @param  CNPO                slope of the excitatory cortical stimulus during a NOTIC window
        @param  CNNO                slope of the inhibitory cortical stimulus during a NOTIC window
        @param  DATO                slope of the dopaminergic stimulus during a TIC window
        @param  DANO                slope of the dopaminergic stimulus during a NOTIC window
        @param  RAPO                slope of the excitatory cortical stimulus during a RANDOM window
        @param  CTPA                amplitude of the excitatory cortical stimulus during a TIC window
        @param  CNPA                amplitude of the excitatory cortical stimulus during a NOTIC window
        @param  CNNA                amplitude of the inhibitory cortical stimulus during a NOTIC window
        @param  DATA                amplitude of the dopaminergic stimulus during a TIC window
        @param  DANA                amplitude of the dopaminergic stimulus during a NOTIC window
        @param  RAPA                amplitude of the excitatory cortical stimulus during a RANDOM window

        """


        self.rng = rng   # random number generator
        self.SEED = SEED   # random seed
        self.TRIALS = TRIALS   # number of trials
        self.RANDOM_EXTERNAL = RANDOM_EXTERNAL   # xternal stimulus is random
            
        self.EP_TIC_P_START = CTPS
        self.EP_NOTIC_P_START = CNPS
        self.EP_NOTIC_N_START = CNNS
        self.EP_P_START = RAPS
        self.DA_TIC_START = DATS
        self.DA_NOTIC_START = DANS
        self.EP_TIC_P_LENGTH = CTPL
        self.EP_NOTIC_P_LENGTH = CNPL
        self.EP_NOTIC_N_LENGTH = CNNL  
        self.DA_TIC_LENGTH = DATL
        self.DA_NOTIC_LENGTH = DANL
        self.EP_P_LENGTH = RAPL     
        self.EP_TIC_P_SIGMA = CTPO
        self.EP_NOTIC_P_SIGMA = CNPO
        self.EP_NOTIC_N_SIGMA = CNNO  
        self.DA_TIC_SIGMA = DATO
        self.DA_NOTIC_SIGMA = DANO
        self.EP_P_SIGMA = RAPO       
        self.EP_TIC_P_AMP = CTPA
        self.EP_NOTIC_P_AMP = CNPA
        self.EP_NOTIC_N_AMP = CNNA
        self.DA_TIC_AMP = DATA
        self.DA_NOTIC_AMP = DANA
        self.EP_P_AMP = RAPA

        # MAIN PARAMS
        for parameter, value in pm.get_main_params().items():
            setattr(self, parameter, value);
           
        self.STIME = self.TRIAL_LENGTH*self.TRIALS                  # duration of the simulation
        self.NI = int(np.floor(self.STIME 
            /float(self.TRIAL_LENGTH)))                             # number of tics
        
        # LESIONS
        self.w_crb2tha *= ( (LESION != LesionTypes.CRB) and (LESION != LesionTypes.CRB_BGTHA) );
        self.w_crb2crbtha *= ( (LESION != LesionTypes.CRB) and (LESION != LesionTypes.CRB_CRBTHA) ); 
        self.w_bg2crb *= ( LESION != LesionTypes.STNCRB ); 
        
        # DOPAMINE 
        self.da_max = DA_VALUE

        # OBJECTS
        self.bg =  BG(
                name='bg', 
                dt=self.DT, 
                tau=[  self.TAU*self.tau_amp_sd1,
                       self.TAU*self.tau_amp_sd2,
                       self.TAU*self.tau_amp_stn, 
                       self.TAU*self.tau_amp_gpi,
                       self.TAU*self.tau_amp_gpe], 
                stime=self.STIME*self.DT,
                N=self.N,
                rng=rng )

        self.thalamus = Layer('thalamus',
                dt=self.DT, 
                tau=self.TAU*self.tau_amp_thal, 
                stime=self.STIME*self.DT, 
                N=self.N )

        self.crbthalamus = Layer('crbthalamus',
                dt=self.DT, 
                tau=self.TAU*self.tau_amp_crbthal, 
                stime=self.STIME*self.DT,
                N=self.N )
        
        self.cortex = Layer('cortex', 
                dt=self.DT,
                tau=self.TAU*self.tau_amp_crx, 
                stime=self.STIME*self.DT,
                N=self.N,
                th = self.th_crx)

        self.cerebellum = Cerebellum('cerebellum',
                dt=self.DT,
                TAU=self.TAU*self.tau_amp_crb,
                stime=self.STIME*self.DT,
                N=self.N,
                rng=rng)

        self.da1p = Layer('dap',
                dt=self.DT,
                tau=self.TAU*self.tau_amp_d1,
                stime=self.STIME*self.DT,
                N=1)

        self.da2p = Layer('dap',
                dt=self.DT,
                tau=self.TAU*self.tau_amp_d2,
                stime=self.STIME*self.DT,
                N=1)

        # BG WEIGHTS
        for parameter, value in pm.get_bg_params().items():
            setattr(self.bg, parameter, value);
         
        
        self.external_stimuli = dict()

    def noise(self, bl, sd, n) :
        return (sd*self.rng.randn(n) +bl)

    def simulate(self):
        
        # INITIALIZE INPUTS

        # DOPAMINE SIGNAL 
        da = self.da_min*np.ones(self.STIME) 
     
        # ADD TIC-SIMULATING INPUT
        for i in xrange(self.NI) :
            if(i>0) : 
                start = i*self.TRIAL_LENGTH
                end = start+self.TIC_INTERVAL 
                da[start:end] += self.da_max

                    
        # STRIATAL INPUT
        inp = self.inp_noise_std*self.rng.randn(self.bg.N, self.STIME) \
                + self.noise_bl_inp   
             
        # CORTICAL INPUT
        crx_inp = self.crx_inp_noise_std*self.rng.randn(self.bg.N, self.STIME) \
                + self.noise_bl_crx_inp 
        
        
        # DEFINE EXTERNAL STIMULUS

        if not self.RANDOM_EXTERNAL :
         
            # THE EXTERNAL STIMULUS IS FIXED 

            x = np.linspace(-5,5,int(self.TRIAL_LENGTH*(self.EP_TIC_P_LENGTH)))
            y1 = np.exp(-(x/self.EP_TIC_P_SIGMA)**2)
            x = np.linspace(-5,5,int(self.TRIAL_LENGTH*(self.EP_NOTIC_P_LENGTH)))
            y2 = np.exp(-(x/self.EP_NOTIC_P_SIGMA)**2)
            x = np.linspace(-5,5,int(self.TRIAL_LENGTH*(self.EP_NOTIC_N_LENGTH)))
            y3 = np.exp(-(x/self.EP_NOTIC_N_SIGMA)**2)
            x = np.linspace(-5,5,int(self.TRIAL_LENGTH*(self.DA_NOTIC_LENGTH)))
            y4 = np.exp(-(x/self.DA_NOTIC_SIGMA)**2)
            x = np.linspace(-5,5,int(self.TRIAL_LENGTH*(self.DA_TIC_LENGTH)))
            y5 = np.exp(-(x/self.DA_TIC_SIGMA)**2)
            
            self.y1 = y1
            self.y2 = y2
            self.y3 = y3
            self.y4 = y4
            self.y5 = y5

            ticr = 1*(np.arange(self.NI)>(self.NI/2))
            self.rng.shuffle(ticr)

            for i in xrange(self.NI) :
                
                if(i>0) : 
                    if ticr[i]>0 :
                        gap = int(self.TRIAL_LENGTH*(self.EP_TIC_P_START)) 
                        start = (i*self.TRIAL_LENGTH) - gap
                        end = start + int(self.TRIAL_LENGTH*(self.EP_TIC_P_LENGTH))
                        self.external_stimuli["TIC_CTX_PS"] = np.vstack(( np.arange(start,end) - start - gap, self.EP_TIC_P_AMP*np.tile(y1, [self.bg.N,1] ) )).T
                        crx_inp[:, start:end] += self.EP_TIC_P_AMP*np.tile(y1, [self.bg.N,1] )  
                    
                        gap = int(self.TRIAL_LENGTH*(self.DA_TIC_START))
                        start = (i*self.TRIAL_LENGTH) - gap
                        end = start + int(self.TRIAL_LENGTH*(self.DA_TIC_LENGTH))   
                        self.external_stimuli["TIC_DA_PS"] = np.vstack(( np.arange(start,end) - start - gap, self.DA_TIC_AMP*y5 )).T
                        da[start:end] += self.DA_TIC_AMP*y5  
                    
                    else:
                        gap = int(self.TRIAL_LENGTH*(self.DA_NOTIC_START))
                        start = (i*self.TRIAL_LENGTH) - gap
                        end = start + int(self.TRIAL_LENGTH*(self.DA_NOTIC_LENGTH))   
                        self.external_stimuli["NOTIC_DA_PS"] = np.vstack(( np.arange(start,end) - start - gap, self.DA_NOTIC_AMP*y4 )).T
                        da[start:end] += self.DA_NOTIC_AMP*y4  
   
                        gap = int(self.TRIAL_LENGTH*(self.EP_NOTIC_P_START))
                        start = (i*self.TRIAL_LENGTH) - gap
                        end = start + int(self.TRIAL_LENGTH*(self.EP_NOTIC_P_LENGTH))
                        self.external_stimuli["NOTIC_CTX_PS"] = np.vstack(( np.arange(start,end) - start - gap, self.EP_NOTIC_P_AMP*np.tile(y2, [self.bg.N,1] ) )).T
                        crx_inp[:, start:end] += self.EP_NOTIC_P_AMP*np.tile(y2, [self.bg.N,1] )  
        
                        gap = int(self.TRIAL_LENGTH*(self.EP_NOTIC_N_START))
                        start = (i*self.TRIAL_LENGTH) - gap
                        end = start + int(self.TRIAL_LENGTH*(self.EP_NOTIC_N_LENGTH))
                        self.external_stimuli["NOTIC_CTX_NS"] = np.vstack(( np.arange(start,end) - start - gap, self.EP_NOTIC_N_AMP*np.tile(y3, [self.bg.N,1] ) )).T
                        crx_inp[:, start:end] -= self.EP_NOTIC_N_AMP*np.tile(y3, [self.bg.N,1] )  
        
        
        else:
            
            # THE EXTERNAL STIMULUS IS RANDOM 
             
            x = np.linspace(-5,5,int(self.TRIAL_LENGTH*(self.EP_P_LENGTH)))
            y = np.exp(-(x/self.EP_P_SIGMA)**2)
            
            self.y = y

            for i in xrange(self.NI) :
                    
                if(i>0) :
                       
                    gap = int(self.TRIAL_LENGTH*(self.EP_P_START)) 
                    start = (i*self.TRIAL_LENGTH) - gap
                    end = start + int(self.TRIAL_LENGTH*(self.EP_P_LENGTH))
                    crx_inp[:, start:end] += self.EP_P_AMP*self.rng.uniform(0,3)*np.outer(self.rng.rand(self.bg.N), y) 


        # CEREBELLAR INPUT
        crb_inp = self.crb_inp_noise_std*self.rng.randn(self.bg.N, self.STIME) \
                + self.noise_bl_crb_inp


        self.cortex.reset_data()
        self.cerebellum.reset_data()
        self.thalamus.reset_data()
        self.bg.reset_data()
        self.da1p.reset_data()
        self.da2p.reset_data()
    
        crb_inp_store = np.zeros([self.STIME, self.N])
        

        # START THE MAIN CICLE 
        for t in range(self.STIME) : 
        
            # INPUTS FROM THE PREVIOUS STEP 
            crx = self.cortex.o
            crb = self.cerebellum.o
            tha = self.thalamus.o
            crbtha = self.crbthalamus.o
            gpi = self.bg.gpi
            da1_out = self.da1p.o 
            da2_out = self.da2p.o 

            # RUN THE INTEGRATION STEP OF EACH OBJECT
            self.bg.step( 
                    + inp[:,t] 
                    + self.w_tha2bg*tha, 
                    crx, 
                    da1_out, 
                    da2_out 
                    )

            self.thalamus.step( 
                    + self.bl_thalamus 
                    + self.w_crx2thal*crx 
                    + self.w_crb2tha*crb 
                    - self.w_bg2tha*gpi  
                    + self.noise(
                        self.noise_bl_thalamus, 
                        self.noise_sd_thalamus, 
                        self.N) 
                    )

            self.crbthalamus.step( 
                    + self.w_crx2crbthal*crx 
                    + self.w_crb2crbtha*crb 
                    + self.noise(self.noise_bl_crbthalamus, 
                        self.noise_sd_crbthalamus, self.N)  
                    )

            self.cortex.step(       
                    + self.w_tha2crx*(tha)     
                    + self.w_crbtha2crx*(crbtha)        
                    + crx_inp[:,t] 
                    )

            self.cerebellum.step( 
                    + self.w_crx2crb*crx  
                    + crb_inp[:,t] 
                    + self.w_bg2crb*self.bg.stn 
                    )

            self.da1p.step( 
                    + self.w_inp2da1*da[t] 
                    )
            self.da2p.step( 
                    + self.w_inp2da2*da[t] 
                    )
         
            # STORE DATA FOR EACH OBJECT
            self.bg.store(t,self.SAVE_POTENTIALS)
            self.thalamus.store(t,self.SAVE_POTENTIALS)
            self.cortex.store(t,self.SAVE_POTENTIALS)   
            self.cerebellum.store(t,self.SAVE_POTENTIALS)   
            self.da1p.store(t,self.SAVE_POTENTIALS)
            self.da2p.store(t,self.SAVE_POTENTIALS)
            
            crb_inp_store[t,:] = (
                    + self.w_crx2crb*crx  
                    + crb_inp[:,t] 
                    + self.w_bg2crb*self.bg.stn 
                    )


        # FIND IF CORTICAL TIC HAPPENS  
        cortex_data = (self.cortex.data).sum(0)/float(self.N)
        maxamp = get_maxamp(cortex_data,  self.NI,  diameter= 101, displace = 50 );
        
        # order based on timewindow occurrence
        maxamp = maxamp[np.argsort(maxamp[:,0]),:] 
        
        # find tic the simple way : based on a threshold 
        idx =  maxamp[:,1]>self.tic_threshold 

        # divide indices in tic group and notic group
        tic_group = maxamp[idx==1,:]
        notic_group = maxamp[idx==0,:]
        tic_group_idx = tic_group[:,0].astype('int')
        notic_group_idx = notic_group[:,0].astype('int')
        
        # store the number of tics
        n_tics = sum(idx)
    
        # SAVE DATA
        
        # collect timeseries to manipulate
        time_series = []
        for area_idx in [self.bg.l_sd1, self.bg.l_sd2, self.bg.l_stn, 
                self.bg.l_gpi, self.bg.l_gpe] :
            time_series.append( self.bg.data[area_idx] )
        time_series.append( self.thalamus.data )
        time_series.append( self.cortex.data )
        if self.SAVE_POTENTIALS:
            time_series.append( 
                        self.cerebellum.data[self.cerebellum.lppc]+ 
                        self.cerebellum.data[self.cerebellum.lpgc].mean(0) 
                    )
        else:
            time_series.append( 
                        self.cerebellum.data[self.cerebellum.lapc]+ 
                        self.cerebellum.data[self.cerebellum.lagc].mean(0) 
                    )
        
        self.raw_dataframe = np.array([])   # store the raw data 
                                            # (not splitted in timewindows)
        self.timewindows_dataframe = np.array([])   # store the timewindows

        # cut real timewindows in the activation data of all areas
        for area_idx in xrange(len(time_series)):
            for channel_idx in xrange(self.bg.N) :        
                
                cur_raw_dataframe = time_series[area_idx][channel_idx]

                cur_timewindows_dataframe = get_tic_timewindows(
                        time_series = time_series[area_idx][channel_idx], 
                        timewindow_number = self.NI,
                        area_idx = area_idx, 
                        channel_idx = channel_idx,
                        tic_group_idx = tic_group_idx, 
                        notic_group_idx = notic_group_idx, 
                        seed = self.SEED,
                        diameter = self.INNER_WINDOW )   
                
                if (area_idx*self.N + channel_idx) == 0 :
                    self.raw_dataframe = cur_raw_dataframe
                    self.timewindows_dataframe = cur_timewindows_dataframe
                else :
                    self.raw_dataframe = np.vstack([self.raw_dataframe, \
                            cur_raw_dataframe])    
                    self.timewindows_dataframe = \
                            np.vstack([self.timewindows_dataframe, \
                            cur_timewindows_dataframe])
        
        # cut real timewindows in the activation data of da
        self.da_timewindow_dataframe = get_timewindows(self.da1p.data, \
                self.NI, diameter=self.INNER_WINDOW  )
        self.da_raw_dataframe = self.da1p.data

    def get_dataframe(self):

        columns = ['SEED', 'AREA', 'TIC', 'WIN', 'CHANNEL'] +\
                range( self.INNER_WINDOW)
        df = pd.DataFrame(self.timewindows_dataframe, columns=columns)
       
        areas = df.AREA.as_matrix().ravel()
        areas[areas == 0] = 0
        areas[areas == 1] = 0
        areas[areas == 6] = 6
        df['META_AREA'] = areas.copy()

        df.AREA = df.AREA.astype("category")
        df.AREA = df.AREA.cat.rename_categories( self.AREAS )
        
        df.META_AREA = df.META_AREA.astype("category")
        df.META_AREA = df.META_AREA.cat.rename_categories( self.META_AREAS )
        
        df.TIC = pd.Categorical(df.TIC, categories=[0,1], ordered=False)
        df.TIC = pd.Categorical(df.TIC).rename_categories(['NOTIC','TIC'])

        self.df = df

        RAW_AREAS = np.tile(self.AREAS,(self.bg.N,1)).T.ravel()
        RAW_AREAS = np.hstack(("DA",RAW_AREAS))
        for x in range(len(RAW_AREAS)) :
            if RAW_AREAS[x].startswith("PUT"):
                RAW_AREAS[x] = "PUT"

        self.raw_df = pd.DataFrame(
                np.vstack((
                    self.da_raw_dataframe,
                    self.raw_dataframe )) )
        self.raw_df.insert(0,"AREA",RAW_AREAS)

        self.raw_df = self.raw_df.groupby("AREA").mean()
        self.raw_df = self.raw_df.reset_index()

    def get_means(self):

        self.get_dataframe()

        meandf = self.df.groupby(['META_AREA','TIC']).mean()
        meandf = meandf.reset_index()

        meandf = meandf.loc[meandf.META_AREA!='STN']
        meandf = meandf.loc[meandf.META_AREA!='TH']
        self.meandf = meandf 

    def score_analytic(self, target):
           
        self.get_means()
        self.targets = target

        mdf = self.df.groupby(['META_AREA','WIN','TIC'], ).mean()
        mdf = mdf.reset_index()
        
        cols = ["META_AREA","TIC","WIN",'SEED']+list(np.linspace(0,self.INNER_WINDOW-1, 
                    self.MEASURE_INTERVALS ).astype(int))
        mdf = mdf.loc[:, cols]
        
        scores = 0
        tics = 0
        wins = mdf.WIN.unique()

        for win in wins:
            
            wdf = mdf[mdf.WIN==win]
            wdf = wdf[ np.logical_not( wdf.SEED.isnull() ) ]    
            wdftic = wdf.TIC.unique()[0]
            tics += wdftic == 'TIC'
            wtarget = target[target.TIC == wdftic]

            measure = wdf
            trg = wtarget
               
            if self.target_areas is None:
                self.target_areas = self.SELECTED_META_AREAS   
            
            select = [area in  self.target_areas for area in measure.META_AREA ]
            measure = measure.loc[ select ]
           
            select = [area in  self.target_areas for area in trg.META_AREA ]
            trg = trg.loc[ select ]


            measure = measure.iloc[:,4:].as_matrix()
            trg = trg.iloc[:,2:].as_matrix()
            
            y = measure
            t = trg
            
            scores += np.sqrt(np.mean((y - t)**2)) / (np.max(y) - np.min(y))
       
        proptics = tics/float(len(wins))

        if proptics < 0.2 or proptics > 0.8:
            return np.nan

        scores /= float(len(wins))

        return scores 

    def score_simple(self, target):
        
        self.get_means()
        means = self.meandf 
        
        cols = ["META_AREA","TIC"]+list(np.linspace(0,self.INNER_WINDOW-1, 
                    self.MEASURE_INTERVALS ).astype(int))
        means = means.loc[:, cols]
        
        self.measures = means   
        self.targets = target
     
        if self.target_areas is None:
            self.target_areas = self.SELECTED_META_AREAS   
        select = [False for x in means.META_AREA]  
        for area in self.target_areas:
            select = select | (means.META_AREA==area)
        
        means.loc[:,"idx"] = range(len(means))
        measure = means.loc[ select ]
        idx = measure.idx.as_matrix()
        measure_target = target.iloc[idx,:]
         
        if all(measure.META_AREA == measure_target.META_AREA) :
        
            measure = measure.iloc[:,2:-1].as_matrix()
            target = measure_target.iloc[:,2:].as_matrix()
            
            y = measure
            t = target

            return np.sqrt(np.mean((y - t)**2)) / (np.max(y) - np.min(y))

        return np.nan


    def score(self, target):

        if self.SCORE == ScoreTypes.ANALYTIC :
            return self.score_analytic(target)
        elif self.SCORE == ScoreTypes.SIMPLE :
            return self.score_simple(target)


def load_target_data():

    datafile = "data_scaled"
    names = [ 'META_AREA','TIC'] + ["p{:03d}".format(x) 
            for x in xrange(TicsSimulation.MEASURE_INTERVALS)]

    data =  pd.read_csv(datafile, names=names,sep = "\s+")
    
    return data



if __name__ == "__main__" :

    TicsSimulation.SAVE_POTENTIALS=False
    TicsSimulation.SCORE = ScoreTypes.ANALYTIC

    import matplotlib.pyplot as plt
    np.set_printoptions(suppress=True, precision=5, linewidth=9999999)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--best',
            help="Choose best parameters",
            action="store_true", default=False) 
    parser.add_argument('-r','--random',
            help="Choose parameters randomly",
            action="store_true", default=False)
    parser.add_argument('-m','--minimal',
            help="Choose parameters randomly with a minimal variance",
            action="store", default=0.5) 
    parser.add_argument('-e','--random_external',
            help="Random stimulus is external",
            action="store_true", default=False)     
    parser.add_argument('-p','--plot',
            help="Plot results",
            action="store_true", default=False)   
    parser.add_argument('-s','--save',
            help="Save data into a file",
            action="store_true", default=False)  
    parser.add_argument('-d','--da_value',
            help="Level of dopamine",
            action="store", default=0.5)     


    args = parser.parse_args()
    

    RAND_PAR = bool(args.random) 
    BESTS = bool(args.best) 
    PLOT = bool(args.plot) 
    MIN_RAND = float(args.minimal) 
    RANDOM_EXTERNAL = float(args.random_external) 
    SAVE = bool(args.save)  
    DA_VALUE = float(args.da_value)  


    SEED = np.fromstring(os.urandom(4), dtype=np.uint32)[0]


    RNG = np.random.RandomState(SEED)
     
    pm = PM()
    if RAND_PAR :
        params = RNG.uniform(-MIN_RAND,MIN_RAND,pm.NUM_PARAMS) +0.5
        pm.set_parameters( params )
    if BESTS :
        import pandas as pd
        best = pd.read_csv("ga_best",  sep='\t', encoding='utf-8', index_col=0 )
        seeds = np.loadtxt("all_seeds",ndmin=1).astype(int)
        params = best.iloc[-1,2:].as_matrix().ravel()
        pm.set_parameters( params)
        RNG = np.random.RandomState(seeds[0])


    sim = TicsSimulation(
                rng = RNG,
                DA_VALUE = DA_VALUE,
                TRIALS = 11, 
                LESION = LesionTypes.NOLESION, 
                SEED = SEED, 
                pm=pm,
                RANDOM_EXTERNAL=RANDOM_EXTERNAL )
   
    sim.simulate()

    data = load_target_data()
    score = sim.score(data)
  
    print
    print "{:>20} : {:=10.2f}".format('score',float(score))
    print "{:>20} : {:<10d}".format( 'seed',SEED)
    print
    
    
    for par,val in sorted(pm.get_all_params().items()):
        if par in pm.ga_params.keys() :
            ga_t = pm.ga_params[par]
            if ga_t == True:     
                lims = pm.set_par_list[par]    
                mar = np.abs(lims[1]-lims[0])/2.0
                low = val - mar
                high = val + mar
            else :
                low = 0.0
                high = 0
            print "{:<20}    {:>10.4f}    {:>10.4f}    {:>10.4f}    {:}"\
                    .format(par, low, high, val, ga_t if "True" else "False" )

    if SAVE == True :
        sim.df.to_csv("d_{:020d}".format(SEED), sep='\t', encoding='utf-8')
    
        for k, v in sim.external_stimuli.items():
            np.savetxt("d_{:020d}_{}".format(SEED,k), v)

    if PLOT :


           
        plt.ion()

        from plot_demo import plot_means
        from plot_demo import plot_raw_layers
        from plot_demo import plot_ctx_channels
        
        plot_means(sim)
        plot_raw_layers(sim)
        plot_ctx_channels(sim)

        plt.figure()
        plt.subplot(2,2,1)
        d = sim.external_stimuli["TIC_CTX_PS"] 
        plt.fill_between(d[:,0], d[:,1]*0, d[:,1])
        plt.xlim([-100,100])
        plt.ylim([0,3.5])
        
        plt.subplot(2,2,2)
        d = sim.external_stimuli["TIC_DA_PS"] 
        plt.fill_between(d[:,0], d[:,1]*0, d[:,1])
        plt.xlim([-100,100])
        plt.ylim([0,3.5])
        
        plt.subplot(2,2,3)
        d = sim.external_stimuli["NOTIC_CTX_PS"] 
        plt.fill_between(d[:,0], d[:,1]*0, d[:,1])
        plt.xlim([-100,100])
        plt.ylim([0,3.5])
        
        plt.subplot(2,2,4)
        d = sim.external_stimuli["NOTIC_DA_PS"] 
        plt.fill_between(d[:,0], d[:,1]*0, d[:,1])
        plt.xlim([-100,100])
        plt.ylim([0,3.5])

        raw_input()
    

