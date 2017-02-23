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


import numpy as np
from numpy.random import RandomState as rng

###################################################################################################################################
## UTILS ##########################################################################################################################

import os
import time

def outfun(x,th=0,slope=1, bound = 4.0) : 
    """ 
    This function passes input through a tanh and the returns only the positive part of the result

    @brief truncated tanh function

    @param  x       Input. It can be a scalar, vector or matrix. 
    @param  th      Threshold. It defines an input threshold above which the output is not zero. 
    @param  slope   Slope of the tanh. It defines how fast the output reach its asymptote as far the input grows.
    @param  bound   saturation bound of the activation
    """

    r = np.tanh(slope*(x-th)/bound)*bound    # tanh of the input 
    r = r*(r>0)    # truncation             

    return r

###################################################################################################################################

def gauss(x,mu,sd) :
    return np.exp(-( (x-mu)**2 )/(2*float(sd)**2) )

###################################################################################################################################


###################################################################################################################################
## BASAL GANGLIA MODULE ###########################################################################################################
###################################################################################################################################
class BG :
    """    
    This class defines a basal ganglia module consisting in five layers: 
        sd1 -> striatum expressing D1-liike receptors
        sd2 -> striatum expressing D2-liike receptors
        stn -> subthalamic nucleus
        gpi -> internal globus pallidus
        gpe -> external globus pallidus

    @brief A class defining a basal-ganglia module

    """
   
    ###############################################################################################################################
    def __init__(self, name,  dt=0.001, stime=1, tau=[.005,.005,.005,.005,.005], N=3, rng=rng(1) ) :
        """ 
        @brief constructor 

        @param  name    Name. A string defining the object to name the file on which data are written. 
        @param  dt      Delta t. A double value defining the integration time-step for the numerical integration of leaky integrators.     
        @param  stime   Simulation time. An integer value defining the duration of the storage window in time-steps.
        @param  tau     Decay of the leaky. A double value defining the decaying time of leaky integrators. 
        @param  N       Number of channels in the module. An integer value.
        @param  rng     Random number generator
        """
        
        self.rng = rng

        ##################################################
        ## PARAMETERS ####################################
        ##################################################
        self.dt = dt     # integration step
        self.stime = int(stime/self.dt)     # interval of data storage
        self.decay = tau     # decay of units
        self.N = N     # number of BG channels

        #################################################
        ## WEIGHTS ###################################### 
        #################################################
        self.w_cri2sd1 = 1     # input to striatum (D1) 
        self.w_cri2sd2 = 2    # input to striatum (D2) 
        self.w_cri2stn = 0.5     # input to STN 
        self.w_cro2sd1 = 4.0     # reciprocal cortex to striatum (D1)
        self.w_cro2sd2 = .5    # reciprocal cortex to striatum (D2)
        self.w_cro2stn = 3    # reciprocal cortex to STN
        self.w_sd12gpi = 5     # striatum (D1) to GPI
        self.w_stn2gpi = 2     # STN to GPI
        self.w_sd22gpe = 4    # striatum (D2) to GPE
        self.w_stn2gpe = 2     # STN to GPE
        self.w_gpi2gpe = 0     # GPI to GPE
        self.w_gpe2gpi = 0.1     # GPE to GPI
        self.w_gpe2stn = 0.7     # GPE to STN
        self.w_str_inn = -.5
        self.str_field = (1-np.eye(self.N))
        
        mask = np.repeat(np.arange(N),N).reshape(N,N).T
        self.stn_fields = np.ones([N,N]) 



        ################################################# 
        ## BASELINES #################################### 
        #################################################  
        self.bl_sd1 = 0.1     # striatum (D1) activation without DA 
        self.da_sd1 = 0.5     # striatum (D1)  DA factor amplitude 
        self.da_th_sd1 = 0.0     # D1R  threshold
        self.bl_sd2 = 0.1    # striatum (D2) activation without DA (inverse value)
        self.da_sd2 = 10    # striatum (D2)  DA factor amplitude 
        self.bl_gpe = 0.1     # GPE baseline
        self.bl_gpi = 0.3     # GPI baseline

        self.blm_sd1 = 0.0
        self.blm_sd2 = 0.0
      
        self.noise_bl_sd1 = 0
        self.noise_sd_sd1 = 0
        
        self.noise_bl_sd2 = 0
        self.noise_sd_sd2  = 0
        
        self.noise_bl_stn = 0
        self.noise_sd_stn  = 0 
        
        self.noise_bl_gpi = 0
        self.noise_sd_gpi  = 0
        
        self.noise_bl_gpe = 0
        self.noise_sd_gpe  = 0



        ################################################# 
        ## Initializations ############################## 
        ################################################# 
        
        # Activations
        self.sd1 = np.zeros(self.N) 
        self.sd2 = np.zeros(self.N) 
        self.stn = np.zeros(self.N) 
        self.gpi = np.zeros(self.N) 
        self.gpe = np.zeros(self.N) 
        self.da = 0 
        
        # Potentials
        self.sd1p = np.zeros(self.N)
        self.sd2p = np.zeros(self.N)
        self.stnp = np.zeros(self.N)
        self.gpip = np.zeros(self.N)
        self.gpep = np.zeros(self.N)
        
        # storage index labels 
        ( self.l_sd1,
                self.l_sd2,
                self.l_stn,
                self.l_gpi,
                self.l_gpe ) = range(5)
        
        # storage arrays
        self.data = dict()
        self.data[self.l_sd1] = np.zeros([self.N,self.stime])
        self.data[self.l_sd2] = np.zeros([self.N,self.stime])
        self.data[self.l_stn] = np.zeros([self.N,self.stime])
        self.data[self.l_gpi] = np.zeros([self.N,self.stime])
        self.data[self.l_gpe] = np.zeros([self.N,self.stime])

        self.name = name     # name of the class object



    def noise(self, bl, sd, n) :
        return (sd*self.rng.randn(n) +bl)


    ###############################################################################################################################
    def reset_data(self):
        """
        @brief Reset storage arrays 
        """
        for k in self.data :
            self.data[k] = self.data[k]*0    # reset each array to zero



    ###############################################################################################################################
    def store(self,timestep, potentials = False ):
        """
        @brief Store activity 

        @param  timestep    Current timestep
        """
        
        windowtimestep = timestep%self.stime    # the % operator implements the remainder of division 

        if not potentials:
            self.data[self.l_sd1][:,windowtimestep] = self.sd1
            self.data[self.l_sd2][:,windowtimestep] = self.sd2
            self.data[self.l_stn][:,windowtimestep] = self.stn 
            self.data[self.l_gpi][:,windowtimestep] = self.gpi 
            self.data[self.l_gpe][:,windowtimestep] = self.gpe 
        else :
            self.data[self.l_sd1][:,windowtimestep] = self.sd1p
            self.data[self.l_sd2][:,windowtimestep] = self.sd2p
            self.data[self.l_stn][:,windowtimestep] = self.stnp
            self.data[self.l_gpi][:,windowtimestep] = self.gpip 
            self.data[self.l_gpe][:,windowtimestep] = self.gpep    
    
    ###############################################################################################################################
    def reset(self):
        """ 
        @brief Reset activations 
        """
         
        self.sd1p = ones(self.N)*self.bl_sd1
        self.sd2p = ones(self.N)*self.bl_sd2
        self.stnp = 0 
        self.gpip = ones(self.N)*self.bl_gpi
        self.gpep = ones(self.N)*self.bl_gpe
         
        self.sd1 = ones(self.N)*self.bl_sd1
        self.sd2 = ones(self.N)*self.bl_sd2
        self.stn = 0 
        self.gpi = ones(self.N)*self.bl_gpi
        self.gpe = ones(self.N)*self.bl_gpe
        da = 0



    ###############################################################################################################################
    def step(self, cri, cro, da1, da2) :
        """
        A step of unmerical integration. Each layer is updated using euler integration:

        the leaky integrator        tau*dy/dt = -y + input + W*f(y) 
        becomes                     y(t) = y(t-1) + (dt/tau) * (-y(t-1) + input +  W*f(y))
        
        @ brief A single spreading step 
        
        @param  cri Cortical overlapping input. A vector of length N defining the input from other cortices.
        @param  cro Cortical loop input. A vector of length N defining the input from the cortex in loop with the current basal gganglia module.
        @param  da  Dopamine. A vector of length N defining the current dopaminergic efflux to each of the N channels.
        @param  da1  Dopamine. A vector of length N defining the current dopaminergic efflux to each of the N channels.
        @param  da2  Dopamine. A vector of length N defining the current dopaminergic efflux to each of the N channels.
        """

        self.da = da1 - da2

        d1_modulation = (self.bl_sd1 + self.da_sd1*( da1 * (da1 > self.da_th_sd1 ) * (da1 - self.da_th_sd1 ) ))
        d2_modulation = self.bl_sd2 + self.da_sd2*da2
         
        
        # numerical integration of sd1 activity
        self.sd1p += (self.dt/self.decay[self.l_sd1])*(
                - self.sd1p
                + d1_modulation*(  # D1 dopamine multiplies : (a + b*DA)*(input)
                + self.blm_sd1
                + self.noise(self.noise_bl_sd1, self.noise_sd_sd1, self.N )
                + self.w_cri2sd1*cri 
                + self.w_cro2sd1*cro 
                + np.dot(self.w_str_inn*self.str_field,self.sd1) )) 
        self.sd1 = outfun(self.sd1p)

        # numerical integration of sd2 activity
        self.sd2p += (self.dt/self.decay[self.l_sd2])*(
                - self.sd2p
                + (d1_modulation/d2_modulation)*( # D2 dopamine divides : a/(b + DA)*(input)
                + self.blm_sd2
                + self.noise(self.noise_bl_sd2, self.noise_sd_sd2, self.N )
                + self.w_cri2sd2*cri
                + self.w_cro2sd2*cro
                + np.dot(self.w_str_inn*self.str_field,self.sd2) )) 
        self.sd2 = outfun(self.sd2p)

        # numerical integration of stn activity
        self.stnp += (self.dt/self.decay[self.l_stn])*(
                - self.stnp
                + self.noise(self.noise_bl_stn, self.noise_sd_stn, self.N )
                + self.w_cri2stn*cri
                + self.w_cro2stn*cro
                - self.w_gpe2stn*self.gpe
                )
        self.stn = outfun(self.stnp)

        # numerical integration of gpe activity
        self.gpep += (self.dt/self.decay[self.l_gpe])*(
                - self.gpep
                + self.bl_gpe
                + self.noise(self.noise_bl_gpe, self.noise_sd_gpe, self.N )
                - self.w_sd22gpe*self.sd2
                + np.dot(self.stn_fields*\
                        self.w_stn2gpe,self.stn)
                )
        self.gpe = outfun(self.gpep)

        # numerical integration of gpi activity
        self.gpip += (self.dt/self.decay[self.l_gpi])*(
                - self.gpip
                + self.bl_gpi
                + self.noise(self.noise_bl_gpi, self.noise_sd_gpi, self.N )
                - self.w_sd12gpi*self.sd1
                + np.dot(self.stn_fields*\
                        self.w_stn2gpi,self.stn)
                - self.w_gpe2gpi*self.gpe
                )
        self.gpi = outfun(self.gpip)


###################################################################################################################################
## CORTICAL  MODULE ###############################################################################################################
###################################################################################################################################
class Layer :
    """ 
    
    This class defines a simplified representation of a cortical layer

    @brief cortical layer module 
    
    """
    
    ###############################################################################################################################
    def __init__(self, name,  dt=0.001, stime=1, tau=0.005, N=3, th=0, slope=1) :
        """ 
        @brief constructor 

        @param  name    Name. A string defining the object to name the file on which data are written. 
        @param  dt      Delta t. A double value defining the integration time-step for the numerical integration of leaky integrators.     
        @param  stime   Simulation time. An integer value defining the duration of the storage window in time-steps.
        @param  tau     Decay of the leaky. A double value defining the decaying time of leaky integrators. 
        @param  N       Number of channels in the module. An integer value.
        @param  th      Threshold of the tanh output function. It defines an input threshold above which the output is not zero. 
        @param  slope   Slope of the tanh output function. It defines how fast the output reach its asymptote as far the input grows.
        """

        
        #################################################
        ## PARAMETERS ################################### 
        #################################################
        self.dt = dt     # integration step
        self.stime = int(stime/self.dt)     # interval of data storage 
        self.decay = tau     # decay of units
        self.N = N     # number of cortical units 

        #################################################
        ## Initializations ############################## 
        ################################################# 
        self.o = np.zeros(self.N)     # Activations
        self.p = np.zeros(self.N)    # Potentials
        self.data = np.zeros([self.N,self.stime])    # storage matrices
        self.name = name     # name of the class object   
        self.th = th
        self.slope = slope

    ###############################################################################################################################
    def reset_data(self):
        """ 
        @brief Reset stores 
        """

        self.data = self.data*0

    ###############################################################################################################################
    def store(self,timestep, potentials=False):
        """
        @brief Store activity 
       
        @param  timestep    Current timestep
        """
        
        windowtimestep = timestep%self.stime
        if not potentials :
            self.data[:,windowtimestep] = self.o
        else :
            self.data[:,windowtimestep] = self.p

    ###############################################################################################################################
    def reset(self):  
        """ 
        @brief Reset activations 
        """
        
        self.p = np.zeros(self.N)
        self.o = np.zeros(self.N)

    ###############################################################################################################################
    def step(self,inp) :
        """   
        A step of unmerical integration. The layer is updated using euler integration:

        the leaky integrator        tau*dy/dt = -y + input + W*f(y) 
        recomes                     y(t) = y(t-1) + (dt/tau) * (-y(t-1) + input +  W*f(y))
        
        @ brief A single spreading step 
        
        @param  inp Input. A vector of length N defining the input.
        """
        
        self.p += (self.dt/self.decay)*(
                - self.p 
                + inp
                )
        self.o = outfun(self.p,
                th=self.th,slope=self.slope)


if __name__ == "__main__" :

    bg = BG("bg", rng=rng(5))

    for t in range(bg.stime) :

        bg.step(np.random.rand(bg.N),np.random.rand(bg.N),np.random.rand(bg.N),np.random.rand(bg.N))
        bg.store(t)


