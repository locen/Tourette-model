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
## CORTICAL  MODULE ###############################################################################################################
###################################################################################################################################
class OnsetLayer :
    """ 
    
    This class defines a simplified representation of a cortical layer

    @brief cortical layer module 
    
    """
    
    ###############################################################################################################################
    def __init__(self, name,  dt=0.001, stime=1, tau=0.005, itau=0.005, N=3, th=0, slope=1) :
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
        self.idecay = itau     # decay of inner units
        self.N = N     # number of cortical units 

        #################################################
        ## Initializations ############################## 
        ################################################# 
        self.o = np.zeros(self.N)     # Activations
        self.p = np.zeros(self.N)    # Potentials
        self.pi = np.zeros(self.N)    # inner potentials
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
            self.data[:,windowtimestep] = (self.p + self.pi)

    ###############################################################################################################################
    def reset(self):  
        """ 
        @brief Reset activations 
        """
        
        self.p = np.zeros(self.N)
        self.pi = np.zeros(self.N)
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
        
        self.pi += (self.dt/self.idecay)*(
                - self.pi 
                + inp
                )
        self.p += (self.dt/self.decay)*(
                - self.p 
                + np.maximum(0, inp - self.pi)
                )

        self.o = outfun(self.p,
                th=self.th,slope=self.slope)

