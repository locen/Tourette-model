#!/usr/bin/python 

import os
import numpy as np
from numpy.random import RandomState as RNG

def getAct(F=1,slope=1,pot=1,offset=0.0): 
    """ 
    This function computes the activation of a neuron through a sigmoid function

    @brief sigmoid function

    @param F      Maximum rate. It is a scalar. 
    @param slope  Slope of the sigmoid. It defines how fast the output reach its asymptote as far the input grows.
    @param pot    Potential of the neuron.
    @param offset Offset.
    """

    r = F/(1+np.exp(-slope*(pot-offset)))
    return r

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

class Cerebellum:
    """    
    This class defines a cerebellum module consisting in six elements: 
            mf -> mossy fibers
            go -> Golgi cell
            gc -> granule cells
            pf -> parallel fibers
            pc -> Purkinje cells
            ip -> interpositus cells
            
    @brief A class defining a cerebellum module
    """

    ###############################################################################################################################
    def __init__(self, 
            name,  
            dt=1.0, 
            stime=1,
            N         = 3,  # number of neurons
            MF_N      = 20,  # number of mossy fibers
            GC_N      = 100,  # number of Granule cells
            GO_N      = 1,  # number of Golgi cells
            TAU       = 5,  # time constant
            MF_SPARSE = 0.1,  # prop of selected mossy fibers
            PC_BL     = 0.2, # baseline Purkinje cells
            DN_BL     = 0.5,  # baseline Dentate cells
            ETA       = 0.005,   # LTD learning rate
            rng       = RNG(1) # random number generator 
            ) :
        """ 
        @brief constructor 

        @param  name        Name. A string defining the object to name the file on which data are written. 
        @param  dt          Delta t. A double value defining the integration time-step for the numerical integration of leaky integrators.     
        @param  stime       Simulation time. An integer value defining the duration of the storage window in time-steps.
        @param  N           Number of neurons
        @param  MF_N        Number of mossy fibers
        @param  GC_N        Number of Granule cells
        @param  GO_N        Number of Golgi cells
        @param  TAU         Time constant
        @param  MF_SPARSE   Prop of selected mossy fibers
        @param  PC_BL       Baseline Purkinje cells
        @param  DN_BL       Baseline Dentate cells
        @param  ETA         LTD learning rate
        @param  rng         Random number generator 
 
        """
        
        self.rng = rng

        ##################################################
        ## PARAMETERS ####################################
        ##################################################
        self.dt = dt     # integration step
        self.stime = int(stime/self.dt)     # interval of data storage
        
        self.inp_N   = N # number of inputs
        
        # Mossy fibers (MF)
        self.mf_N   = self.inp_N # number of mossy fibers
        self.mf_tau = TAU  # time constant mossy fibers
        self.mf_sparse  = MF_SPARSE  # prop of selected mossy fibers
        
        # Granule cells (GC)
        self.gc_N   = GC_N # number of granule cells
        self.gc_tau = TAU*2 # time constant granule cells
        
        # Golgi cells (GO)
        self.go_N   = GO_N   # number of Golgi cells
        self.go_tau = TAU # time constant Golgi cells

        # Purkinje cells (PC)
        self.pc_N   = N  # number of Purkinje cells
        self.bl_pc = PC_BL # baseline Purkinje cells
                
        self.pc_tau = TAU*2 # time constant Purkinje cells

        # Dentate nuclei cells (DN)
        self.dn_N   = N  # number of Dentate cells
        self.bl_dn = DN_BL # baseline Dentate cells
        self.dn_tau = TAU*2 # time constant Purkinje cells

        self.ETA = ETA

        #################################################
        ## WEIGHTS ###################################### 
        #################################################
        
        
        # Mossy fibers (MF)
        self.w_inp2mf = self.rng.uniform(0.0,2.0, self.inp_N*self.mf_N)
        self.w_inp2mf = self.w_inp2mf.reshape(self.mf_N, self.inp_N)

        # Granule cells (GC)
        self.w_mf2gc = self.rng.uniform(0.4,0.6, self.gc_N*self.mf_N)
        self.w_mf2gc *= self.rng.uniform(0,1, self.gc_N*self.mf_N ) < self.mf_sparse
        self.w_mf2gc = self.w_mf2gc.reshape(self.gc_N, self.mf_N)
        
        # Golgi cell (GO)
        self.w_go2gc = self.rng.uniform(0.8,1.2,self.gc_N*self.go_N).reshape(self.gc_N, self.go_N)
        self.w_gc2go = self.rng.uniform(0.01,0.03,self.go_N*self.gc_N).reshape(self.go_N, self.gc_N)
        self.w_mf2go = self.rng.uniform(0.1,0.3,self.go_N*self.mf_N).reshape(self.go_N, self.mf_N)


        # Purkinje cell (PC)
        self.w_gc2pc = self.rng.uniform(-0.01, 0.01, self.pc_N*self.gc_N)
        self.w_gc2pc = self.w_gc2pc.reshape(self.pc_N, self.gc_N) 
        if os.path.exists("crb_weights") :
            self.w_gc2pc = np.loadtxt("crb_weights")
        
        # Dentate cell (DN)
        self.w_pc2dn = np.ones(self.dn_N)
        self.w_mf2dn = self.rng.uniform(0.01,0.02,self.dn_N*self.mf_N).reshape(self.dn_N, self.mf_N)

        ################################################# 
        ## Initializations ############################## 
        ################################################# 

        # Mossy fibers (MF)
        self.mf_pot         = np.zeros(self.mf_N) # mossy fibers potential
        self.mf_MF          = np.zeros(self.mf_N) # mossy fibers activation

        # Granule fibers (GC)
        self.gc_GC  = np.zeros(self.gc_N)  # granule cells activation
        self.gc_pot = np.zeros(self.gc_N)  # granule cells potential 
        
        #Golgi cells (GO)
        self.go_GO  = np.zeros(self.go_N)  # Golgi cells activation
        self.go_pot = np.zeros(self.go_N)  # Golgi cells potential
        
        # Purkinje cell (PC)
        self.pc_PC  = np.zeros(self.pc_N)  # Purkinje cells activation
        self.pc_pot = np.zeros(self.pc_N)  # Purkinje cells potential
        
        # Dentate nuclei cells (DN)
        self.dn_DN  = np.zeros(self.dn_N)  # Dentate cells activation
        self.dn_pot = np.zeros(self.dn_N)  # Dentate cells potential
        self.o  = self.dn_DN
        
        # Test input learning
        self.des_input = np.zeros(self.pc_N)
        ###############################################################################################################################

        ( self.lpmf, self.lamf, 
            self.lpgc, self.lagc,
            self.lpgo, self.lago,
            self.lppc, self.lapc,
            self.lpdn, self.ladn  ) = range(10) 
        
        self.data = dict()
        self.data[self.lpmf] = np.zeros([self.mf_N, self.stime])
        self.data[self.lamf] = np.zeros([self.mf_N, self.stime])
        self.data[self.lpgc] = np.zeros([self.gc_N, self.stime])
        self.data[self.lagc] = np.zeros([self.gc_N, self.stime])
        self.data[self.lpgo] = np.zeros([self.go_N, self.stime])
        self.data[self.lago] = np.zeros([self.go_N, self.stime])
        self.data[self.lppc] = np.zeros([self.pc_N, self.stime])
        self.data[self.lapc] = np.zeros([self.pc_N, self.stime])
        self.data[self.lpdn] = np.zeros([self.dn_N, self.stime])
        self.data[self.ladn] = np.zeros([self.dn_N, self.stime])

    def store(self, t, potentials=False ):
        
        tt = t%self.stime
        self.data[self.lamf][:,tt] = self.mf_MF  
        self.data[self.lagc][:,tt] = self.gc_GC
        self.data[self.lago][:,tt] = self.go_GO
        self.data[self.lapc][:,tt] = self.pc_PC
        self.data[self.ladn][:,tt] = self.dn_DN

        if potentials :
            self.data[self.lpmf][:,tt] = self.mf_pot
            self.data[self.lpgc][:,tt] = self.gc_pot
            self.data[self.lpgo][:,tt] = self.go_pot
            self.data[self.lppc][:,tt] = self.pc_pot
            self.data[self.lpdn][:,tt] = self.dn_pot

    def reset(self) :

        # Mossy fibers (MF)
        self.mf_pot = np.zeros(self.mf_N) # mossy fibers potential
        self.mf_MF  = np.zeros(self.mf_N) # mossy fibers activation

        # Granule fibers (GC)
        self.gc_GC  = np.zeros(self.gc_N)  # granule cells activation
        self.gc_pot = np.zeros(self.gc_N)  # granule cells potential 
        
        #Golgi cells (GO)
        self.go_GO = np.zeros(self.go_N)  # Golgi cells activation
        self.go_pot = np.zeros(self.go_N)  # Golgi cells potential
        
        # Purkinje cell (PC)
        self.pc_PC  = np.zeros(self.pc_N)  # Purkinje cells activation
        self.pc_pot = np.zeros(self.pc_N)  # Purkinje cells potential
        
        self.dn_DN  = np.zeros(self.dn_N)  # Dentate cells activation
        self.dn_pot = np.zeros(self.dn_N)  # Dentate cells potential
        self.o  = self.dn_DN
        
        # Test input learning
        self.des_input = np.zeros(self.pc_N)

    def reset_data(self) :
        
        for k in self.data :
            self.data[k] = self.data[k]*0    # reset each array to zero

    def step(self, inp):

        """
        A step of numerical integration. Each layer is updated using euler integration:

        the leaky integrator        tau*dy/dt = -y + input + W*f(y) 
        becomes                     y(t) = y(t-1) + (dt/tau) * (-y(t-1) + input +  W*f(y))
        
        @ brief A single spreading step 
        
        @param  inp  input to the granule cells through the mossy fibers.
        """
         
        # Mossy fibers
        self.mf_pot += (self.dt/self.mf_tau)*(
                        - self.mf_pot
                        + np.dot(self.w_inp2mf, np.maximum(0,inp-2.0) )
                        )
        self.mf_MF = outfun(self.mf_pot, 0.0, 1.0)
        
        # TODO
        self.mf_MF = inp-2.0
        
        # Granule cells
        self.gc_pot += (self.dt/self.gc_tau)*(
                        - self.gc_pot
                        + np.dot(self.w_mf2gc,self.mf_MF)
                        - np.dot(self.w_go2gc,self.go_GO)
                        )
        self.gc_GC = outfun(self.gc_pot, 0.0, 2)


        # Golgi cells
        self.go_pot += (self.dt/self.go_tau)*(
                        - self.go_pot
                        + np.dot(self.w_gc2go, self.gc_GC)
                        + np.dot(self.w_mf2go, self.mf_MF)
                        )
        self.go_GO = outfun(self.go_pot, 0, 0.3)

        # Purkinje cells
        self.pc_pot += (self.dt/self.pc_tau)*(
                        - self.pc_pot
                        + np.dot(np.ones(self.pc_N), np.maximum(0,inp - .5))/self.pc_N
                        + np.dot(self.w_gc2pc, self.gc_GC)
                        )	
        self.pc_PC = outfun(self.pc_pot, 0, 2.0)
    
        # Dentate cells
        self.dn_pot += (self.dt/self.dn_tau)*(
                        - self.dn_pot
                        + self.bl_dn
                        - self.w_pc2dn*self.pc_PC
                        )	
        self.dn_DN = outfun(self.dn_pot, 0, 1)
        self.o  = self.dn_DN

    def ltd(self, teach):
        """
        LTD learning

        @param  teach       Teaching signal
        """

        eta = self.ETA
        x = self.gc_GC 
        y = teach
        k = self.pc_PC
        w = self.w_gc2pc

        w -=  eta * np.outer( y-k, x ) 


