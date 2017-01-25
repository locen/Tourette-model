#!/usr/bin/env python
import numpy as np
import os
from numpy.random import RandomState as RNG

def popinit(numGenes, numIndividuals=None):
    '''
    Create a matrix of indices so that the rows contain indices of the
    exaustive combinations of the all the 'bin_vals' gene values.
    
    General issue: finding a matrix of M sequences of N elements in which
    K different values are uniformly combined. 
    
    M = fil_n
    N = p
    K = binnum
    
    Algorithm:
    
    a) Create a matrix of N identical columns with numbers 0:(M-1).
    
    b) Iterate over the N columns 
    
        1) Find the period P (number of rows) in which the K values are
            iterated. This P depends on the K range of values and the amount
            of rows in which each value reamins the same (based on the n-th
            index).
    
        2) Convert each n-th column so that it contains a sequence of
            integers increasing every P rows.
    
    c) Convert each n-th column so that indices range from 1 to k.


        :param      numGenes            number of parameters per individual
        :param      numIndividuals      number of individuals in the population

    returns:                                         
                    pop                 the initialized population         
                    reqpop              the minimal required population
                                        length                             
    '''
      
    p = numGenes
    n = numIndividuals

    pop = np.zeros([p, n])
    reqpop = 1
 
    # compute the minimal required number of individuals
    while True:
        if np.floor(np.power(reqpop, 1/float(p)))>1 :
            break 
        reqpop += 1 

    # verify that the individual requested are not too much 
    binnum = np.floor(np.power(n, 1/float(p)))
    binwidth = 1/binnum
    reminder = 1 - binnum*binwidth
    fil_n = int(np.power(binnum, p))
    correct = binnum > 1

    if n is None :
        n = reqpop


    if correct :

        # find the intervals of gene values on a 0:1 scale 
        bin_vals = np.linspace(binwidth/2.0, 1-binwidth/2.0, binnum)
  
        # CORE ALGORITHM
        #
        #----------------------------------------
        # a)
        idcs = np.tile(np.arange(fil_n), [p, 1])
        
        #----------------------------------------
        # b)
        for t in xrange(p):
            # 1)
            dim_idx_period = np.power(binnum, t)  
    
            # 2)
            idcs[t,:] = np.cumsum( (np.mod(idcs[t,:], dim_idx_period) + 1) == 1)
        
        # c) 
        idcs = np.mod(idcs - 1, binnum).astype(int)
  
        for r in xrange(p) :
            for c in xrange(fil_n):
                pop[r, c] = bin_vals[idcs[r, c]]
        
        for r in xrange(p):
            for c in xrange(fil_n,n):  
                pop[r, c] = np.random.rand()

        return pop, reqpop, binnum

    else:  
        return None, reqpop, binnum


#--------------------------------------------------------------------------------

class GeneratorTypes:
    RANDOM = 0 
    GRID = 1 

class PopGenerator:

    def __init__(self, n_pars, n_ind, gen_type=GeneratorTypes.RANDOM):
        
        self.i = 0
        self.n_pars = n_pars
        self.n_ind = n_ind 

        self.seed =  np.fromstring(os.urandom(4), dtype=np.uint32)[0] 
        self.rng = RNG(self.seed) 

        if gen_type==GeneratorTypes.RANDOM:
            self.make_random()
        else:
            self.make_grid()

    def make_iterator(self, population):
        if population is not None:
            self.itr = iter(population.T.ravel())
        else:
            print " "            
            print "POPINIT: "            
            print "POPINIT: "            
            print "POPINIT: POPULATION TOO SMALL FOR {} PARAMETERS!!!".format(self.n_pars)
            print "POPINIT: MINIMUM NUMBER OF INDIVIDUALS: {}".format(self.reqpop)
            print "POPINIT: "
            print "POPINIT: "
            print " "
            
            self.itr = None

    def make_random(self) :
    
        population = self.rng.rand(self.n_pars, self.n_ind)
        self.make_iterator(population)
    
    def make_grid(self) :
    
        population, self.reqpop,_ = popinit(self.n_pars, self.n_ind)
        self.make_iterator(population)

    def __call__(self):

        if self.i >= self.n_pars*self.n_ind :
            return None
            
        self.i+=1
            
        return self.itr.next()

if __name__ == "__main__" :

    import matplotlib.pyplot as plt
    
    p = 2
    n = 200
    gen = PopGenerator(p, n)
    
    d = np.zeros([p, n])
    for col in xrange(n):
        for row in xrange(p):
            d[row, col] = gen()

    plt.figure() 
    plt.scatter(*d)

    gen = PopGenerator(p, n,gen_type=GeneratorTypes.GRID)
    
    d = np.zeros([p, n])
    for col in xrange(n):
        for row in xrange(p):
            d[row, col] = gen()

    plt.figure() 
    plt.scatter(*d)

    plt.show()
