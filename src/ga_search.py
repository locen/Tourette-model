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
import sys
import random
import time 
import numpy as np
from numpy.random import RandomState as RNG
from deap import creator, base, tools, algorithms
from ga_popinit import GeneratorTypes as GTypes
from ga_popinit import PopGenerator as PopGenerator

import tics_simulation
from tics_simulation import LesionTypes 
from tics_simulation import ScoreTypes 
from tics_simulation import TicsSimulation as Sim
from parameter_manager import ParameterManager as PM
from itertools import repeat
from collections import Sequence
#import multiprocessing
import mpipool.core 
import argparse

# manage a Genetic Algorithm using deap to find the parameters of the model


# enum fro the types of 
# objective functions

class ObjTypes :
    SIMPLE = 0
    COMPLEX = 1

# a functor class managing objective 
# functions of the GA
class Objective:

    OUTLIER_SCORE = 100.0
    N_TRIALS = 30

    def __init__(self, obj_type=ObjTypes.SIMPLE):

        self.OBJ_TYPE = obj_type

        # parameter manager
        self.pm = PM()
            
        # target data
        self.des_data = tics_simulation.load_target_data()
        
        # prepare seeds
        self.seed_n = 1
        self.seeds =  [np.fromstring(os.urandom(4), dtype=np.uint32)[0] 
                for x in xrange(self.seed_n) ]

    def __call__(self, params):
         
        # scale parameters
        self.pm.set_parameters( params ) 
        
        #----------------------------------------------------

        ##### 1) external stimulus
        #####    
        #####    * for each seed run a simulation with da = 0.5 and fixed external stimulus
        #####    * the score is the fitting of activation means with those in the paper
        
        DA_VALUE = .5
         
        scores = []

        for individual_seed in self.seeds:

            rng = RNG(individual_seed) 

            # init the simulation
    
            sim = Sim(
                    rng = rng,
                    SEED = individual_seed,
                    DA_VALUE = DA_VALUE, 
                    TRIALS = self.N_TRIALS, 
                    LESION = LesionTypes.NOLESION, 
                    RANDOM_EXTERNAL=False,
                    pm=self.pm)

            # run the simulation
            sim.simulate()

            # compute fitting
            score = sim.score(self.des_data)

            # correct score in case of nan
            if not np.isnan(score) :
                scores.append(score)
            else:
                scores.append(self.OUTLIER_SCORE)

        # the actual score is the mean over all individuals
        es_score = np.mean(scores)

        if self.OBJ_TYPE == ObjTypes.COMPLEX :
            #----------------------------------------------------

            ##### 2) external random stimulus
            #####    
            #####    * for each seed run 3 simulations with da = 0.1, 0.5, 0.9 and random external stimulus
            #####    * the score is given by the distance of the three tic proportions with desired 
            #####      proportions (0.1, 0.5, 0.9)
            
            da_values = np.array([.1, .5, .9])

            scores = []
            for individual_seed in self.seeds:

                prop_of_tics = []
                for DA_VALUE in da_values:

                    rng = RNG(individual_seed) 

                    # init the simulation
                    
                    sim = Sim(
                            rng = rng,
                            SEED = individual_seed,
                            DA_VALUE = DA_VALUE, 
                            TRIALS = self.N_TRIALS, 
                            LESION = LesionTypes.NOLESION, 
                            RANDOM_EXTERNAL=True,
                            pm=self.pm)

                    # run the simulation
                    sim.simulate()

                    sim.get_dataframe()

                    # compute tic proportion
                    tics_in_wins = sim.df[['WIN','TIC']].groupby(['WIN','TIC']).head(1)
                    tic_wins = tics_in_wins[tics_in_wins.TIC=="TIC"]
                    if len(tic_wins) > 0:
                        num_of_tics = tic_wins["WIN"].as_matrix()[0]
                    else:
                        num_of_tics = 0

                    # append to the list of tic proportions
                    prop_of_tics.append( num_of_tics/float(self.N_TRIALS) )

                # compute the score as the distance from desired proportions
                scores.append(np.linalg.norm( da_values - prop_of_tics))

            # the actual score is the mean over all individuals
            ers_score = np.mean(scores) 

            #----------------------------------------------------

            # the overall score is weighted by both fitting ad standard da level
            # and tic proportions at multiple da levels
            res = (es_score + ers_score)/2.

        else :

            res = es_score
        
        return res,
        

#--------------------------------------------------------------------------------

def boundedMutGaussian(individual, sigma, indpb):
    """
    This function applies a gaussian mutation of mean 0 and standard
    deviation *sigma* on the input individual. This mutation expects a
    :term:`sequence` individual composed of real valued attributes.
    The *indpb* argument is the probability of each attribute to be mutated.
    Results are always bounded within [0,1]
    
    :param individual: Individual to be mutated.
    :param sigma: Standard deviation or :term:`python:sequence` of 
                  standard deviations for the gaussian addition mutation.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    
    This function uses the :func:`~random.random` and :func:`~random.gauss`
    functions from the python base :mod:`random` module.
    """
    size = len(individual)

    if not isinstance(sigma, Sequence):
        sigma = repeat(sigma, size)
    elif len(sigma) < size:
        raise IndexError("sigma must be at least the size of individual: %d < %d" % (len(sigma), size))
    
    for i,  s in zip(xrange(size), sigma):
        if random.random() < indpb:
            individual[i] += random.gauss(0, s)
            individual[i] = np.maximum(0, individual[i])
            individual[i] = np.minimum(1, individual[i])
    
    return individual,

#--------------------------------------------------------------------------------


if __name__ == "__main__" :
    
    # init the pool object to work in parallel using openmpi
    pool = mpipool.core.MPIPool(loadbalance=True)
    
    gen_time = 0.0 # a time counter
    np.set_printoptions(suppress=False, precision=5, linewidth=9999999)

    #----------------------------------------------------------------------------

    # ARGUMENT PARSING ----------------------------------------------------------
    args = None
    comm = pool.comm
    try:
        if comm.Get_rank() == 0:
            parser = argparse.ArgumentParser()
            
            parser.add_argument('-g','--ngen',
                    help="Number of generations",
                    action="store", default=2)

            parser.add_argument('-p','--npop',
                    help="Number of individual in the population",
                    action="store", default=2)
            
            parser.add_argument('-s','--sigma',
                    help="Standard deviation of the gaussian mutation",
                    action="store", default=0.5)

            parser.add_argument('-m','--mutatepb',
                    help="Probability of mutation of a specific parameter",
                    action="store", default=0.05)

            parser.add_argument('-c','--matepb',
                    help="Probability of crossover of a specific parameter",
                    action="store", default=0.05)
            
            parser.add_argument('-M','--MUTATEPB',
                    help="Probability of mutation of an individual",
                    action="store", default=0.1)
            
            parser.add_argument('-C','--MATEPB',
                    help="Probability of crossover of an individual",
                    action="store", default=0.5)
            
            parser.add_argument('-i','--inittype',
                    help="type of population initialization: {}: RANDOM, {}: GRID".format(
                        GTypes.RANDOM, GTypes.GRID ),
                    action="store", default=GTypes.RANDOM)
            
            parser.add_argument('-r','--scoretype',
                    help="type of score computation: {}: SIMPLE, {}: ANALYTIC".format(
                        ScoreTypes.SIMPLE, ScoreTypes.ANALYTIC ),
                    action="store", default=ScoreTypes.SIMPLE)
            
            parser.add_argument('-o','--objtype',
                    help="type of compex score computation: {}: SIMPLE, {}: COMPLEX".format(
                        ObjTypes.SIMPLE, ObjTypes.COMPLEX ),
                    action="store", default=ObjTypes.SIMPLE)
            
            parser.add_argument('-v','--savepotentials',
                    help="Save potentials (default save activations)",
                    action="store_true", default=False)
            
            parser.add_argument('-t','--exectime',
                    help="execution time (hours)",
                    action="store", default=0.5)
            args = parser.parse_args()    
        

    finally:
        args = comm.bcast(args, root=0)
    
    if args is None:
        exit(0)

    NGEN                   =  int(args.ngen)
    NPOP                   =  int(args.npop)
    SIGMA                  =  float(args.sigma) 
    INDMUTATEPB            =  float(args.mutatepb)
    INDMATEPB              =  float(args.matepb)
    MUTPB                  =  float(args.MUTATEPB)
    CXPB                   =  float(args.MATEPB)
    GTYPE                  =  int(args.inittype)
    Sim.SAVE_POTENTIALS    =  bool(args.savepotentials)
    Sim.SCORE              =  int(args.scoretype)
    OBJTYPE                =  int(args.objtype)
    EXECTIME               =  float(args.exectime)


    #----------------------------------------------------------------------------
    
    # instantiate the functor for the objective
    objective = Objective(obj_type=OBJTYPE)
    
    # the master thread saves the seeds
    if pool.is_master() : 
    
        print " saving  all_seeds..."
        np.savetxt("all_seeds", objective.seeds )
        print "   saved"
    
    # init the GA registering all methods

    toolbox = base.Toolbox()
    
    # generate the functions to create individuals (individuals
    # have minimizing fitness)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    # generate the initial population
    generator = PopGenerator(n_pars=PM.NUM_PARAMS, n_ind=NPOP, gen_type=GTYPE)
    toolbox.register("attr_float", generator)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
            toolbox.attr_float, n=PM.NUM_PARAMS)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # here we register our functor as the objective of the GA -------------------
    toolbox.register("evaluate", objective )
    #----------------------------------------------------------------------------

    # other GA parameters
    toolbox.register("mate", tools.cxUniform, indpb=INDMATEPB  )
    toolbox.register("mutate", boundedMutGaussian, sigma=SIGMA, indpb=INDMUTATEPB)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # register pool.map as the map function so that individuals 
    # are run in parallel with openmpi
    toolbox.register("map", pool.map)
    
    # get the population
    population = toolbox.population(n=NPOP)

    # iterate over generations
    for gen in range(NGEN):
        
        # read clock time
        gen_time = time.clock()/3600.0
        
        # print the current generation and time info
        if pool.is_master() :
            print "gen:  {}".format(gen)
            print "hour: {:2.3f}".format(gen_time)
                
        # control if the execution walltime has been exceeded
        if gen_time >= EXECTIME :
            if pool.is_master() :
                print "stopped at hours {:2.3f} from the beginning".format(gen_time/60.0) 
            pool.close()
            break
        
        # run the generation and wait for all threads to finish
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        fits = toolbox.map(toolbox.evaluate, offspring)
        pool.close()
        
        # get the current population
        population = toolbox.select(offspring, k=len(population))
        
        # the master thread gives info about the current population
        if pool.is_master() : 

            
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit
            top = tools.selBest(population, k=1)
    
            print "   score       :{:5.4f} ".format(top[0].fitness.values[0])
            
            print "   saving gen {}...".format(gen)
            final_data= []
            for ind in population:
                ind_data = np.hstack(( gen, ind.fitness.values[0], list(ind) ))
                final_data.append(ind_data)
            final_data = np.vstack(final_data)
    
            np.savetxt("population_{:06d}".format(gen), final_data)
            print "   saved"
            print

    # end of simulation
    if pool.is_master() : 
    
        print "   end of simulation"
    
    
