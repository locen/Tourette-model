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
import glob
import argparse

import numpy as np
import pandas as pd
from numpy.random import RandomState as RNG

import tics_simulation
from tics_simulation import LesionTypes 
from tics_simulation import ScoreTypes 
from tics_simulation import TicsSimulation as Sim
from parameter_manager import ParameterManager as PM
from parameter_manager import scale
np.set_printoptions(suppress=True, precision=5, linewidth=9999999)

Sim.SAVE_POTENTIALS=False
Sim.SCORE = ScoreTypes.ANALYTIC



# ARGUMENT PARSING -------------------------------------

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-r','--range',
        help="Range of output limits",
        action="store", default=0.2)

parser.add_argument('-t','--threshold',
        help="Score threshold of the best individuals",
        action="store", default=0.08)

parser.add_argument('-i','--interactive',
        help="plot online",
        action="store_true", default=False)

args = parser.parse_args()    
LIMITS  =  float(args.range)
THRESHOLD  =  float(args.threshold)
INTERACTIVE  =  float(args.interactive)


# manage matplotlib interactivity -----------------------

if not INTERACTIVE :
    import matplotlib
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
from plot_demo import plot_raw_layers
from plot_demo import plot_means

if  INTERACTIVE :
    plt.ion()
#--------------------------------------------------------



def test (individual_seed, params, data):
    """
    :param  individual_seed     the seed of the test
    :param  params              the list of parameters
    :param  data                the target data to compute 
                                the objective
    """

    pm = PM()
    
    DA_VALUE = .5
    pm.set_parameters( params ) 

    rng = RNG(individual_seed) 
   
    # init the simulation
    sim = Sim(
            rng = rng,
            SEED = individual_seed,
            DA_VALUE = DA_VALUE, 
            TRIALS = 30, 
            LESION = LesionTypes.NOLESION, 
            pm=pm)

    # run the simulation
    sim.simulate()
    
    score = sim.score(data)
    
    return sim, pm, score

#-----------------------------------------------------------------------------------
# load data about all populations and seeds
# build the pandas dataframe
# query for the best scores x generation

n_par = PM.NUM_PARAMS

print "load populations..."
popfiles = glob.glob('population_*')
gen_data=[]
for popfile in popfiles:
    print "    loading {}...".format(popfile)
    
    popdata = np.loadtxt(popfile)
    gen_data.append(popdata)
gen_data = np.vstack(gen_data)

print "load seeds..."
seeds = np.hstack((np.loadtxt("all_seeds"),)).astype(int)

print "make dataframe..."
gen_data_df = pd.DataFrame(
        gen_data, 
        columns= np.hstack(( 
            'gen',
            'score',
            ['p{:03d}'.format(x) for x in xrange(n_par)] 
            )) )

print "make bests dataframe..."
bests = gen_data_df.groupby('gen', as_index=False)
bests = bests.apply(lambda g: g[g['score'] == np.min(g['score']) ].iloc[0,:] )

#-----------------------------------------------------------------------------------
# query for the overall best score and save it to csv
# get parameters of the best score
# load target data

print """



        1) Plots of the mean firing rates of the best simulation per area
        2) Print the best score ans seed and its corresponding parameters (real scale)

"""

last = bests[bests.score == np.min(bests.score)]
last.to_csv("ga_best", sep='\t', encoding='utf-8')

params = last.iloc[0,2:].as_matrix()
data = tics_simulation.load_target_data()


#-----------------------------------------------------------------------------------
# iterate over seeds and do the tests with the optimized parameters

frames =[]
for seed in seeds:
    
    sim, pm, score = test(seed, params, data)
    
    frames.append(sim.df)


    plot_means(sim)
    if not INTERACTIVE: plt.savefig("means.png", dpi=400)
    plot_raw_layers(sim)
    if not INTERACTIVE: plt.savefig("raw_layers.png", dpi=400)
    
    print
    print "{:>20} : {:=10.4f}".format('score',float(score))
    print "{:>20} : {:<10d}".format( 'seed',seed)
    print
    for key,value in pm.get_all_params().items():
        if value is not None:
            print "{:>20} : {:=10.2f}".format(key,float(value))
    print
    print

#-----------------------------------------------------------------------------------
# save data of the seeds
# save target data
# get the ordered list of parameters


allframes = pd.concat(frames)

allframes.to_csv("analysis_data", sep='\t', encoding='utf-8')
data.to_csv("data_target", sep='\t', encoding='utf-8')
        
par_list = np.array(pm.par_list)

raw_input("Press key to go on")

#-----------------------------------------------------------------------------------
# query for all the overthreshold scores over all generations
# print graphs

print """



        3) ASCII histogram of the parameters (scaled) of the individuals below the score threshold (over all generations)
        4) Plot histogram of the parameters (scaled) of the individuals below the score threshold (over all generations)
        5) ASCII histogram of the parameters (scaled) of the last generation individuals below the score threshold

""".format(THRESHOLD)


almostbests = gen_data_df[gen_data_df.score<THRESHOLD]
if almostbests.size == 0 :
    print
    print "NO SCORES < {:8.6f}".format(THRESHOLD)
    print
else :
    print
    bestmeans = almostbests.iloc[:,2:].mean().as_matrix()
    beststds = almostbests.iloc[:,2:].std().as_matrix()
    idcs = np.argsort(beststds)
    
    print
    for i in idcs:
        print "{:20} ".format(par_list[i])+(("@")*int((100*bestmeans[i]))) + \
                (("-")*int((100*beststds[i])))+"|"
    print

    fig = plt.figure("parameters")
    ax = fig.add_subplot(111)
    x = np.arange(len(bestmeans))

    idcs = np.argsort(beststds)
    bars = ax.bar(0,.8, bestmeans[idcs], x, xerr=beststds[idcs], 
            color="#ff0000", ecolor="black",align="center", orientation="horizontal")
    for b,std in zip(bars.get_children(), beststds[idcs]):
        b.set_alpha(.1+np.max(beststds) - std)
    ax.set_ylim([-2,pm.NUM_PARAMS+1])
    ax.set_yticks( range(pm.NUM_PARAMS) )
    ax.set_yticklabels(par_list[idcs])
    
    if not INTERACTIVE: plt.savefig("parameters.png", dpi=400)

raw_input("Press key to go on")

#-----------------------------------------------------------------------------------
# query for all the  scores of the last generation 
# print graphs


print """



        6) ASCII histogram of the parameters (scaled) of the last generation individuals 

"""
lasts = gen_data_df[gen_data_df.gen==np.max(gen_data_df.gen)]
lastmeans = lasts.iloc[:,2:].mean().as_matrix()
laststds = lasts.iloc[:,2:].std().as_matrix()

idcs = np.argsort(laststds)
print
for i in idcs:
    print "{:20} ".format(par_list[i])+(("@")*int((100*lastmeans[i]))) + \
            (("-")*int((100*laststds[i])))+"|"
print
print

raw_input("Press key to go on")

#-----------------------------------------------------------------------------------
# compute ranges of the last generation 

print """



        7) Print ranges around best parameters 



"""
print ">- copy to ga_parameters ------------------------------------------------"
print 
for l,mn,std in zip(par_list, lastmeans, laststds):
    lims = pm.set_par_list[l]
    mn = scale(mn, lims[0], lims[1] )
    std = scale(std, lims[0], lims[1] )
    print "{:<20}  {:>10.4f} {:>10.4f} {:>10.4f} True".format(l, mn-std*LIMITS, mn+std*LIMITS, mn)
print 
print ">------------------------------------------------------------------------"
print 
print 

raw_input("Press key to go on")

#-----------------------------------------------------------------------------------
# compute fitness graph (best scores x generation)  

print """



        8) Plot of the history of best scores of the GA (scoresXgenerations)

"""


fig = plt.figure("fitness")
ax = fig.add_subplot(111)
ax.plot( bests.score.as_matrix() )
print
if not INTERACTIVE: plt.savefig("fitness.png", dpi=400)

raw_input("Press key to end")





