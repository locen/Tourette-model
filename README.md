# Tourette-model
Authors: Daniele Caligiore, Francesco Mannella, Michael A. Arbib and Gianluca Baldassarre

```
├── README.md
└── src
    ├── bg.py                   # Basal-Ganglia model
    ├── cerebellum.py           # Cerebellum model
    ├── tics_simulation.py      # the simulation 
    ├── parameter_manager.py    # manages the parameter of simulations
    ├
    ├── ga_search.py            # run simulations as individuals within a genetic algorithm
    ├── ga_test.py              # run test on the results of a genetic algorithm run with ga_search.py
    ├
    └── plot_utils.py           # utility plots (used by ga_test.py)

```

***ga_search.py***

Run genetic algorithms in parallel via mpi

example: **mpirun -n 8 python ga_search.py -g 100 -p 100 -t 1.0**

runs a genetic algorithm with populations of 100 individuals
for 100 generations. The simulation time maximum limit is 01:00 hour 

```
usage:ga_search.py [-h] [-g NGEN] [-p NPOP] [-s SIGMA] [-m MUTATEPB] [-c MATEPB]
                   [-M MUTATEPB] [-C MATEPB] [-i INITTYPE] [-r SCORETYPE] [-v]
                   [-t EXECTIME]

optional arguments:
  -h, --help            show this help message and exit
  -g NGEN, --ngen NGEN  Number of generations
  -p NPOP, --npop NPOP  Number of individual in the population
  -s SIGMA, --sigma SIGMA
                        Standard deviation of the gaussian mutation
  -m MUTATEPB, --mutatepb MUTATEPB
                        Probability of mutation of a specific parameter
  -c MATEPB, --matepb MATEPB
                        Probability of crossover of a specific parameter
  -M MUTATEPB, --MUTATEPB MUTATEPB
                        Probability of mutation of an individual
  -C MATEPB, --MATEPB MATEPB
                        Probability of crossover of an individual
  -i INITTYPE, --inittype INITTYPE
                        type of population initialization: 0: RANDOM, 1: GRID
  -r SCORETYPE, --scoretype SCORETYPE
                        type of score computation: 1: SIMPLE, 0: ANALYTIC
  -v, --savepotentials  Save potentials (default save activations)
  -t EXECTIME, --exectime EXECTIME
                        execution time (hours)
```

***ga_test.py***

Shows the results of a genetic algorithm.
Produces: 

1. Plots of the mean firing rates of the best simulation per area
2. Plot of the history of best scores of the GA (scoresXgenerations)
3. Print the chosen parameters (actual scales)
4. ASCII histogram of the parameters (scaled) of the individuals below the score threshold (over all generations)
5. Plot histogram of the parameters (scaled) of the individuals below the score threshold (over all generations)
6. ASCII histogram of the parameters (scaled) of the last generation individuals below the score threshold
7. ASCII histogram of the parameters (scaled) of the last generation individuals 
8. Parameter ranges built from the current parameters, with limits built uing the standard deviations of the last population scaled times a value given in the command line (see below)
9. Plot hystory of scores


example: **python ga_test.py -r 0.5**

use (standard deviations)*0.5 in (8)

```

usage:ga_test.py [-r] [--range RSCALE]

optional arguments:
  -h, --help                 show this help message and exit
  -r RSCALE, --range RSCALE  scaling of parameters' standard deviatiations
                   to  build range limits
```
