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

def scale(param, min_val, max_val):
    '''
    Scales a parameter value from the standard interval [0;1]
    to the real interval [min_val;max_val]

    :param  param       the current parameter value
    :param  min_val     the lower bound of the unscaled range
    :param  max_val     the upper bound of the unscaled range
    '''
    return param*(max_val - min_val) + min_val


class ParameterManager:
    '''
    Manages te parameters of the simulations.

    Reads from the file 'ga_parameters'

    '''
    
    NUM_PARAMS = 32
    
    def __init__(self):

        self.ga_params = dict()
        self.main_params = dict()
        self.bg_params = dict()
        self.set_default_params() 
       
        self.set_par_list = dict()
        self.par_list = [] 

        # read parameters and ranges from ga_parameters
        with open('ga_parameters', 'r') as parfile:
            lines = parfile.readlines()
            for line in lines[2:]:
                if line.strip() :
                    (par, low, high, val, ga) = line.split()
                    low = float(low)
                    high = float(high)
                    val = float(val)
                    ga = ga == "True"
                    self.ga_params[par] = ga
                    if ga == True :
                        self.init_parameter(par, low, high)
                    else :
                        self.set_parameter(par, val)

        assert(len(self.par_list) == self.NUM_PARAMS)
    
    def init_parameter(self, par, pmin, pmax):
        '''
        Prepare the list of parameters to be taken from the Genetic Algorithm

        :param  par         a string defining the name of the parameter
        :param  pmin        the lower bound of the unscaled range
        :param  pmax        the upper bound of the unscaled range
        '''
        
        self.par_list.append(par)
        self.set_par_list[par] = [pmin, pmax]
        
    def set_parameter(self, par, value) :
           '''
            Prepare the list of fixed parameters

            :param  par         a string defining the name of the parameter
            :param  value       the value of hte parameter
            '''
         
            if par in self.main_params.keys():
                self.main_params[par] = value 
            elif par in self.bg_params.keys():
                self.bg_params[par] = value


    def set_parameters(self, params) :
        '''
        Set all parameters to the fixed values

        :param  params  list of all parameters
        '''

        assert(len(params)==self.NUM_PARAMS)

        param = iter(params)

        for par in self.par_list:
            
            lims = self.set_par_list[par]
            self.set_parameter(par, scale(param.next(), lims[0], lims[1] ) )


    def get_all_params(self):
        '''
        Create a dictionary with all parameters
        '''

        all_pars = dict()
        all_pars.update(self.main_params)
        all_pars.update(self.bg_params)
        return all_pars

    def get_main_params(self):
        '''
        Get parameters of the main system
        '''

        return self.main_params
    
    def get_bg_params(self):
        '''
        Get basal-ganglia parameters
        '''
        return self.bg_params

    def set_default_params(self):
        '''
        Set default parameters (in case there is no ga_parameters file)
        '''
        
        self.main_params.setdefault(            'TRIAL_LENGTH', 808                )
        self.main_params.setdefault(            'TIC_INTERVAL', 1                  )
        self.main_params.setdefault(                      'DT', 0.001              )
        self.main_params.setdefault(                     'TAU', 0.01               )
        self.main_params.setdefault(                       'N', 3                  )
        
        self.main_params.setdefault(                'w_bg2tha', 850.0              )
        self.main_params.setdefault(               'w_tha2crx', 1.0                )
        self.main_params.setdefault(              'w_crx2thal', 10.0               )
        self.main_params.setdefault(                'w_tha2bg', 0.1                )
        self.main_params.setdefault(               'w_crb2tha', 0.01               )
        self.main_params.setdefault(            'w_crb2crbtha', 0.01               )
        self.main_params.setdefault(                'w_bg2crb', 0.2                )
        self.main_params.setdefault(               'w_crx2crb', 0.5                )
        self.main_params.setdefault(           'w_crx2crbthal', 0.2                )
        self.main_params.setdefault(            'w_crbtha2crx', 0.1                )
        self.main_params.setdefault(             'bl_thalamus', 0.0                )
        self.main_params.setdefault(               'w_inp2da1', 1.9                )
        self.main_params.setdefault(               'w_inp2da2', 1.8                )
        self.main_params.setdefault(                  'th_crx', 0.4                )

        self.main_params.setdefault(                  'da_min', .01                )
        self.main_params.setdefault(           'inp_noise_std', 0.0                )
        self.main_params.setdefault(            'noise_bl_inp', 0.0                )
        self.main_params.setdefault(       'crx_inp_noise_std', 1.9                )
        self.main_params.setdefault(        'noise_bl_crx_inp', -1.16              )
        self.main_params.setdefault(       'crb_inp_noise_std', 1.2                )
        self.main_params.setdefault(        'noise_bl_crb_inp', 0.4                )
        self.main_params.setdefault(       'noise_sd_thalamus', 0.01               )
        self.main_params.setdefault(       'noise_bl_thalamus', 0.0                )
        self.main_params.setdefault(    'noise_sd_crbthalamus', 0.01               )
        self.main_params.setdefault(    'noise_bl_crbthalamus', 0.0                )
        self.main_params.setdefault(           'tic_threshold', 0.4                )
        self.main_params.setdefault(            'target_areas', None               )

        self.main_params.setdefault(            'tau_amp_thal', .2                 ) 
        self.main_params.setdefault(         'tau_amp_crbthal', .2                 ) 
        self.main_params.setdefault(             'tau_amp_crx', .8                 ) 
        self.main_params.setdefault(             'tau_amp_crb', .8                 ) 

        self.main_params.setdefault(             'tau_amp_sd1', 0.3                ) 
        self.main_params.setdefault(             'tau_amp_sd2', 0.3                ) 
        self.main_params.setdefault(             'tau_amp_stn', 0.3                ) 
        self.main_params.setdefault(             'tau_amp_gpe', 1.2                ) 
        self.main_params.setdefault(             'tau_amp_gpi', 2.8                ) 
        self.main_params.setdefault(              'tau_amp_d1', 0.3                ) 
        self.main_params.setdefault(              'tau_amp_d2', .6                 ) 

        self.bg_params.setdefault(                 'w_cri2sd1', .5                 )
        self.bg_params.setdefault(                 'w_cri2sd2', .5                 )
        self.bg_params.setdefault(                 'w_cri2stn', 100.0              )
        self.bg_params.setdefault(                 'w_cro2sd1', 2.6                )
        self.bg_params.setdefault(                 'w_cro2sd2', 2.6                )
        self.bg_params.setdefault(                 'w_cro2stn', 2.2                )
        self.bg_params.setdefault(                 'w_sd12gpi', 20.0               )
        self.bg_params.setdefault(                 'w_stn2gpi', 0.1                )
        self.bg_params.setdefault(                 'w_sd22gpe', 6.9                )
        self.bg_params.setdefault(                 'w_stn2gpe', .3                 )
        self.bg_params.setdefault(                 'w_gpi2gpe', 0                  )
        self.bg_params.setdefault(                 'w_gpe2gpi', 0.1                )
        self.bg_params.setdefault(                 'w_gpe2stn', 0.2                )
        self.bg_params.setdefault(                    'bl_sd1', 0.01               )
        self.bg_params.setdefault(                    'da_sd1', 2800               )
        self.bg_params.setdefault(                 'da_th_sd1', 0.1                )
        self.bg_params.setdefault(                    'bl_sd2', 0.1                )
        self.bg_params.setdefault(                    'da_sd2', 5                  )
        self.bg_params.setdefault(                    'bl_gpe', 1.4                )
        self.bg_params.setdefault(                    'bl_gpi', 1.2                )
        self.bg_params.setdefault(                   'blm_sd1', 0.1                )
        self.bg_params.setdefault(                   'blm_sd2', 0.1                )
        self.bg_params.setdefault(              'noise_sd_sd1', .2                 )
        self.bg_params.setdefault(              'noise_sd_sd2', .2                 )
        self.bg_params.setdefault(              'noise_sd_stn', .01                )
        self.bg_params.setdefault(              'noise_sd_gpi', 1.5                 )
        self.bg_params.setdefault(              'noise_sd_gpe', 0.6                 )
        
