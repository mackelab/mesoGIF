"""
Created Mon May 29 2017

author: Alexandre René
"""

import logging
import os.path
import sys
import time
import copy
import hashlib
import numpy as np
import scipy as sp
import collections
from collections import namedtuple, OrderedDict, Iterable

import parameters

import theano_shim as shim

import fsgif_model as gif

# try:
#     import matplotlib.pyplot as plt
# except ImportError:
#     logging.warning("Unable to import matplotlib. Plotting won't work.")
#     do_plots = False
# else:
#     do_plots = True

############################
# Basic configuration
# Sets logger, default filename and whether to use Theano
############################

#import os
#os.environ['THEANO_FLAGS'] = "compiledir=theano_compile"

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
def _init_logging_handlers():
    # Only attach handlers if running as a script
    import logging.handlers
    fh = logging.handlers.RotatingFileHandler('fsgif_main.log', mode='w', maxBytes=5e5, backupCount=5)
    fh.setLevel(sinn.LoggingLevels.MONITOR)
    fh.setFormatter(sinn.config.logging_formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(sinn.config.logging_formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

def load_theano():
    """
    Run this function to use Theano for computations.
    Currently this is not supported for data generation.
    """
    shim.load_theano()

rndstream = None
stream_seed = None
# Store loaded objects like model instances
# loaded = {}
# filenames = {}  # filenames of loaded objects which are also saved to disk
# params = {}
# compiled = {}
# run_params = None

###########
# Step sizes
#spike_dt = None
#mf_dt = None
###########

###########################
# Project manager
###########################

data_dir = "data"
input_subdir = "inputs"
spikes_subdir = "spikes"

def get_filename(params):
    # We need a sorted dictionary of parameters, so that the hash is consistent
    flat_params = _params_to_arrays(params).flatten()
        # flatten avoids need to sort recursively
        # _params_to_arrays normalizes the data
    sorted_params = OrderedDict( (key, flat_params[key]) for key in sorted(flat_params) )
    return hashlib.sha1(bytes(repr(sorted_params), 'utf-8')).hexdigest()

def get_pathname(subdir, params):
    return os.path.normpath(data_dir + '/' + subdir) + '/' + get_filename(params)

##########################
# Parameters ?
##########################

def load_parameters(filename):
    """
    Load a parameter file.
    `np.array` is called on every non-string iterable parameter,
    so that nested lists and tuples become Nd arrays.
    """
    params = parameters.ParameterSet(filename)
    return _params_to_arrays(params)

def _params_to_arrays(params):
    for name, val in params.items():
        if isinstance(val, parameters.ParameterSet):
            params[name] = _params_to_arrays(val)
        elif not isinstance(val, str) and isinstance(val, Iterable):
            params[name] = np.array(val)
    return params

def get_random_stream(seed=314):
    global rndstream, stream_seed
    if rndstream is None:
        rndstream = shim.config.RandomStreams(seed)
        stream_seed = seed
    else:
        if seed == stream_seed:
            pass
            #logger.info("Tried to create a second random stream. Reusing the first.")
        else:
            logger.warning("Tried to obtain random stream with different seed than the current one. "
                           "The current stream was returned nonetheless.")
    return rndstream

def resolve_linked_param(params, param_name):
    """
    Allow parameter values to refer to values defined in nested parameter sets.
    Links are given by a string whose value is another key in the parameter set.
    """
    val = params[param_name]
    if ( isinstance(val, str)
         and val[-2:] == '->'
         and val[:-2] in params ):
        return resolve_linked_param(params[val[:-2]], param_name)
    else:
        return params[param_name]

def get_model_params(params):
    """Convert a ParameterSet to the internal parameter type used by models.
    Will become deprecated when models use ParameterSet."""


    # Generate the random connectivity
    #N = np.array((500, 100)) # No. of neurons in each pop
    #p = np.array(((0.1009, 0.1689), (0.1346, 0.1371))) # Connection probs between pops
    Γ = gif.GIF_spiking.make_connectivity(params.N, params.p)

    # Most parameters taken from Table 1, p.32
    # or the L2/3 values from Table 2, p. 55
    model_params = gif.GIF_spiking.Parameters(
        N      = params.N,
        R      = params.R,     # Ω, membrane resistance; no value given (unit assumes I_ext in mA)
        u_rest = params.u_rest,   # mV, p. 55
        p      = params.p,                    # Connection probability
        w      = params.w,    # mV, p. 55, L2/3
        Γ      = Γ,               # Binary connectivity matrix
        τ_m    = params.τ_m,    # s,  membrane time constant
        #τ_m    = (0.02, 0.003),    # DEBUG
        t_ref  = params.t_ref,  # s,  absolute refractory period
        u_th   = params.u_th,        # mV, non-adapting threshold  (p.54)
        u_r    = params.u_r,          # mV, reset potential   (p. 54)
        c      = params.c,        # Hz, escape rate at threshold
        Δu     = params.Δu,          # mV, noise level  (p. 54)
        Δ      = params.Δ,# s,  transmission delay
        τ_s    = params.τ_s,# mV, synaptic time constants (kernel ε)
                                  # Exc: 3 ms, Inh: 6 ms
        # Adaptation parameters   (p.55)
        J_θ    = params.J_θ,        # Integral of adaptation kernel θ (mV s)
        τ_θ    = params.τ_θ
        #τ_θ    = (1.0, 0.001)     # Adaptation time constant (s); Inhibitory part is undefined
                                  # since strength is zero; we just set a value != 0 to avoid dividing by 0
    )

    return model_params

