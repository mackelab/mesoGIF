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

try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.warning("Unable to import matplotlib. Plotting won't work.")
    do_plots = False
else:
    do_plots = True

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

_BASENAME = "fsgif"
#sinn.config.set_floatX('float32')  # Hardcoded; must match theano's floatX

def load_theano():
    """
    Run this function to use Theano for computations.
    Currently this is not supported for data generation.
    """
    shim.load_theano()

# Store loaded objects like model instances
loaded = {}
filenames = {}  # filenames of loaded objects which are also saved to disk
params = {}
compiled = {}
run_params = None

###########
# Step sizes
#spike_dt = None
#mf_dt = None
###########

###########################
# Project manager
###########################

output_dir = "data"

def get_filename(params):
    # We need a sorted dictionary of parameters, so that the hash is consistent
    flat_params = params.flatten()
        # Avoid needing to sort recursively
    sorted_params = OrderedDict( (key, flat_params[key]) for key in sorted(flat_params) )
    return hashlib.sha1(bytes(repr(sorted_params), 'utf-8')).hexdigest()

def get_pathname(subdir, params):
    return os.path.normpath(output_dir + '/' + subdir) + '/' + get_filename(params)

##########################
# Parameters ?
##########################

def load_parameters(filename):
    global run_params
    run_params = parameters.ParameterSet(filename)

def get_params():
    global spike_dt, mf_dt

    if run_params is None:
        # parameter_file is a global parameter
        raise RuntimeError("Parameters were not loaded.")

    spike_dt = run_params.sim.spike_dt
    mf_dt = run_params.sim.mf_dt
    memory_time = run_params.sim.memory_time  # Adjust according to τ

    # Model parameters are stored as lists -> convert to arrays
    run_params.model.replace_values( **{pname:np.array(pval)
                                        for pname, pval
                                        in run_params.model.parameters()} )

    # Generate the random connectivity
    #N = np.array((500, 100)) # No. of neurons in each pop
    #p = np.array(((0.1009, 0.1689), (0.1346, 0.1371))) # Connection probs between pops
    Γ = gif.GIF_spiking.make_connectivity(run_params.model.N, run_params.model.p)

    # Most parameters taken from Table 1, p.32
    # or the L2/3 values from Table 2, p. 55
    model_params = gif.GIF_spiking.Parameters(
        N      = run_params.model.N,
        R      = run_params.model.R,     # Ω, membrane resistance; no value given (unit assumes I_ext in mA)
        u_rest = run_params.model.u_rest,   # mV, p. 55
        p      = run_params.model.p,                    # Connection probability
        w      = run_params.model.w,    # mV, p. 55, L2/3
        Γ      = Γ,               # Binary connectivity matrix
        τ_m    = run_params.model.τ_m,    # s,  membrane time constant
        #τ_m    = (0.02, 0.003),    # DEBUG
        t_ref  = run_params.model.t_ref,  # s,  absolute refractory period
        u_th   = run_params.model.u_th,        # mV, non-adapting threshold  (p.54)
        u_r    = run_params.model.u_r,          # mV, reset potential   (p. 54)
        c      = run_params.model.c,        # Hz, escape rate at threshold
        Δu     = run_params.model.Δu,          # mV, noise level  (p. 54)
        Δ      = run_params.model.Δ,# s,  transmission delay
        τ_s    = run_params.model.τ_s,# mV, synaptic time constants (kernel ε)
                                  # Exc: 3 ms, Inh: 6 ms
        # Adaptation parameters   (p.55)
        J_θ    = run_params.model.J_θ,        # Integral of adaptation kernel θ (mV s)
        τ_θ    = run_params.model.τ_θ
        #τ_θ    = (1.0, 0.001)     # Adaptation time constant (s); Inhibitory part is undefined
                                  # since strength is zero; we just set a value != 0 to avoid dividing by 0
    )

    return model_params, memory_time

