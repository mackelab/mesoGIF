# -*- coding: utf-8 -*-
"""
Created Mon May 29 2017

author: Alexandre René
"""

import logging
import os.path
import time
import numpy as np
import scipy as sp
from collections import namedtuple, OrderedDict

try:
    import matplotlib.pyplot as plt
except ImportError:
    logging.warning("Unable to import matplotlib. Plotting won't work.")
    do_plots = False
else:
    do_plots = True

import theano_shim as shim
import sinn
import sinn.histories as histories
import sinn.models.noise as noise
import sinn.iotools as io
import sinn.analyze as anlz
import sinn.analyze.sweep as sweep

############################
# Model import
import fsgif_model as gif
############################

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
fh = logging.handlers.RotatingFileHandler('fsgif_main.log', mode='w', maxBytes=5e5, backupCount=5)
fh.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(fh)
logger.addHandler(ch)

_BASENAME = "fsgif"

################################
# Model creation
# Sets parameters and external input
################################

def get_params():
    # Generate the random connectivity
    N = np.array((500, 100)) # No. of neurons in each pop
    p = np.array(((0.1009, 0.1689), (0.1346, 0.1371))) # Connection probs between pops
    Γ = gif.GIF_spiking.make_connectivity(N, p)

    # Most parameters taken from Table 1, p.32
    # or the L2/3 values from Table 2, p. 55
    model_params = gif.GIF_spiking.Parameters(
        N      = N,
        R      = np.array((1, 1)),     # Ω, membrane resistance; no value given (unit assumes I_ext in mA)
        u_rest = np.array((20, 20)),   # mV, p. 55
        p      = p,                    # Connection probability
        w      = ((0.176, -0.702),
                  (0.176, -0.702)),    # mV, p. 55, L2/3
        Γ      = Γ,               # Binary connectivity matrix
        τ_m    = (0.02, 0.02),    # s,  membrane time constant
        t_ref  = (0.004, 0.004),  # s,  absolute refractory period
        u_th   = (15, 15),        # mV, non-adapting threshold
        u_r    = (0, 0),          # mV, reset potential
        c      = (10, 10),        # Hz, escape rate at threshold
        Δu     = (2, 2),          # mV, noise level
        Δ      = ((0.001, 0.001),
                  (0.001, 0.001)),# s,  transmission delay
        τ_s    = ((0.003, 0.006),
                  (0.003, 0.006)),# mV, synaptic time constants (kernel ε)
                                  # Exc: 3 ms, Inh: 6 ms
        # Adaptation parameters   (p.55)
        J_θ    = (1.0, 0),        # Integral of adaptation kernel θ (mV s)
        τ_θ    = (1.0, 0.001)     # Adaptation time constant (s); Inhibitory part is undefined
                                  # since strength is zero; we just set a value != 0 to avoid dividing by 0
    )
    memory_time = 0.553  # Adjust according to τ

    return model_params, memory_time

def create_spike_history(spike_history=None, datalen=None, model_params=None):
    # Create the spike history if it wasn't passed as an argument
    if spike_history is not None:
        # if input_history is None:
        #     raise ValueError("You must provide the input history which generated these spikes.")
        shist = spike_history
        # If we were passed a history, it should already be filled with data
        assert(shist._cur_tidx.get_value() >= shist.t0idx + len(shist) - 1)
        shist.lock()
    else:
        if datalen is None:
            raise ValueError("You must specify the data length to create a new spike history.")
        shist = histories.Spiketrain(name='s',
                                     pop_sizes = model_params.N,
                                     t0 = 0,
                                     tn = datalen,
                                     dt = 0.001)
        # Set the connection weights
        # TODO: If parameters were less hacky, w would already be properly
        #       cast as an array
        w = gif.GIF_spiking.expand_param(np.array(model_params.w), model_params.N) * model_params.Γ
            # w includes both w and Γ from Eq. 20
        shist.set_connectivity(w)

    return shist

def create_activity_history(activity_history=None, datalen=None, model_params=None):
    # Create the activity history if it wasn't passed as an argument
    if activity_history is not None:
        # if input_history is None:
        #     raise ValueError("You must provide the input history which generated this activity.")
        Ahist = activity_history
        # If we were passed a history, it should already be filled with data
        assert(Ahist._cur_tidx.get_value() >= Ahist.t0idx + len(Ahist) - 1)
        Ahist.lock()
    else:
        if datalen is None:
            raise ValueError("You must specify the data length to create a new activity history.")
        Ahist = histories.Series(name='A',
                                 shape = (len(model_params.N),),
                                 t0 = 0,
                                 tn = datalen,
                                 dt = 0.001)

    return Ahist

def create_input_history(input_history=None, output_history=None, model_params=None, rndstream=None):
    # Create the input history if it wasn't passed as an argument

    if rndstream is None:
        rndstream = shim.config.RandomStreams(seed=314)

    def input_fn(t):
        import numpy as np
        # import ensures that proper references to dependencies are pickled
        # This is only necessary for scripts directly called on the cli – imported modules are fine.
        amp = np.array([6, 2])
        if not shim.isscalar(t):
            amp = amp[np.newaxis, :]
        res = amp * (1 + shim.sin(t*2*np.pi)[..., np.newaxis]) + noise_hist[t]
        return res
    if input_history is not None:
        Ihist = input_history
        # If we were passed a history, it should already be filled with data
        assert(Ihist._cur_tidx.get_value() >= Ihist.t0idx + len(Ihist) - 1)
        Ihist.lock()
    else:
        if output_history is None:
            raise ValueError("Cannot create an input history without a target output")
        noise_params = noise.GaussianWhiteNoise.Parameters(
            std = (.06, .06),
            shape = (2,)
        )
        noise_hist = histories.Series(output_history, name='ξ', shape=model_params.N.shape)
        noise_model = noise.GaussianWhiteNoise(noise_params, noise_hist, rndstream)

        Ihist = histories.Series(output_history, name='I (sin)', shape=model_params.N.shape, iterative=False)

    Ihist.set_update_function(input_fn)
    Ihist.add_inputs([noise_hist])   # TODO: Deprecate this

    return Ihist, rndstream

def init_spiking_model(spike_history=None, input_history=None, datalen=None):
    """
    Parameters
    ----------
    spike_history: sinn.Spiketrain instance
        Optional. If not given, one is be created.
    input_history: sinn.Series instance
        Optional. If not given, one is be created. Must be provided if
        `spike_history` is also provided.
    datalen: float
        Amount of data (in seconds) to generate. Required if `spike_history`
        is unspecified, ignored otherwise.

    Returns
    -------
    GIF_spiking instance
    """

    model_params, memory_time = get_params()

    shist = create_spike_history(spike_history, datalen, model_params)

    Ihist, rndstream = create_input_history(input_history, shist, model_params)

    # GIF spiking model
    spiking_model = gif.GIF_spiking(model_params, shist, Ihist, rndstream,
                                    memory_time=memory_time)
    return spiking_model


def init_mean_field_model(activity_history=None, input_history=None, datalen=None):
    """
    Parameters
    ----------
    activity_history: sinn.Spiketrain instance
        Optional. If not given, one is be created.
    input_history: sinn.Series instance
        Optional. If not given, one is be created. Must be provided if
        `spike_history` is also provided.
    datalen: float
        Amount of data (in seconds) to generate. Required if `spike_history`
        is unspecified, ignored otherwise.

    Returns
    -------
    GIF_spiking instance
    """

    model_params, memory_time = get_params()

    Ahist = create_activity_history(activity_history, datalen, model_params)

    Ihist, rndstream = create_input_history(input_history, Ahist, model_params)

    # GIF spiking model
    mf_model = gif.GIF_mean_field(model_params, Ahist, Ihist, rndstream,
                                  memory_time=memory_time)
    return mf_model

#############################
# Data generation functions
#############################

def generate_spikes(datalen, filename=None, autosave=True):

    if filename is None:
        filename = _BASENAME + "_spikes.dat"

    try:
        # Try to load precomputed data
        logger.info("Checking for precomputed data...")
        spiking_model = io.load(filename)
    except FileNotFoundError:
        logger.info("No precomputed data found. Generating new data...")
        spiking_model = init_spiking_model(datalen=datalen)
        shist = spiking_model.s
        Ihist = spiking_model.I_ext

        t1 = time.perf_counter()
        shist.set() # Compute the spikes
        t2 = time.perf_counter()
        Ihist.set() # Compute extra input time bins unused for shist

        if autosave:
            # Autosave the new data. The raw format is robust to future changes
            # of the library
            fn, ext = os.path.splitext(filename)
            io.saveraw(fn + "_spikes" + ext, shist)
            io.saveraw(fn + "_input" + ext, Ihist)

        logger.info("Done.")

    else:
        logger.info("Precomputed data found. Skipping data generation")

    return spiking_model


def generate_activity(datalen, filename=None, autosave=True):

    if filename is None:
        filename = _BASENAME + ".dat"

    try:
        # Try to load precomputed data
        logger.info("Checking for precomputed data...")
        mf_model = io.load(filename)
    except FileNotFoundError:
        logger.info("No precomputed data found. Generating new data...")
        mf_model = init_mean_field_model(datalen=datalen)
        Ahist = mf_model.A
        Ihist = mf_model.I_ext

        t1 = time.perf_counter()
        Ahist.set() # Compute the spikes
        t2 = time.perf_counter()
        Ihist.set() # Compute extra input time bins unused for shist

        if autosave:
            # Autosave the new data. The raw format is robust to future changes
            # of the library
            fn, ext = os.path.splitext(filename)
            io.saveraw(fn + "_mf" + ext, Ahist)
            io.saveraw(fn + "_input" + ext, Ihist)

        logger.info("Done.")

    else:
        logger.info("Precomputed data found. Skipping data generation")

    return mf_model


###########################
# Allow running as a script
# (to debug a function, call it from here)
###########################

if __name__ == '__main__':
    generate_activity(4)

##########################
# cli interface
##########################

try:
    import click
except ImportError:
    pass
else:
    # Root level cli entry point
    @click.group()
    def cli():
        pass

    @click.group()
    def generate():
        pass

    cli.add_command(generate)

    # TODO: Specify datalen with units (e.g. 4s or 300ms)
    @click.command()
    @click.option('--datalen', type=float)
    @click.option('--filename', default="")
    @click.option('--save/--nosave', default=True)
    def spikes(datalen, filename, save):
        return generate_spikes(datalen, filename, save)

    # TODO: Specify datalen with units (e.g. 4s or 300ms)
    @click.command()
    @click.option('--datalen', type=float)
    @click.option('--filename', default="")
    @click.option('--save/--nosave', default=True)
    def activity(datalen, filename, save):
        return generate_activity(datalen, filename, save)

    generate.add_command(spikes)
    generate.add_command(activity)
