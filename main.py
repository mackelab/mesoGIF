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
import collections
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
if __name__ == "__main__":
    # Only attach handlers if running as a script
    fh = logging.handlers.RotatingFileHandler('fsgif_main.log', mode='w', maxBytes=5e5, backupCount=5)
    fh.setLevel(logging.INFO)
    fh.setFormatter(sinn.config.logging_formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(sinn.config.logging_formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

_BASENAME = "fsgif"

################################
# Model creation
# Sets parameters and external input
################################

###########
# Step sizes
spike_dt = 0.0002
mf_dt = 0.001
###########

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
        u_rest = np.array((20.123, 20.362)),   # mV, p. 55
        p      = p,                    # Connection probability
        w      = ((0.176, -0.702),
                  (0.176, -0.702)),    # mV, p. 55, L2/3
        Γ      = Γ,               # Binary connectivity matrix
        τ_m    = (0.02, 0.02),    # s,  membrane time constant
        t_ref  = (0.004, 0.004),  # s,  absolute refractory period
        u_th   = (15, 15),        # mV, non-adapting threshold  (p.54)
        u_r    = (0, 0),          # mV, reset potential   (p. 54)
        c      = (10, 10),        # Hz, escape rate at threshold
        Δu     = (5, 5),          # mV, noise level  (p. 54)
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
                                     dt = spike_dt)
        # Set the connection weights
        # TODO: If parameters were less hacky, w would already be properly
        #       cast as an array
        w = gif.GIF_spiking.expand_param(np.array(model_params.w), model_params.N) * model_params.Γ
            # w includes both w and Γ from Eq. 20
        shist.set_connectivity(w)

    return shist

def create_activity_history(activity_history=None, template_history=None, datalen=None, model_params=None):
    # Create the activity history if it wasn't passed as an argument
    if activity_history is not None:
        # if input_history is None:
        #     raise ValueError("You must provide the input history which generated this activity.")
        Ahist = activity_history
        # If we were passed a history, it should already be filled with data
        assert(Ahist._cur_tidx.get_value() >= Ahist.t0idx + len(Ahist) - 1)
        Ahist.lock()
    elif template_history is not None:
        Ahist = histories.Series(template_history, name='A')
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

def init_spiking_model(spike_history=None, input_history=None, datalen=None,
                       model_params=None, memory_time=None):
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

    _model_params, _memory_time = get_params()
    if model_params is None:
        model_params = _model_params
    if _memory_time is None:
        memory_time = _memory_time

    shist = create_spike_history(spike_history, datalen, model_params)

    Ihist, rndstream = create_input_history(input_history, shist, model_params)

    # GIF spiking model
    spiking_model = gif.GIF_spiking(model_params, shist, Ihist, rndstream,
                                    memory_time=memory_time)
    return spiking_model


def init_mean_field_model(activity_history=None, input_history=None, datalen=None,
                          model_params=None, memory_time=None):
    """
    Parameters
    ----------
    activity_history: sinn.Spiketrain instance
        Optional. If not given, one is be created.
    input_history: sinn.Series instance
        Optional. If not given, one is be created. Must be provided if
        `spike_history` is also provided.
    datalen: float
        Amount of data (in seconds) to generate. Required if `activity_history`
        is unspecified, ignored otherwise.

    Returns
    -------
    GIF_spiking instance
    """

    _model_params, _memory_time = get_params()
    if model_params is None:
        model_params = _model_params
    if _memory_time is None:
        memory_time = _memory_time

    Ahist = create_activity_history(activity_history, input_history, datalen, model_params)

    Ihist, rndstream = create_input_history(input_history, Ahist, model_params)

    # GIF spiking model
    mf_model = gif.GIF_mean_field(model_params, Ahist, Ihist, rndstream,
                                  memory_time=memory_time)
    return mf_model

#############################
# Data generation functions
#############################

def generate_spikes(datalen, filename=None, autosave=True, recalculate=False):

    if filename is None:
        filename = _BASENAME + "_spikes.dat"

    datafound = False

    if not recalculate:
        # Try to load precomputed data
        logger.info("Checking for precomputed data...")
        try:
            spiking_model = io.load(filename)
            datafound = True
        except FileNotFoundError:
            try:
                fn, ext = os.path.splitext(filename)
                shist = histories.Spiketrain.from_raw(io.loadraw(fn + "_spikes" + ext))
                Ihist = histories.Series.from_raw(io.loadraw(fn + "_input" + ext))
                spiking_model = init_spiking_model(shist, Ihist)
                datafound = True
            except FileNotFoundError:
                pass

    if not datafound:
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
            io.save(filename, spiking_model)
            fn, ext = os.path.splitext(filename)
            io.saveraw(fn + "_spikes" + ext, shist)
            io.saveraw(fn + "_input" + ext, Ihist)

        logger.info("Done.")

    else:
        logger.info("Precomputed data found. Skipping data generation")

    sinn.flush_log_queue()

    return spiking_model


def generate_activity(datalen, filename=None, autosave=True, recalculate=False):

    if filename is None:
        filename = _BASENAME + ".dat"

    datafound = False
    if not recalculate:
        # Try to load precomputed data
        logger.info("Checking for precomputed data...")
        try:
            mf_model = io.load(filename)
            datafound = True
        except FileNotFoundError:
            try:
                fn, ext = os.path.splitext(filename)
                Ahist = histories.Spiketrain.from_raw(io.loadraw(fn + "_mf" + ext))
                Ihist = histories.Series.from_raw(io.loadraw(fn + "_input" + ext))
                mf_model = init_mean_field_model(shist, Ihist)
                datafound = True
            except FileNotFoundError:
                pass

    if not datafound:
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
            io.save(filename, mf_model)
            fn, ext = os.path.splitext(filename)
            io.saveraw(fn + "_mf" + ext, Ahist)
            io.saveraw(fn + "_input" + ext, Ihist)

        logger.info("Done.")

    else:
        logger.info("Precomputed data found. Skipping data generation")

    sinn.flush_log_queue()

    return mf_model

###########################
# Perform a 2D likelihood sweep
###########################

def get_sweep_param(name, index, fineness):
    # index is including so that ranges may depend on it
    if name == 'J_θ':
        return sweep.linspace(0, 3, fineness)
    elif name == 'τ_m':
        return sweep.logspace(0.003, 0.07, fineness)
    elif name == 'w':
        return sweep.linspace(-0.5, 0.5, fineness)
    else:
        raise NotImplementedError


def likelihood_sweep(param1, param2, fineness,
                     mean_field_model = None,
                     output_filename = None,
                     recalculate = False,
                     ipp_url_file=None, ipp_profile=None):
    """
    […]
    Parameters
    ----------
    param1: (param, index) tuple
        First parameter to sweep (abscissa).
        `param` is stre equal to one of the elements of mean_field_model.params.
        `index` is a tuple indicating the particular index of that
        parameter we want to sweep. If `param` is scalar, specify as `None`.
    param2: (param, index) tuple
        Second parameter to sweep (ordinate). See `param1`
    fineness: float or array-like
        Determines the number of steps along each axis in the sweep.
        Roughly corresponds to the number of steps within an interval of
        10 (linear sweeps) or a decade (log sweeps).
    recalculate: bool
        If True, will compute the likelihood even if a file
        matching `output_filename` already exists. Default is `False`.
    ipp_url_file: str
        Passed to ipyparallel.Client as `url_file`.
    ipp_profile: bytes
        Passed to ipyparallel.Client as `profile`. Ignored if ipp_url is provided.

    Returns:
    --------
    sinn.HeatMap
    """
    if output_filename is None:
        output_filename = _BASENAME + '_loglikelihood' + '.dat'

    if mean_field_model is None:
        mean_field_model = _BASENAME + '.dat'

    if isinstance(mean_field_model, str):
        try:
            # Try to load precomputed data
            mean_field_model = io.load(_filename)
        except FileNotFoundError:
            raise FileNotFoundError("Unable to find data file {}. To create one, run "
                                    "`generate` – this is required to compute the likelihood."
                                    .format(model_filename))

    if not recalculate:
        try:
            loglikelihood = io.load(output_filename)
        except FileNotFoundError:
            pass
        else:
            logger.info("Log likelihood already computed. Skipping computation.")
            return loglikelihood

    logger.info("Computing log likelihood...")
    Ihist = mean_field_model.I_ext

    # Construct the arrays of parameters to try
    #fineness = 1#75
    burnin = 0.5
    data_len = 3.2
    param_sweep = sweep.ParameterSweep(mean_field_model)
    #J_sweep = sweep.linspace(-1, 10, fineness)          # wide
    #τ_sweep = sweep.logspace(0.0005, 0.5, fineness)     # wide
    #J_sweep = sweep.linspace(1, 5, fineness)  #3, 5
    #τ_sweep = sweep.logspace(0.003, 0.07, fineness)  # 0.01, 0.07
    try:
        if len(fineness) == 1:
            fineness = list(fineness) * 2
    except TypeError:
        fineness = [fineness] * 2
    param1_stops = get_sweep_param(param1[0], param1[1], fineness[0])
    param2_stops = get_sweep_param(param2[0], param2[1], fineness[1])
    param_sweep.add_param(param1[0], idx=param1[1], axis_stops=param1_stops)
    param_sweep.add_param(param2[0], idx=param2[1], axis_stops=param2_stops)

    param_sweep.set_function(mean_field_model.get_loglikelihood(start=burnin,
                                                              stop=burnin+data_len),
                             'log $L$')

    ippclient = sinn.get_ipp_client(ipp_profile, ipp_url_file)

    if ippclient is not None:
        # Initialize the environment in each cluster process
        ippclient[:].use_dill().get()
            # More robust pickling

    # Compute the likelihood
    t1 = time.perf_counter()
    loglikelihood = param_sweep.do_sweep(output_filename, ippclient)
            # This can take a long time
            # The result will be saved in output_filename
    t2 = time.perf_counter()
    logger.info("Calculation of the likelihood took {}s."
                .format((t2-t1)))

    sinn.flush_log_queue()

    return loglikelihood

###########################
# Plotting
###########################

def plot_likelihood(loglikelihood_filename = None,
                    ellipse = None,
                    true_params = None,
                    **kwargs):
    """
    Parameters
    ----------
    loglikelihood_filename: str
        The file to load, where the log-likelihood has been stored.
    ellipse: float or array-like
        (Optional) If given, an ellipse will be drawn around the likelihood,
        with ellipse=1 indicating one standard deviation. Multiple numbers can
        be specified to draw multiple ellipses
    true_params: float tuple
        (Optional) If given, should be a two-element tuple, with first element
        indicating the true value of the parameter on the abscissa, and second
        element the true value of the parameter on the ordinate.
    **kwargs:
        These are passed on to analyze.plot.
    """
    # TODO: environment variable or get path from sinn path
    plt.style.use('../sinn/sinn/analyze/stylelib/mackelab_default.mplstyle')

    if loglikelihood_filename is None:
        loglikelihood_filename = _BASENAME + '_loglikelihood' + '.dat'

    try:
        # See if the loglikelihood has already been computed
        loglikelihood = io.load(loglikelihood_filename)
    except:
        raise RuntimeError("Unable to load loglikelihood file '{}'".format(loglikelihood_filename))

    # Convert to the likelihood. We first make the maximum value 0, to avoid
    # underflows when computing the exponential
    likelihood = (loglikelihood - loglikelihood.max()).apply_op("L", np.exp)

    # Plot the likelihood
    likelihood.cmap = 'viridis'
    likelihood.set_ceil(likelihood.max())
    likelihood.set_floor(0)
    likelihood.set_norm('linear')
    ax, cb = anlz.plot(likelihood)
        # analyze recognizes loglikelihood as a heat map, and plots accordingly
        # anlz.plot returns a tuple of all plotted objects. For heat maps there
        # are two: the heat map axis and the colour bar
    # ax.set_xlim((2, 8))
    # ax.set_ylim((0.01, 1))

    if ellipse is not None:
        if not isinstance(ellipse, collections.Iterable):
            ellipse = [ellipse]
        for d in ellipse:
            anlz.analyze.plot_stddev_ellipse(likelihood, d)

    if true_params is not None:
        color = anlz.stylelib.color_schemes.cmaps[likelihood.cmap].white
        plt.axvline(true_params[0], c=color)
        plt.axhline(true_params[1], c=color)

    plt.show(block=False)
    return


###########################
# Allow running as a script
# (to debug a function, can call it from here
#  and start the script with `pdb`)
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
