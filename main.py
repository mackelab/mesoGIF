# -*- coding: utf-8 -*-
"""
Created Mon May 29 2017

author: Alexandre René
"""

import logging
import os.path
import time
import copy
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

############################
# Basic configuration
# Sets logger, default filename and whether to use Theano
############################

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

def load_spikes(filename=None):
    global loaded, filenames

    if 'spiking model' in loaded:
        return filenames['spiking model']

    if filename is None:
        filename = _BASENAME + "_spikes.dat"

    # Temporarily unload Theano since it isn't supported by spike history
    use_theano = shim.config.use_theano
    shim.load(load_theano=False)

    logger.info("Checking for precomputed data...")
    try:
        raise FileNotFoundError
            # HACK: Loading model directly at present doesn't populate sinn.inputs
        # loaded['spiking model'] = io.load(filename)
        # fixparams(loaded['spiking model'])
    except FileNotFoundError:
        try:
            fn, ext = os.path.splitext(filename)
            shist = histories.Spiketrain.from_raw(io.loadraw(fn + "_spikes" + ext))
            Ihist = histories.Series.from_raw(io.loadraw(fn + "_input" + ext))
            loaded['spiking model'] = init_spiking_model(shist, Ihist)
        except FileNotFoundError:
            pass

    if 'spiking model' in loaded:
        # The spiking model is considered the ground truth
        loaded['true params'] = copy.deepcopy(loaded['spiking model'])

    # Reload Theano if it was loaded when we entered the function
    shim.load(load_theano=use_theano)

    filenames['spiking model'] = filename
    return filename

def load_mf(filename=None):
    global loaded, filenames

    if 'mf model' in loaded:
        return filenames['mf model']

    if filename is None:
        filename = _BASENAME + ".dat"

    logger.info("Checking for precomputed data...")
    try:
        raise FileNotFoundError
            # HACK: Loading model directly at present doesn't populate sinn.inputs
        loaded['mf model'] = io.load(filename)
        fixparams(loaded['mf model'])
    except FileNotFoundError:
        try:
            fn, ext = os.path.splitext(filename)
            Ahist = histories.Spiketrain.from_raw(io.loadraw(fn + "_mf" + ext))
            Ihist = histories.Series.from_raw(io.loadraw(fn + "_input" + ext))
            loaded['mf model'] = init_mean_field_model(Ahist, Ihist)
        except FileNotFoundError:
            pass

    filenames['mf model'] = filename
    return filename

def fixparams(model):
    """HACK Fix for models loaded from files that don't set the parameters' names"""
    for key, val in zip(model.params._fields, model.params):
        val.name = key

def generate_spikes(datalen, filename=None, autosave=True, recalculate=False):
    global loaded

    if not recalculate:
        # Try to load precomputed data
        filename = load_spikes(filename)
    elif 'spiking model' in loaded:
        del loaded['spiking model']

    if 'spiking model' not in loaded:
        # Spike data was not found
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
        loaded['spiking model'] = spiking_model

    else:
        logger.info("Precomputed data found. Skipping data generation")
        spiking_model = loaded['spiking model']

    sinn.flush_log_queue()

    return spiking_model


def generate_activity(datalen, filename=None, autosave=True, recalculate=False):
    global loaded

    if not recalculate:
        # Try to load precomputed data
        filename = load_mf(filename)
    elif 'mf model' in loaded:
        del loaded['mf model']

    if 'mf model' not in loaded:
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
        loaded['mf model'] = mf_model

    else:
        logger.info("Precomputed data found. Skipping data generation")

    sinn.flush_log_queue()

    return mf_model

###########################
# Data processing functions
###########################

def compute_spike_activity(filename=None):

    if 'spike activity' in loaded:
        return loaded['spike activity']

    # Temporarily unload Theano since it isn't supported by spike history
    use_theano = shim.config.use_theano
    shim.load(load_theano=False)

    load_spikes(filename)
    shist = loaded['spiking model'].s

    spikeAhist = anlz.mean(shist, shist.pop_slices) / shist.dt
    spikeAhist.name = "A (spikes)"
    spikeAhist.lock()

    # Subsample the activity and input
    Ahist = anlz.subsample(spikeAhist, np.rint(mf_dt / spike_dt).astype('int'))
    Ahist.lock()
    Ihist = anlz.subsample(loaded['spiking model'].I_ext, np.rint(mf_dt / spike_dt).astype('int'))
    Ihist.lock()

    # Remove dependencies of the subsampled data on the original
    # (this is to workaround some of sinn's intricacies)
    sinn.inputs[Ahist].clear()
    sinn.inputs[Ihist].clear()

    loaded['spike activity'] = { 'Ahist': Ahist,
                                 'Ihist': Ihist }

    # Reload Theano if it was loaded when we entered the function
    shim.load(load_theano=use_theano)


def derive_mf_model_from_spikes(filename=None):

    if 'derived mf model' in loaded:
        return loaded['derived mf model']

    compute_spike_activity(filename)

    spikemodel = loaded['spiking model']
    Ahist = loaded['spike activity']['Ahist']
    Ihist = loaded['spike activity']['Ihist']

    if shim.cf.use_theano:

        logger.info("Producing Theano mean-field model.")

        Ahist.convert_to_theano()
        Ihist.convert_to_theano()
            # These are safe to call (noops) on Theano histories

        mfmodel_params = gif.GIF_mean_field.Parameters(
            **sinn.convert_parameters_to_theano(spikemodel.params))
    else:
        mfmodel_params = spikemodel.params

    mfmodel = init_mean_field_model(Ahist,
                                    Ihist,
                                    model_params=mfmodel_params)

    if shim.cf.use_theano:
        logger.info("Theano model complete.")

    loaded['derived mf model'] = mfmodel
    return mfmodel

###########################
# Likelihood functions
###########################

def compile_theano_loglikelihood():
    global compiled

    logL = make_loglikelihood_graph2()

    logger.info("Compiling loglikelihood graph")
    compiled['logL'] = shim.gettheano().function([], [logL])
    logger.info("Done compilation.")

    return compiled['logL']

def make_loglikelihood_step(filename=None):
    global loaded

    if 'logL step' in loaded:
        return loaded['logL step'], loaded['logL step inputs']

    mfmodel = derive_mf_model_from_spikes(filename)

    logger.info("Producing the likelihood graph.")

    ####################
    # Some hacks to get around current limitations
    loaded['spiking model'].λ.name = 'spikeλ'   # remove duplicate name
    loaded['spiking model'].u.name = 'spikeu'

    histnames_to_delete = ['A_subsampled_by_5_smoothed']
    dellist = []
    for h in mfmodel.history_inputs:
        if h.name in histnames_to_delete:
            dellist.append(h)
    for h in dellist:
        del mfmodel.history_inputs[h]

    dellist = []
    for h in sinn.inputs:
        if h.name in histnames_to_delete:
            dellist.append(h)
    for h in dellist:
        del sinn.inputs[h]
    # End hacks
    #####################

    #tidx = shim.getT().scalar('tidx', dtype='int32')
    tvar = shim.getT().scalar('t', dtype='float32')
    p = sinn.clip_probabilities(mfmodel.nbar[tvar] / mfmodel.params.N)
    n = shim.cast(mfmodel.n[tvar], 'int32')
    N = mfmodel.params.N

    logL_step = ( -shim.log(shim.factorial(n, exact=False))
                  -(N-n)*shim.log(N - n) + N-n + n*shim.log(p)
                  + (N-n)*shim.log(1-p)
                ).sum()

    logger.info("Likelihood graph complete.")

    loaded['logL step'] = logL_step
    loaded['logL step inputs'] = [tvar]
    return logL_step, [tvar]

def compile_theano_loglikelihood2():
    global compiled

    if 'logL' in compiled:
        return compiled['logL']

    if not shim.config.use_theano:
        load_theano()

    logL, inputs = make_loglikelihood_step()

    logger.info("Compiling loglikelihood step")
    logL_step = shim.gettheano().function(
        inputs, logL, updates=shim.get_updates())
    logger.info("Done compiling.")

    def logL_fn(burnin, datalen):

        logger.info("Computing loglikelihood.")
        for t in np.arange(0, burnin, mfmodel.n.dt, dtype='float32'):
            # Fill the data without saving log L
            logL_step(t)

        logL = sum( logL_step(t)
                    for t in np.arange(burnin, burnin+datalen,
                                       mfmodel.n.dt, dtype='float32') )
        logger.info("Done.")
        return logL

    compiled['logL'] = logL_fn
    return logL_fn

def compile_theano_gradloglikelihood2(wrt):
    global compiled

    if 'grad logL' in compiled:
        return compiled['grad logL']

    if not shim.config.use_theano:
        load_theano()

    logL, inputs = make_loglikelihood_step()
    mfmodel = loaded['derived mf model']

    logger.info("Compiling loglikelihood gradient steps.")
    gradlogL_step = shim.gettheano().function(
        inputs, shim.getT().grad(logL, wrt), updates=shim.get_updates())
    logger.info("Done compiling.")

    def gradlogL_fn(burnin, datalen):

        logger.info("Computing loglikelihood gradient.")
        for t in np.arange(0, burnin, mfmodel.n.dt, dtype='float32'):
            # Fill the data without saving grad log L
            gradlogL_step(t)

        gradlogL = sum( gradlogL_step(t)
                    for t in np.arange(burnin, burnin+datalen,
                                       mfmodel.n.dt, dtype='float32') )
        logger.info("Done.")
        return gradlogL

    compiled['grad logL'] = gradlogL_fn
    return gradlogL_fn

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
                     burnin = 0.5, datalen = 3.4,
                     #mean_field_model = None,
                     input_filename = None,
                     output_filename = None,
                     recalculate = False,
                     ipp_url_file=None, ipp_profile=None):
    """
    […]
    Parameters
    ----------
    param1: (param, index) tuple
        First parameter to sweep (abscissa).
        `param` is equal to one of the elements of mean_field_model.params.
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

    # TODO: clean up / make treatment of mfmodel consistent with other functions
    #if mean_field_model is None:
    #    mean_field_model = _BASENAME + '.dat'
    mean_field_model = derive_mf_model_from_spikes(input_filename)

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

    # param_sweep.set_function(mean_field_model.get_loglikelihood(start=burnin,
    #                                                             stop=burnin+data_len),
    #                          'log $L$')

    # HACK: Defining the logL function with batches instead of one scan
    mbatch_size=2
    burnin_idx = mean_field_model.get_t_idx(burnin)
    stop_idx = mean_field_model.get_t_idx(burnin+data_len)

    # HACK: Workaround for issue with `sinn`
    loaded['spiking model'].λ.name = 'spikeλ'
    loaded['spiking model'].u.name = 'spikeu'

    if shim.config.use_theano:
        mean_field_model.theano_reset()
        mean_field_model.clear_unlocked_histories()
        tidx = shim.getT().lscalar('tidx')
        logL_graph, upds = mean_field_model.loglikelihood(tidx, tidx+mbatch_size)
        logger.info("Compiling Theano loglikelihood")
        logL_step = shim.gettheano().function([tidx], logL_graph,
                                              updates=upds)
        logger.info("Done compilation.")
        mean_field_model.theano_reset()
        def logL_fn(model):
            mean_field_model.clear_unlocked_histories()
            return sum(logL_step(i)
                       for i in range(burnin_idx, stop_idx, mbatch_size))
    else:
        def logL_fn(model):
            return mean_field_model.loglikelihood(burnin_idx, stop_idx-1)[0]

    param_sweep.set_function(logL_fn, 'log $L$')

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

def plot_raster(burnin, datalen, filename=None):
    load_spikes(filename)
    spikemodel = loaded['spiking model']

    plt.title("Generated spikes")
    anlz.plot(spikemodel.s, label="Population",
              start=burnin, stop=burnin+datalen,
              lineheight=1,
              markersize=1,
              alpha=0.3)
    plt.legend()

def plot_spike_activity(filename=None):
    if 'spike activity' not in loaded:
        compute_spike_activity(filename)

    Ahist = loaded['spike activity']['Ahist']

    plt.title("Activity (summed spikes, smoothed, 10ms window)")
    anlz.plot(anlz.smooth(Ahist, 10), label='A (spikes)', alpha=0.7)
    plt.legend()


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
    _init_logging_handlers()
    #load_theano()
    #generate_activity(4)
    likelihood_sweep(('w', (0,0)),
                     ('τ_m', (1,)),
                     fineness=1,
                     input_filename = 'fsgif_4s_sin-input_hi-res.dat',
                     output_filename = 'fsgif_4s_sin-input_loglikelihood.data')
                     #recalculate = recalculate,
                     #ipp_url_file = ipp_url_file,
                     #ipp_profile = ipp_profile

##########################
# cli interface
##########################

try:
    import click
    import multiprocessing
except ImportError:
    pass
else:
    ####
    # Internal state variables
    _prevent_exit = False

    ####
    # Root level cli entry point
    @click.group()
    @click.option('--theano/--no-theano', default=False)
    def cli(theano):
        _init_logging_handlers()
        if theano:
            load_theano()

    ####
    # Data generation commands
    @click.group()
    def generate():
        pass

    cli.add_command(generate)

    # TODO: Specify datalen with units (e.g. 4s or 300ms)
    @click.command()
    @click.option('--datalen', type=float)
    @click.option('--filename', default="")
    @click.option('--save/--no-save', default=True)
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

    ####
    # Computation commands
    @click.group()
    def compute():
        pass

    cli.add_command(compute)

    def _isindex(s):
        idxchars = set('0123456789,()[]')
        return all((c in idxchars) for c in s)
    def _parsetuple(s):
        idxstr = s.replace('[', '(').replace(']', ')')
        assert(all(c in idxchars[:11]) for c in idxstr[1:-1])
            # Apart from first & last characters, only numbers or commas
        assert(idxstr[0] != ')' and idxstr[-1] != '(')
        if idxstr[0] != '(':
            idxstr = '(' + idxstr
        if idxstr[-1] != ')':
            idxstr = idxstr + ')'
        if idxstr[-2] != ',':
            idxstr = idxstr[:-1] + ',)'
        return eval(idxstr)

    @click.command()
    @click.argument('params', nargs=-1)
    @click.option('--input', default="")
    @click.option('--output', default="")
    @click.option('--fineness', default="")
    @click.option('--recalculate/--use-saved', default=False)
    @click.option('--ipp_url_file', default="")
    @click.option('--ipp_profile', default="")
    def loglikelihood(params,
                      input,
                      output,
                      fineness,
                      recalculate,
                      ipp_url_file, ipp_profile):
        if len(params) < 2:
            raise ValueError("You must provide at least 2 parameters to sweep.")
        param1str = params[0]
        if _isindex(params[1]):
            param1idx = _parsetuple(params[1])
            if len(params) < 3:
                raise ValueError("You must provide at least 2 parameters to sweep.")
            i = 2
        else:
            param1idx = None
            i = 1
        param2str = params[i]
        if _isindex(params[i+1]):
            param2idx = _parsetuple(params[i+1])
        else:
            param2idx = None
        input = input if input is not "" else None
        output = output if output is not "" else None
        fineness = _parsetuple(fineness) if fineness != "" else (1,)
        fineness = fineness if len(fineness) > 1 else fineness[0]
        ipp_url_file = ipp_url_file if ipp_url_file is not "" else None
        ipp_profile = ipp_profile if ipp_profile is not "" else None

        #mfmodel = derive_mf_model_from_spikes(input)

        return likelihood_sweep((param1str, param1idx),
                                (param2str, param2idx),
                                fineness,
                                input_filename = input,
                                output_filename = output,
                                recalculate = recalculate,
                                ipp_url_file = ipp_url_file,
                                ipp_profile = ipp_profile)

    compute.add_command(loglikelihood)

    ####
    # Plotting commands

    @click.group(chain=True)
    def plot():
        pass

    cli.add_command(plot)

    @plot.resultcallback()
    def show_plots(retvals):
        """
        Prevent the script from exiting immediately, which would remove plots.
        """
        input("Press any key to exit.")

    # TODO: Allow more options for plotting likelihood
    @click.command()
    @click.option('--input', default="")
    def likelihood(input):
        if input == "":
            input = None
        kwargs = {}
        plot_likelihood(loglikelihood_filename = input,
                        ellipse = None,
                        true_params = None)
        return

    plot.add_command(likelihood)
