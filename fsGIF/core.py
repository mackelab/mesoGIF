"""
Created Mon May 29 2017

author: Alexandre René
"""

import logging
import os.path
import sys
import argparse
import time
import copy
import hashlib
import inspect
import numpy as np
import scipy as sp
import collections
from collections import namedtuple, OrderedDict, Iterable
import pymc3 as pymc

import mackelab as ml
import mackelab.iotools
import mackelab.parameters
import theano_shim as shim
import sinn
import sinn.iotools as iotools
import sinn.analyze as anlz
from sinn.analyze.heatmap import HeatMap

from parameters import ParameterSet
from fsGIF.fsgif_model import GIF_spiking

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

logger = logging.getLogger('fsgif')
logger.setLevel(logging.DEBUG)
def init_logging_handlers():
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

# def load_theano():
#     """
#     Run this function to use Theano for computations.
#     Currently this is not supported for data generation.
#     """
#     shim.load_theano()

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
# Exceptions
###########################
class FileExists(Exception):
    pass
class FileNotFound(Exception):
    pass
class FileRenamed(Exception):
    pass
###########################

###########################
# Run manager
###########################

class RunMgr:
    """
    Run Manager
    Implements a few convenience functions for scripts designed to be called using Sumatra.
    The idea is to route all calls to the file system (e.g. to ask for a filename or save data)
    through this class. This allows it to adjust filenames to follow the Sumatra recommended workflow.

    Features:
      - Unique filename creation based on the parameter set.
      - Caching: calculations with the same parameter set simply return the saved data when it is present.
      - Adds a 'recalculate' command line option, to override caching mechanism. The old data is then
        renamed with an appended number.
      - Automatically takes care of the Sumatra 'label' mechanism, appending the label to the read/write
        root directory. In the Sumatra configuration, the label option should be set to 'cmdline'.

    Usage:
      Subclass in your own project directory and set the class-level attributes.
      Then import the subclassed manager in your scripts.

    TODO
    =====
    - Search add different labels directories when looking for a free filename
    - Idem for existing filename
    - Update the Sumatra records database when renaming
    """

    data_dir = "data"
    label_dir = "run_dump"
    subdirs = {
        'input'     : "inputs",
        'spikes'    : "spikes",
        'activity'  : "activity",
        'logL_sweep': "likelihood",
        'sgd'       : "fits",
        'mcmc'      : "mcmc_nosync",
        }
    smtlabel = 'cmdline'
        # Either 'cmdline' or None. 'parameters' not currently supported
    _load_fn = iotools.loadraw
         # FIXME: Make this work as self._load_fn. Currently that is seen as a
         #   method, and so it is passed 'self' as first argument.

    def __init__(self, description="", calc=""):
        """
        Parameters
        ----------
        description: str,
            Description text shown at the top of the usage description.
        calc: str,
            Type of calculation. If it matches one of the keys in `subdirs`, output
            is saved in the corresponding subdirectory.
        load_fn: callable
            (Optional) Function used to load data given a filename. If not specified,
            `np.load` is used.
            If load_fn checks multiple paths (e.g. different extensions/subdirectories),
            it should provide the `return_path` keyword which, when True, returns the
            loaded path along with the data as `(data, path)`.
        """
        self.parser = argparse.ArgumentParser(description=description)
        self.parser.add_argument('parameters', type=str, help="Parameter file.")
        self.parser.add_argument('--recalculate', action='store_true',
                                 help="If passed, force the recalculation of data. If a result file "
                                 "matching the given parameters is found, a number is appended to it, "
                                 "allowing the new data to take the expected filename.")
        self._mgr_argnames = ['parameters', 'recalculate']
            # This list is used to distinguish internally defined parameters from those
            # a calling script might add

        #self.parser.add_argument('--label', type=str,
        #                         help="Label parameter, automatically provided by Sumatra.",
        #                         default="")
        self.calc = calc
        if calc in self.subdirs:
            self.subdir = self.subdirs[calc]
        else:
            self.subdir = ""

        self.params = None

        # if load_fn is None:
        #     self._load_fn = np.load
        # else:
        #     self._load_fn = load_fn

    def get_filename(self, params=None, suffix=None):
        if params is None:
            params = self.params
        return mackelab.parameters.get_filename(params, suffix)
        #return self._get_filename(params, suffix)

    @classmethod
    def _get_filename(cls, params, suffix=None):
        return mackelab.parameters.get_filename(params, suffix)
        # if params == '':
        #     basename = ""
        # else:
        #     # We need a sorted dictionary of parameters, so that the hash is consistent
        #     flat_params = cls._params_to_arrays(params).flatten()
        #         # flatten avoids need to sort recursively
        #         # _params_to_arrays normalizes the data
        #     sorted_params = OrderedDict( (key, flat_params[key]) for key in sorted(flat_params) )
        #     basename = hashlib.sha1(bytes(repr(sorted_params), 'utf-8')).hexdigest()
        #     basename += '_'
        # if isinstance(suffix, str):
        #     suffix = suffix.lstrip('_')
        # if suffix is None or suffix == "":
        #     assert(len(basename) > 1 and basename[-1] == '_')
        #     return basename[:-1] # Remove underscore
        # elif isinstance(suffix, str):
        #     return basename + suffix
        # elif isinstance(suffix, Iterable):
        #     assert(len(suffix) > 0)
        #     return basename + '_'.join([str(s) for s in suffix])
        # else:
        #     return basename + str(suffix)

    def get_pathname(self, params=None, suffix=None, subdir=None, label=""):
        """
        Construct a pathname by hashing a ParameterSet. The resulting path will be
        [data_dir]/[label_dir]/[label]/[subdir]/hash([params])_suffix
        All parameters are optional; the effects of their defaults are given below.
        (data_dir and label_dir are class attributes)

        Parameters
        ----------
        params: ParameterSet
            ParameterSet instance. Its SHA1 hash will form the filename.
            Default is to take the parameter instance obtained from `load_parameters`.

        suffix: str
            String appended to the filename. Useful to differentiate different output files
            obtained from the same parameter set.
            Default is not to add a suffix.

        subdir: str
            Subdirectory (below the label) in which to put/get the file.
            Default is to use the subdirectory defined by the run manager's 'calc' attribute.
            (see RunMgr.__init__)
            If the string begins with '+', it is appended to the default subdirectry.
            I.e. if the latter is 'likelihood' and subdir is '+run1', then the returned
            path will contain 'likelihood/run1' as a subdirectory.

        label: str or None
            Prefix directory. This matches the use of Sumatra for the 'label' being a run-
            specific root directory in which to put files. This allows it to differentiate
            the output of different simultaneous runs.
            Default is not to add a label. If the value is None, the label value provided by
            Sumatra is used (this requires that `load_parameters` has already been executed).
        """

        if params is None and self.params is None:
            raise RuntimeError("You must call `load_parameters` before getting a path name.")
        if subdir is None:
            subdir = self.subdir
        elif subdir[0] == '+':
            subdir = os.path.join(self.subdir, subdir[1:])
        if label is None:
            label = self.label
        label_dir = "" if label == "" else self.label_dir
            # Only add the label directory when there's a label
        return os.path.join(self.data_dir, label_dir, label, subdir,
                            self.get_filename(params, suffix))

    @staticmethod
    def rename_to_free_file(path):
        new_f, new_path = ml.iotools.get_free_file(path)
        new_f.close()
        os.rename(path, new_path)
        return new_path

    @classmethod
    def find_path(cls, path):
        """
        Find the path at which a file resides. Uses load_fn internally, and so
        searches the same paths.
        """
        try:
            _, datapath = cls.load_fn(path, return_path=True)
        except IOError:
            return None
        else:
            return datapath

    def load(self, pathname, cls=None, calc=None, recalculate=None):
        """
        Parameters
        ----------
        pathname: str
            Path name as returned by a call to `get_pathname`.
        calc: str
            (Optional) Same possible values as __init__'s 'calc' parameter.
            At present only used for error message.
            Default is to use the instance's corresponding attribute.
        cls: class or function
            (Optional) If specified, will be applied to the loaded data before returning.
            I.e. the returned value will be `cls(self.load_fn(pathname))`, instead of
            `self.load_fn(pathname)` if unspecified.
        recalculate: bool
            (Optional) Indicate whether to force recalculation.
            Default is to use the instance's corresponding attribute (if set), otherwise False.
        """
        # Set the default values
        if calc is None:
            calc = self.calc
        if recalculate is None:
            try:
                recalculate = self.recalculate
            except AttributeError:
                recalculate = False

        # Try loading the data
        try:
            data, datapath = self.load_fn(pathname, return_path=True)
        except IOError:
            # This data does not exist
            raise FileNotFound("File '{}' does not exist."
                                   .format(pathname))
        else:
            if recalculate:
                # Data does already exist, but we explicitly asked to recalculate it:
                # move the current data to a new filename.
                # The data is not loaded.
                new_path = self.rename_to_free_file(datapath)
                logger.info("Recalculating. Previous {} data moved to {}."
                            .format(calc, new_path))
                raise FileRenamed("File '{}' was renamed to {}."
                                  .format(datapath, new_path))
            else:
                # Data was found; load it.
                logger.info("Precomputed {} data found."
                            .format(calc))
                if cls is None:
                    return data
                else:
                    return cls(data)

    def add_argument(self, *args, **kwargs):
        """Wrapper for the internal ArgParse parser instance."""
        self.parser(*args, **kwargs)

    def load_parameters(self):
        """
        Load a parameter file.
        `np.array` is called on every non-string iterable parameter,
        so that nested lists and tuples become Nd arrays.
        """
        # parser.add_argument('--theano', action='store_true',
        #                     help="If specified, indicate tu use Theano. Otherwise, "
        #                          "the Numpy implementation is used.")
        #params = core.load_parameters(sys.argv[1])
        if self.smtlabel == 'cmdline':
            # Remove the label Sumatra appended before processing cmdline options
            self.label = sys.argv.pop()
        args = self.parser.parse_args()

        self.recalculate = args.recalculate

        self.params = self._params_to_arrays(ParameterSet(args.parameters))
        if 'theano' in self.params and self.params.theano:
            shim.load_theano()

        self.args = ParameterSet( {name: val
                                   for name, val in vars(args).items()
                                   if name not in self._mgr_argnames} )

        # Add flags so that 'params' uniquely identifies this data
        # parameter_flags = ['theano']
        # for flag in parameter_flags:
        #     setattr(params, flag, getattr(args, flag))

        # Other flags that don't affect the data (e.g. Sumatra label)
        #flags = {}

        #return _params_to_arrays(params), flags

    @classmethod
    def load_fn(cls, pathname, return_path=False):
        """
        Custom data loading functions should allow a 'return_path' keyword,
        if they try to load from multiple paths.
        This function provides a consistent interface to the load_fn set
        during initialization: if it accepts return_path, that is used, otherwise
        the `pathname` is simply returned when `return_path` is True.
        """
        sig = inspect.signature(cls._load_fn)
        if 'return_path' in sig.parameters:
            return cls._load_fn(pathname, return_path=return_path)
        else:
            data = cls._load_fn(pathname)
            if return_path:
                return data, pathname
            else:
                return data

    @classmethod
    def _params_to_arrays(cls, params):
        return mackelab.parameters.params_to_arrays(params)
        # """Also converts dictionaries to parameter sets."""
        # for name, val in params.items():
        #     if isinstance(val, (ParameterSet, dict)):
        #         params[name] = cls._params_to_arrays(val)
        #     elif (not isinstance(val, str)
        #         and isinstance(val, Iterable)
        #         and all(type(v) == type(val[0]) for v in val)):
        #             # The last condition leaves objects like ('lin', 0, 1) as-is;
        #             # otherwise they would be casted to a single type
        #         params[name] = np.array(val)
        # return ParameterSet(params)

def _split_number(s):
    """
    Split a string on the first character which is a number.
    If no number is found, returns `s, None`.
    """
    start_i = -1
    for i, c in enumerate(s):
        if c.isdigit():
            start_i = i
            break
    if start_i == -1:
        return s, None
    else:
        assert(s[start_i:].isdigit())
    return s[:start_i], s[start_i:]

def get_suffixes(filename):
    basename, _ = os.path.splitext(os.path.basename(filename))
    suffixes = basename.split("_")[1:]
    return {key: val
            for key, val in
            [_split_number(suffix) for suffix in suffixes]}

def isarchived(filename):
    """
    Return True if a file is archived and should be ignored.
    Archived files are recognized by having a trailing, numerical-only suffix
    """
    basename, _ = os.path.splitext(os.path.basename(filename))
    suffixes = basename.split("_")[1:]
    if len(suffixes) > 0 and suffixes[-1].isdigit():
        return True
    else:
        return False

def get_param_values(param_desc):
    """
    Takes a description of parameters in a particular format, and
    converts to a 'cartesian' format.
    param_desc may include the 'random' key, in which case an
    appropriate random value is returned.
    """
    if 'random' in param_desc and param_desc.random:
        if ( 'seed' in param_desc and param_desc.seed is not None ):
            np.random.seed(param_desc.seed)
        logger.debug("RNG state: {}".format(np.random.get_state()[1][0]))

    if 'format' not in param_desc or param_desc.format == 'cartesian':
        if 'random' in param_desc and param_desc.random:
            raise NotImplementedError
        new_param_desc = param_desc

    elif param_desc.format in ['polar', 'spherical']:

        if 'center' in param_desc:
            if 'centre' in param_desc:
                raise ValueError("The parameter description defines both 'centre' and 'center'. "
                                 "This is ambiguous as they are synonymous: remove one.")
            param_desc.centre = param_desc.center
        centre = OrderedDict( (name, np.array(param_desc.centre[name]))
                               for name in param_desc.variables )
        # The total number of variables is the sum of each variable's number of elements
        nvars = sum( np.prod(var.shape) for var in centre.values() )

        # Get the coordinate angles
        if 'random' in param_desc and param_desc.random:
            # All angles except last [0, π)
            # Last angle [0, 2π)
            angles = np.uniform(0, np.pi, nvars - 1)
            angles[-1] = 2*angles[-1]
        else:
            # The angles may be given with nested structure; this is just to help
            # legibility, so flatten everything.
            angles = np.concatenate([np.array(a).flatten() for a in param_desc.angles])
            if len(angles) != nvars - 1:
                raise ValueError("Number of coordinate angles (currently {}) must be "
                                    "one less than the number of variables. (currently {})."
                                    .format(len(param_desc.angles), len(param_desc.variables)))

        # Compute point on the unit sphere
        sines = np.concatenate(([1], np.sin(angles)))
        cosines = np.concatenate((np.cos(angles), [1]))
        unit_vals_flat = np.cumprod(sines) * cosines
        # "unflatten" the coordinates
        unit_vals = []
        i = 0
        for name, val in centre.items():
            varlen = np.prod(val.shape)
            unit_vals.append(unit_vals_flat[i:i+varlen].reshape(val.shape))
            i += varlen

        # rescale coords
        radii = []
        for name, val in centre.items():
            radius = param_desc.radii[name]
            if shim.isscalar(radius):
                radii.append( np.ones(val.shape) * radius )
            else:
                if radius.shape != val.shape:
                    raise ValueError("The given radius has shape '{}'. It should "
                                        "either be scalar, or of shape '{}'."
                                        .format(radius.shape, val.shape))
                radii.append(radius)
        rescaled_vals = [val * radius for val, radius in zip(unit_vals, radii)]

        # add the centre
        recentred_vals = [c + r for c, r in zip(centre.values(), rescaled_vals)]

        # construct the new parameter set
        new_param_desc = ParameterSet({
            'format': 'cartesian',
            'random': False,
            'variables': param_desc.variables,
        })
        for name, val in zip(centre.keys(), recentred_vals):
            new_param_desc[name] = val

    else:
        raise ValueError("Unrecognized parameter format '{}'.".format(param_desc.format))

    return new_param_desc

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

# def resolve_linked_param(params, param_name):
#     """
#     Allow parameter values to refer to values defined in nested parameter sets.
#     Links are given by a string whose value is another key in the parameter set.
#     """
#     val = params[param_name]
#     if ( isinstance(val, str)
#          and val[-2:] == '->'
#          and val[:-2] in params ):
#         return resolve_linked_param(params[val[:-2]], param_name)
#     else:
#         return params[param_name]
def get_sampler(dists):
    # var: shared variable to fill with the sample
    def _get_sample(distparams, var):
        shape = var.get_value().shape
        if len(shape) == 0:
            shape = None

        factor = distparams.factor if 'factor' in distparams else 1

        if distparams.dist == 'normal':
            return factor * np.random.normal(distparams.loc,
                                             distparams.scale, size=shape)
        elif distparams.dist == 'expnormal':
            return factor * np.exp(
                np.random.normal(distparams.loc,
                                 distparams.scale, size=shape) )
        elif distparams.dist in ['exponential', 'exp']:
            return factor * np.random.exponential(distparams.scale,
                                                  size=shape)
        elif distparams.dist == 'gamma':
            return factor * np.random.gamma(shape=distparams.a, scale=distparams.scale,
                                            size=distparams.shape)
        elif distparams.dist == 'mixed':
            comps = distparams.components
            distlist = [distparams[comp] for comp in comps]
            idx = np.random.choice(len(comps), p=distparams.probabilities)
            return factor * _get_sample(distlist[idx], var)
        else:
            raise ValueError("Unrecognized distribution type '{}'."
                             .format(distparams.dist))

    def sampler(var):
        if var.name not in dists:
            raise ValueError("There is no distribution associated to the "
                             "variable name '{}'.".format(var.name))
        return _get_sample(dists[var.name], var)

    return sampler

def get_model_params(params, model_type):
    """Convert a ParameterSet to the internal parameter type used by models.
    Will become deprecated when models use ParameterSet."""


    # Generate the random connectivity
    #N = np.array((500, 100)) # No. of neurons in each pop
    #p = np.array(((0.1009, 0.1689), (0.1346, 0.1371))) # Connection probs between pops
    if model_type == 'GIF_spiking':
        Γ = GIF_spiking.make_connectivity(params.N, params.p)
    elif model_type == 'GIF_mean_field':
        Γ = None
    else:
        raise ValueError("Unrecognized model type '{}'.".format(model_type))

    # Most parameters taken from Table 1, p.32
    # or the L2/3 values from Table 2, p. 55
    model_params = GIF_spiking.Parameters(
        N      = params.N,
        R      = params.R,     # Ω, membrane resistance; no value given (unit assumes I_ext in mA)
        u_rest = params.u_rest,   # mV, p. 55
        p      = params.p,                    # Connection probability
        w      = params.w,    # mV, p. 55, L2/3
        Γ      = Γ,               # Binary connectivity matrix
        τ_m    = params.τ_m,    # s,  membrane time constant
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
        τ_θ    = params.τ_θ       # Adaptation time constant (s); Inhibitory part is undefined
                                  # since strength is zero; we just set a value != 0 to avoid dividing by 0
    )

    return model_params

def construct_model(model_module, model_params, data_history, input_history, initializer=None):
    """
    Parameters
    ----------
    model_module: module
        Models are defined in .py files and imported. This is the imported module.
    model_params: ParameterSet
        Parameter set defining the model. Must minimally contain:
           - 'type': Class name in `model_module`. Selects which model will be created.
           - 'params': The set of parameters expected by the model selected with `type`
        'initializer' is also usually defined.
    data_history: History
        History used as 'data' (e.g. activity, spikes, rate, etc.)
    input_history: History
        History used as input when generating/obtaining the data. Typically an instance of Series
    initializer: str
        Flag indicating how to initialize the model; the chosen model must define the corresponding
        initializer.
        By default the value of `model_params.initializer` is used.
    """
    module_params = get_model_params(model_params.params, model_params.type)
    return getattr(model_module, model_params.type)(
        module_params,
        data_history,
        input_history,
        initializer=initializer)

###########################
# Data processing functions
###########################

def compute_spike_activity(spike_history, activity_dt=None):
    """
    Parameters
    ----------
    spike_history: Spiketrain instance
        If given, use this spike_history rather than the one already loaded
        or the one found under `filename`.
    activity_dt: float
        Time step of the activity trace. Default is to use that of the spike history;
        must be an integer multiple of the latter.
    """

    # Temporarily unload Theano since it isn't supported by spike history
    use_theano = shim.config.use_theano
    shim.load(load_theano=False)

    # Compute the activity with time bins same as the spikes
    spikeAhist = anlz.mean(spike_history, spike_history.pop_slices) / spike_history.dt
    spikeAhist.name = "A (spikes)"
    spikeAhist.lock()

    # Subsample the activity to match the desired bin length
    if activity_dt is None:
        activity_dt = spikeAhist.dt
    Ahist = subsample(spikeAhist, activity_dt)

    # Reload Theano if it was loaded when we entered the function
    shim.load(load_theano=use_theano)

    return Ahist

def subsample(hist, target_dt, max_len = None):
    """
    max_len: float
        (Optional) Maximum length of data to keep. If the source data (`hist`)
        is longer, the resulting activity and input arrays are truncated.
        If specified as an integer, considered as a number of bins rather
        than time units.
    """
    newhist = anlz.subsample(hist, np.rint(target_dt / hist.dt).astype('int'))
    if max_len is not None:
        idx = newhist.get_t_idx(max_len)
        if idx < len(newhist._tarr) - 1:
            newhist._unpadded_length = idx - newhist.t0idx + 1
            newhist._original_data = shim.shared(np.array(newhist._original_data[:idx+1]))
            newhist._data = hist._original_data
            newhist._tarr = np.array(hist._tarr[:idx+1])
            newhist.tn = hist._tarr[-1]

    newhist.lock()

    # Remove dependencies of the subsampled data on the original
    # (this is to workaround some of sinn's intricacies)
    # sinn.inputs[newhist].clear()

    return newhist

# Windowed crosscorrelation function
from numpy.lib.stride_tricks import as_strided
def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x
def crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.

    https://stackoverflow.com/a/34558964
    """
    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)

def exploglikelihood(loglikelihood):
    """
    Return the likelihood, given the loglikelihood.

    Parameters
    ----------
    loglikelihood: HeatMap
    """
    assert(isinstance(loglikelihood, HeatMap))

    # Convert to the likelihood. We first make the maximum value 0, to avoid
    # underflows when computing the exponential
    likelihood = (loglikelihood - loglikelihood.max()).apply_op("L", np.exp)

    # Plot the likelihood
    likelihood.cmap = 'viridis'
    likelihood.set_ceil(likelihood.max())
    likelihood.set_floor(0)
    likelihood.set_norm('linear')

    return likelihood


##############################
# Debug helpers
##############################

def match_params(params, *values):
    """
    Check if the values of `params` matches those in `values`.
    Example: to check whether params.x[0,1] equals 4.1, and
    params.y equals 1, the call should be

         match_params(params, ('x', (0,1), 4.1), ('y', 1))

    The comparison only considers as many significant digits as provided.
    TODO: Allow to specifiy sig digits.

    Parameters
    ----------
    params: ParameterSet

    *values: tuples
        One tuple per parameter to check. Each tuple of the form
        `(name, idx, val)` or `(name, val)`
        where 'name' is a string and we want to test params.name[idx] == val.

    Returns
    -------
    bool
    """
    if len(values) == 0:
        logger.warning("Testing match on zero parameters. This will always return True.")
        return True

    matches = True
    for valtuple in values:
        # Determine number of significant digits / tolerance
        if abs(valtuple[-1]) >= 1:
            atol = 10**len(str(int(abs(valtuple[-1]))))
        else:
            frac = abs(valtuple[-1]) - int(abs(valtuple[-1]))
            atol = 10**( - (len(str(frac))-2) )  # -2 for the '0.'
        if len(valtuple) == 2:
            # param is a scalar
            if not sinn.isclose(getattr(params, valtuple[0]), valtuple[-1], atol=atol):
                matches = False
                break
        elif len(valtuple) == 3:
            # param is an array
            if not sinn.isclose(getattr(params, valtuple[0])[valtuple[1]], valtuple[-1], atol=atol):
                matches = False
                break
    if matches:
        pass

    return matches
