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
import itertools
import hashlib
import inspect
import numpy as np
import scipy as sp
import collections
from collections import namedtuple, OrderedDict, Iterable, deque
import pymc3 as pymc

import mackelab as ml
import mackelab.iotools
import mackelab.parameters
import mackelab.tqdm
import theano_shim as shim
import sinn
import sinn.histories as histories
import sinn.iotools as iotools
import sinn.analyze as anlz
from sinn.analyze.heatmap import HeatMap

from parameters import ParameterSet
from fsGIF.fsgif_model import GIF_spiking

import mackelab as ml
import mackelab.utils

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

ml.parameters._filename_printoptions['legacy'] = '1.13'
    # Allow files created with Numpy v1.13 to be loaded with v1.14
ml.parameters._remove_whitespace_for_filenames = False
    # Maintain compability with existing file names

logger = logging.getLogger('fsgif')
logger.setLevel(logging.DEBUG)
def init_logging_handlers():
    # Only attach handlers if running as a script
    import logging.handlers
    fh = logging.handlers.RotatingFileHandler('fsgif_main.log', mode='w', maxBytes=5e5, backupCount=5)
    fh.setLevel(sinn.LoggingLevels.MONITOR)
    fh.setFormatter(sinn.config.logging_formatter)
    ch = ml.tqdm.LoggingStreamHandler()
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
        'input'      : "inputs",
        'spikes'     : "spikes",
        'activity'   : "activity",
        'spike_nbar' : "spike_nbar",
        'logL_sweep' : "likelihood",
        'sgd'        : "fits",
        'mcmc'       : "mcmc_nosync",
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
        self.parser.add_argument('--debug', action='store_true',
                                 help="Indicate that this is a debug run. All file output is "
                                 "redirected to the current directory instead of being nested "
                                 "under a 'label' directory. 'subdir' is ignored.")
        self.parser.add_argument("--threadidx", type=int, default=0,
                                 help="If running multiple versions of a script in parallel, "
                                 "provide to each a different process/thread index. Scripts "
                                 "may use this information to e.g. display separate progress "
                                 "bars, or log to separate files.")
        self._mgr_argnames = ['parameters', 'recalculate']
            # This list is used to distinguish internally defined parameters from those
            # a calling script might add
            # Specified parameters are removed from 'args' before returning to the calling
            # script, so don't include parameters that should be accessible by it.
            # 'parameters' is an illegal key in a ParameterSet, so it must be removed.

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

        if params is None:
            if self.params is None:
                raise RuntimeError("You must call `load_parameters` before getting a path name.")
            else:
                params = self.params
        if subdir is None:
            subdir = self.subdir
        elif subdir[0] == '+':
            subdir = os.path.join(self.subdir, subdir[1:])
        if label is None:
            label = self.label
        return get_pathname(self.data_dir, params, suffix, subdir, self.label_dir, label)


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
                if cls is None or not isinstance(data, np.lib.npyio.NpzFile):
                    return data
                else:
                    return cls(data)

    def add_argument(self, *args, **kwargs):
        """Wrapper for the internal ArgParse parser instance."""
        self.parser(*args, **kwargs)

    def load_parameters(self, args=None, param_file=None):
        """
        Load a parameter file.
        `np.array` is called on every non-string iterable parameter,
        so that nested lists and tuples become Nd arrays.

        Parameters
        ----------
        args: list of strings
            Provide the command line parameters, overriding them if they are present.
        param_file: string
            Path to parameter file. Overrides the path obtained from arguments.
        """
        if args is None:
            if param_file is None:
                arglist = sys.argv[1:]
            else:
                arglist = []
        else:
            arglist = args
        # Get parameter file from command line arguments
        # parser.add_argument('--theano', action='store_true',
        #                     help="If specified, indicate tu use Theano. Otherwise, "
        #                          "the Numpy implementation is used.")
        #params = core.load_parameters(sys.argv[1])
        if (len(arglist) > 0 and self.smtlabel == 'cmdline'
            and args is None and param_file is None):
            # Remove the label Sumatra appended before processing cmdline options
            # Only do this if parameters were obtained from the command line
            self.label = arglist.pop()
        if param_file is not None:
            if len(arglist) == 0:
                # Don't force users to use dummy parameters if param_file is given
                arglist.append(param_file)
            else:
                arglist[-1] = param_file
        args = self.parser.parse_args(arglist)

        if hasattr(args, 'recalculate'):
            self.recalculate = args.recalculate
        else:
            self.recalculate = False

        self.params = self._params_to_arrays(ParameterSet(args.parameters))
        if 'theano' in self.params and self.params.theano:
            shim.load_theano()
        if 'sinn_compat_version' in self.params:
            sinn.config.compat_version.set(self.params.sinn_compat_version)

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

def get_trace_params(traces, posterior_desc, displaynames=None, varnames=None,
                     descriptions=None, long_descriptions=None,
                     key_sanitizer="${},_^"):
    """
    Construct the list of parameters for 1D and 2D marginals from MCMC traces,
    by flattening the parameters in those traces into a single 1D list suitable
    for iteration.
    Returns a SanitizedOrderedDict of ParamDim's; use of sanitization allows for
    "close enough" indexing.

    Parameters
    ---------------
    traces: PyMC3 MultiTrace
        The MCMC trace or traces for which we want to compute marginals.
    posterior_desc: ParameterSet
        Posterior parameters used for the MCMC trace.
    varnames: list of tuples
        List of tuple pairs (trace_name, posterior_name), associating
        names of the variables in traces to those in the posterior desc.
        If not specified, `get_marginals()` will try to associate them itself.
    displaynames: dict or ParameterSet
        keys: model names
        values: nested lists or array of display names, of same shape as the
            variable corresponding to the key.
        'model name' refers to the name of an untransformed variable, as defined
        in `posterior_desc`.
        A 'display name' is the name that will be used as an axis label. Each
        display name must be unique, as they also used as a handle to refer to
        particular variable components.
        If not specified, the model names are combined with a component index
        to create a unique label/key for each axis.
    descriptions: dict or ParameterSet
        Same format as `displaynames`. May be used to store additional free
        form text, which can retrieved with the parameters' `desc` attribute.
    long_descriptions: dict or ParameterSet
        Allows saving a second free form string, like `shortdesc`. Stored
        in the parameters' `longdesc` attribute.
    key_sanitizer: function, or list of characters
        Passed on to intializer of ml.utils.SanitizedOrderedDict.

    Returns
    -------
    SanitizedOrderedDict
        keys: Sanitized display names
        values: ParamDim instance
    """
    ParamDim = sinn.analyze.heatmap.MarginalCollection.ParamDim
    idx_filter = "0123456789," #When reconstructing index, only these characters are kept
    prior_desc = posterior_desc.model.prior
    masks = posterior_desc.mask
    varnames = dict(varnames) if varnames is not None else {}
    if displaynames is None:
        displaynames = {}
    else:
        displaynames = displaynames.copy()
    if descriptions is None:
        descriptions = {}
    else:
        descriptions = descriptions.copy()
    if long_descriptions is None:
        long_descriptions = {}
    else:
        long_descriptions = long_descriptions.copy()

    def get_trace_name(varname, idx):
        # Use the provided mapping if it is available
        search_name = varnames.get(varname, varname)
        # Find all traces with matching name
        candidate_names = [name for name in traces.varnames
                           if search_name in name
                           and '__' not in name]
                           # Variables with '__' are PyMC3-internal transformations
        # Find the one with matching index
        if len(candidate_names) == 0:
            raise ValueError("'{}' does not match any trace variable. (searched for '{}')"
                             .format(varname, search_name))
        elif len(candidate_names) == 1:
            # TODO: Check that there is no index ?
            return candidate_names[0]
        else:
            for candidate in candidate_names:
                start = candidate.find(search_name)
                suffix = candidate[start+len(search_name):]
                idx_str = "(" + ''.join(c for c in suffix if c in idx_filter) + ")"
                    # Standardize by stripping anything that isn't a number or a comma
                if idx == type(idx)(idx_str):
                    return candidate
            raise RuntimeError("No trace of variable '{}' (searched '{}') corresponds to index '{}'"
                               .format(varname, search_name, idx_str))

    # Construct a flat list of all the parameters
    #   - Parameters with multiple components appear multiple times
    #   - Accompanied by a flat_idcs list which distinguishes components.
    #     Indices in flat_idcs refer to the column index in the MultiTrace, so
    #     parameters can be retrieved as `traces.paramname[idx]`
    flat_params = ml.utils.SanitizedOrderedDict(sanitize=key_sanitizer)
        # substrings will be stripped in order, for e.g. 'log_{10}' should come before '{' or '}'
    for varname in posterior_desc.variables:
        if not any(varname in tracename for tracename in traces.varnames):
            # There are no traces for this parameter
            continue
        # Get model, transformed and trace variable names, which may or may not differ
        if hasattr(prior_desc[varname], 'transform'):
            modelname, transformedname = [name.strip()
                  for name in prior_desc[varname].transform.name.split('->')]
            assert(modelname == varname)
            to_desc = prior_desc[varname].transform.to
            back_desc = prior_desc[varname].transform.back
        else:
            modelname = transformedname = varname
            to_desc = None
            back_desc = None
        # Get mask
        mask = (np.ones(np.asarray(posterior_desc.model.params[modelname]).shape)
                * np.array(masks[varname])).astype(bool)
            # If necessary, broadcast mask to the variable's full size
        # Construct a flat list of parameters
        if np.any(mask):
            idcs = list(itertools.product(*(range(s) for s in mask.shape)))
            # Set the display name
            if modelname not in displaynames:
                displaynames[modelname] = np.array([modelname + str(idx) for idx in idcs]
                                                  )
            else:
                # Ensure that we can index the display names with array indices
                # dnames = np.array(displaynames[modelname])
                # displaynames[modelname] = dnames[mask.reshape(dnames.shape)].flatten()
                displaynames[modelname] = np.array(displaynames[modelname]).flatten()
            # Normalize the descriptions
            if modelname not in descriptions:
                descriptions[modelname] = [""]*len(idcs)
            else:
                # descs = np.array(descriptions[modelname])
                # descriptions[modelname] = descs[mask.reshape(descs.shape)].flatten()
                descriptions[modelname] = np.array(descriptions[modelname]).flatten()
            if modelname not in long_descriptions:
                long_descriptions[modelname] = [""]*len(idcs)
            else:
                # ldescs = np.array(long_descriptions[modelname])
                # long_descriptions[modelname] = ldescs[mask.reshape(ldescs.shape)].flatten()
                long_descriptions[modelname] = np.array(long_descriptions[modelname]).flatten()
            flatidcs = range(np.prod(mask.shape)) # Indices to the flattened array
            for idx, flatidx, displayname, desc, longdesc in zip(
                  idcs, flatidcs, displaynames[modelname],
                  descriptions[modelname], long_descriptions[modelname]):
                if mask[idx]:
                    assert(displayname not in flat_params)
                        # Ensure display names are unique
                    flat_params[displayname] = ParamDim(
                          varname,
                          transformedname,
                          get_trace_name(transformedname, idx),
                          displayname, desc, longdesc,
                          idx, flatidx,
                          to_desc,
                          back_desc)

    return flat_params

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

def add_extension(filename):
    """Append either '.npr', or '.sir' to a filename, depending on
    whether a file with that extension exists. If no file exists,
    append '.npr'
    """
    if os.path.exists(filename):
        pass
    elif os.path.exists(filename + '.npr'):
        filename += '.npr'
    elif os.path.exists(filename + '.sir'):
        filename += '.sir'
    else:
        filename += '.npr'
    return filename

def get_pathname(data_dir, params, suffix, subdir, label_dir=None, label=""):
    label_dir = "" if label == "" else label_dir
        # Only add the label directory when there's a label
    assert(label is not None)
    assert(label_dir is not None)  # label_dir only optional if label==""
    return os.path.join(data_dir, label_dir, label, subdir,
                        ml.parameters.get_filename(params, suffix))

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
    # TODO: Merge with 'ml.parameters.ParameterSetSampler
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

    # Convert to arrays
    params = ml.parameters.params_to_arrays(params)

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
    if not isinstance(spike_history, histories.Spiketrain):
        raise TypeError("'spike_history' must be a sinn Spiketrain.")

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
    if shim.is_theano_object(hist):
        raise NotImplementedError("Subsampling only implemented for non-"
                                  "symbolic histories.")
    newhist = anlz.subsample(hist, np.rint(target_dt / hist.dt).astype('int'))
    if max_len is not None:
        Δidx = newhist.index_interval(max_len)
        if Δidx < len(newhist._tarr) - 1:
            newhist._unpadded_length = Δidx + 1
            newhist._original_data = shim.shared(np.array(newhist._original_data[:Δidx+1]))
            newhist._data = newhist._original_data
            newhist._tarr = np.array(newhist._tarr[:Δidx+1])
            newhist.tn = newhist._tarr[-1]

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
