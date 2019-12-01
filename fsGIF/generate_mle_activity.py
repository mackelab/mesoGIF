import sys
import os.path
from itertools import chain
import numpy as np
from tqdm import tqdm

from parameters import ParameterSet

import theano_shim as shim
from sinn.histories import Series, Spiketrain
from sinn.optimize.gradient_descent import FitCollection

import mackelab_toolbox as ml
import mackelab_toolbox.iotools as iotools
import mackelab_toolbox.parameters
import mackelab_toolbox.utils

from fsGIF import core
from fsGIF.core import update_params
logger = core.logger
############################
# Model import
from fsGIF import fsgif_model as gif
data_dir = "data"
label_dir = "run_dump"
############################

from fsGIF.generate_activity import get_model, add_suffix
from fsGIF.gradient_descent import load_latest, get_sgd_pathname
from sumatra.parameters import _dict_diff

def load_fitcollection(paramsets):
    fitcoll = FitCollection()

    sgd_files = []
    for paramset in paramsets:
        pathname = get_sgd_pathname(paramset, label='')
        try:
            sgd, _ = load_latest(pathname)
        except (core.FileNotFound, core.FileRenamed):
            sgd = None
            logger.warning("Unable to find fit {}.".format(pathname))
        else:
            fitcoll.load(sgd, parameters=paramset)

    return fitcoll

def create_activity_paramset(paramset, base_paramset, export=None):
    """
    Construct parameter set in same format as for generating activities
    If export is provided, writes the result to that file, overwriting
    the content.

    A copy of `base_paramset` is made, so the original is preserved.

    Parameters
    ----------
    paramset: ParameterSet
        As given by FitCollection.result

    base_parameset: ParameterSet
        As required by `generate_activity.py`

    """
    # `base_paramset` can be either a string or a ParameterSet
    # If the latter, make a copy.
    if isinstance(base_paramset, str):
        base_paramset = ParameterSet(base_paramset)
    elif isinstance(base_paramset, ParameterSet):
        params = base_paramset.copy()
    else:
        raise ValueError("`base_paramset` must be either a string or a "
                         "parameter set.")
    # Check that `base_paramset` has the right format
    assert(all(key in base_paramset for key in ('input', 'model', 'seed', 'initializer', 'theano')))

    # `paramset` can be either a string or a ParameterSet
    if isinstance(paramset, str):
        paramset = ParameterSet(paramset)
    elif not isinstance(paramset, dict):
        raise ValueError("`paramset` must be either a string or a "
                         "parameter set.")

    # Create the updated parameter set.
    params.model = update_params(params.model, paramset)

    # Export to file if required
    if export is not None:
        export_params = ml.parameters.params_to_lists(params)
        export_params.save(export)

    return params


if __name__ == "__main__":

    raise NotImplementedError(
        "This script is currently unsupported because it uses parameter "
        "definitions that can't be parsed by ParameterSet, and I'm unwilling "
        "to write my own parser for this. (See `param_type` for some comments "
        "on the work this would require)\n"
        "Instead just write the MLE params to a file, and use "
        "`generate_activity.py` to generate a simulation from that.")

    core.init_logging_handlers()
    mgr = core.RunMgr(description="Generate activity with inferred parameters",
                      calc='activity')

    # Parse arguments and recover parameter sets
    args = mgr.parse_args()
    with open(args.parameters, 'r') as f:
        expanded_params = ml.parameters.expand_params(f.read())
    paramsets = [ParameterSet(s) for s in expanded_params]

    # Load collection of fits
    fitcoll = load_fitcollection(p.fits for p in paramsets)
    if len(fitcoll.fits) == 0:
        logger.warning("No fits were found. Exiting.")
        sys.exit(0)

    # Make sure that all parameter sets describe the same model
    for paramset in paramsets:
        del paramset.fits['init_vals']
        del paramset.fits['sgd']
        # Posterior params should be the same for everyone
    p0 = paramsets[0]
    for p in paramsets:
        d0, d = _dict_diff(p0, p)
        if not len(d0) == len(d) == 0:
            # Something differs somewhere
            diffkeys = set(d0).union(set(d))
            raise ValueError("For generating from fit results, parameter "
                             "expansion only works for the gradient descent "
                             "parameters. Here the parameters differ on keys "
                             "{}.".format(diffkeys))
    del params['fits']

    # Construct parameter set in same format as for generating activities
    #params = p0.copy()
    #params.model = updated_params(params.model, fitcoll.result)
    params = create_activity_paramset(p0, fitcoll.result)

    activity_filename = core.get_pathname(data_dir = data_dir,
                                          params   = params,
                                          subdir   = 'activity',
                                          suffix   = '',
                                          label_dir = label_dir,
                                          label    = '')

    try:
        iotools.load(activity_filename)
    except FileNotFoundError:
        # Get pathname with run label
        if args.debug:
            activity_filename = "activity_debug.npr"
        else:
            activity_filename = core.get_pathname(data_dir = data_dir,
                                                  params   = params,
                                                  subdir   = 'activity',
                                                  suffix   = '',
                                                  label_dir = label_dir,
                                                  label    = mgr.label)

        mfmodel = get_model(params)

        # Generate the activity trace
        # We could just call mfmodel.advance('end'), but doing it sequentially allows the progress bar
        # And since we know that variables have to be computed iteratively anyway, there's not much
        # cost to doing this.
        for i in tqdm(range(mfmodel.t0idx, mfmodel.tnidx),
                      position=args.threadidx):
            mfmodel.advance(i)

        # Save to file
        iotools.save(activity_filename, mfmodel.A, format='npr')
        iotools.save(add_suffix(activity_filename, 'nbar'), mfmodel.nbar, format='npr')
        iotools.save(add_suffix(activity_filename, 'u'), mfmodel.u, format='npr')
        iotools.save(add_suffix(activity_filename, 'vartheta'), mfmodel.varθ, format='npr')
