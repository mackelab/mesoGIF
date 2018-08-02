import sys
import os.path
import numpy as np
from tqdm import tqdm

from parameters import ParameterSet

import theano_shim as shim
from sinn.histories import Series, Spiketrain
from sinn.optimize.gradient_descent import FitCollection

import mackelab as ml
import mackelab.iotools as iotools
import mackelab.parameters

from fsGIF import core
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

if __name__ == "__main__":
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

    # Construct parameter set in same format as for generating activities
    params = p0.copy()  # all p's guaranteed to be the same at this point
    params.model = ml.parameters.params_to_arrays(params.model)
    result = fitcoll.result
    for key, val in result.items():
        result[key] = val.reshape(params.model[key].shape)
    params.model.update(result)
    del params['fits']

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
                                                  label    = None)

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
