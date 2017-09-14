import os.path
import numpy as np

import theano_shim as shim
from sinn.histories import Series, Spiketrain
import sinn.iotools as iotools

import core
from core import logger
############################
# Model import
import fsgif_model as gif
############################

def generate_activity(mgr):

    params = mgr.params
    seed = params.seed
    rndstream = core.get_random_stream(seed)

    logger.info("Generating new activity data...")
    Ihist = core.subsample(
        Series.from_raw(iotools.loadraw(
            mgr.get_pathname(params=params.input,
                             subdir=mgr.subdirs['input']))),
        params.dt)
    # Create the spiking model
    # We check if different run parameters were specified,
    # otherwise those from Ihist will be taken
    runparams = { name: params[name] for name in params
                  if name in ['t0', 'tn', 'dt'] }

    model_params = core.get_model_params(params.model)
        # Needed for now because fsgif_model does not yet use ParameterSet
    Ahist = Series(Ihist, name='A', shape=(len(model_params.N),), iterative=True,
                   **runparams)

    # GIF activity model
    mfmodel = gif.GIF_mean_field(model_params, Ahist, Ihist,
                                 params.initializer, rndstream)
    # Generate the activity trace
    mfmodel.advance('end')

    return mfmodel

if __name__ == "__main__":
    core.init_logging_handlers()
    mgr = core.RunMgr(description="Generate activity", calc='activity')
    mgr.load_parameters()
    activity_filename = mgr.get_pathname(label='')

    try:
        mgr.load(activity_filename, cls=Series.frow_raw)
    except (core.FileDoesNotExist, core.FileRenamed):
        # Get pathname with run label
        activity_filename = mgr.get_pathname(label=None)
        # Create mean-field model and generate activity
        mfmodel = generate_activity(mgr)
        # Save to file
        iotools.saveraw(activity_filename, mfmodel.A)

