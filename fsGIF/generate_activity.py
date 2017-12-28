import os.path
import numpy as np

import theano_shim as shim
from sinn.histories import Series, Spiketrain
import sinn.iotools as iotools

from fsGIF import core
logger = core.logger
############################
# Model import
from fsGIF import fsgif_model as gif
############################

def generate_activity(mgr):

    params = mgr.params
    seed = params.seed
    rndstream = core.get_random_stream(seed)

    logger.info("Generating new activity data...")
    # Ihist = core.subsample(
    #     Series.from_raw(iotools.loadraw(
    #         mgr.get_pathname(params=params.input,
    #                          subdir=mgr.subdirs['input']))),
    #     params.dt)
    input_filename = core.add_extension(
        mgr.get_pathname(params.input,
                         subdir=mgr.subdirs['input'],
                         label=''))
    Ihist = iotools.load(input_filename)
    if isinstance(Ihist, np.lib.npyio.NpzFile):
        # Support older data files
        Ihist = Series.from_raw(Ihist)
    # Create the spiking model
    # We check if different run parameters were specified,
    # otherwise those from Ihist will be taken
    runparams = { name: params[name] for name in params
                  if name in ['t0', 'tn', 'dt'] }

    model_params = core.get_model_params(params.model, 'GIF_mean_field')
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
        mgr.load(activity_filename, cls=Series.from_raw)
    except (core.FileNotFound, core.FileRenamed):
        # Get pathname with run label
        activity_filename = core.add_extension(mgr.get_pathname(label=None))
        # Create mean-field model and generate activity
        mfmodel = generate_activity(mgr)
        # Save to file
        iotools.save(activity_filename, mfmodel.A, format='npr')

