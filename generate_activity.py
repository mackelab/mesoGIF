import sys
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

def generate_activity(params):

    seed = params.seed
    rndstream = core.get_random_stream(seed)

    logger.info("Generating new activity data...")
    Ihist = core.subsample(
        Series.from_raw(iotools.loadraw(
            core.get_pathname(core.input_subdir, params.input))),
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
    return mfmodel

if __name__ == "__main__":
    params = core.load_parameters(sys.argv[1])
    mfmodel = generate_activity(params)
    # Save to file
    iotools.saveraw(core.get_pathname(core.activity_subdir, params), mfmodel.A)

