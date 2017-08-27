import sys
import numpy as np

import parameters

import theano_shim as shim
from sinn.histories import Series, Spiketrain
import sinn.iotools as iotools

import core
from core import logger
############################
# Model import
import fsgif_model as gif
############################

"""
Expected parameters format:
{ 'input': {input_params},
  'model': {model_params},
  ['t0', 'tn', 'dt']
}
"""

def generate_spikes(params):
    # Temporarily unload Theano since it isn't supported by spike history
    use_theano = shim.config.use_theano
    shim.load(load_theano=False)

    seed = core.resolve_linked_param(params, "seed")
    rndstream = core.get_random_stream(seed)

    logger.info("Generating new spike data...")
    Ihist = Series.from_raw(iotools.loadraw(core.get_pathname(core.input_subdir, params.input)))

    # Create the spiking model
    # We check if different run parameters were specified,
    # otherwise those from Ihist will be taken
    runparams = { name: val for name, val in params.items()
                   if name in ['t0', 'tn', 'dt'] }
    # TODO: if dt different from Ihist, subsample Ihist
    shist = Spiketrain(Ihist, name='s', pop_sizes = params.model.N, iterative=True,
                       **runparams)
    model_params = core.get_model_params(params.model)
        # Needed for now because fsgif_model does not yet use ParameterSet
    spiking_model = gif.GIF_spiking(model_params, shist, Ihist, rndstream)
    w = gif.GIF_spiking.expand_param(np.array(model_params.w), model_params.N) * model_params.Γ
        # w includes both w and Γ from Eq. 20
    shist.set_connectivity(w)


    # Generate the spikes
    shist.set()

    # Save to file
    iotools.saveraw(core.get_pathname(core.spikes_subdir, params), shist)
    logger.info("Done.")

    # Reload Theano if it was loaded when we entered the function
    shim.load(load_theano=use_theano)


if __name__ == "__main__":
    params = core.load_parameters(sys.argv[1])
    generate_spikes(params)


