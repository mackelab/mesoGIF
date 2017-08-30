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

    seed = params.seed
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
    spiking_model = gif.GIF_spiking(model_params, shist, Ihist,
                                    params.initializer, rndstream)
    w = gif.GIF_spiking.expand_param(np.array(model_params.w), model_params.N) * model_params.Γ
        # w includes both w and Γ from Eq. 20
    shist.set_connectivity(w)

    # Generate the spikes
    shist.set()

    logger.info("Done.")

    # Reload Theano if it was loaded when we entered the function
    shim.load(load_theano=use_theano)

    return spiking_model

if __name__ == "__main__":
    parser = core.argparse.ArgumentParser(description="Generate spikes")
    params = core.load_parameters(parser)
    spike_filename = core.get_pathname(core.spikes_subdir, params)
    try:
        # Try to load data to see if it's already been calculated
        spikes_raw = iotools.loadraw(spike_filename)
    except IOError:
        spiking_model = generate_spikes(params)
        # Save to file
        iotools.saveraw(spike_filename, spiking_model.s)
    else:
        logger.info("Precomputed data found. Skipping spike generation.")
        spiking_model = Spiketrain.from_raw(spikes_raw)

    spike_activity_filename = core.get_pathname(core.spikes_subdir, params, 'activity')
    if not os.path.exists(spike_activity_filename):
        logger.info("Computing activity from spike data")
        Ahist = core.compute_spike_activity(spiking_model.s)
        iotools.saveraw(spike_activity_filename, Ahist)

