import sys
import core

import parameters

############################
# Model import
import fsgif_model as gif
############################

"""
Expected parameters format:
{ 'input': [input_params],
  'spikes': [spike_params],
  'model: [model_params]
}
"""

def generate_spikes(params):
    # Temporarily unload Theano since it isn't supported by spike history
    use_theano = shim.config.use_theano
    shim.load(load_theano=False)

    rndstream = core.get_random_stream(params.seed)

    logger.info("Generating new spike data...")
    Ihist = Series.from_raw(iotools.loadraw(core.get_pathname(core.input_subdir, params.input)))

    # Create the spiking model
    # We check if different run parameters were specified,
    # otherwise those from Ihist will be taken
    runparams = { name: val for name in params.spikes
                   if name in ['t0', 'tn', 'dt'] }
    shist = Spiketrain(Ihist, name='s', pop_sized = params.model.N,
                       **runparams)
    spiking_model = init_spiking_model(params.model, shist, Ihist, rndstream)
    w = gif.GIF_spiking.expand_param(np.array(model_params.w), model_params.N) * model_params.Γ
        # w includes both w and Γ from Eq. 20
    shist.set_connectivity(w)


    # Generate the spikes
    shist.set()

    # Save to file
    iotools.saveraw(core.get_pathname(core.spikes_subdir, params))
    logger.info("Done.")

    # Reload Theano if it was loaded when we entered the function
    shim.load(load_theano=use_theano)


if __name__ == "__main__":
    params = core.load_parameters(sys.argv[1])
    generate_spikes(params)


