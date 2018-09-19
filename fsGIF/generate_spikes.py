import os.path
import numpy as np
from tqdm import tqdm

import theano_shim as shim
import mackelab as ml
import mackelab.parameters
import mackelab.iotools as iotools
import sinn
from sinn.histories import Series, Spiketrain
import sinn.analyze as anlz

from fsGIF import core
logger = core.logger
############################
# Model import
from fsGIF import fsgif_model as gif
data_dir = "data"
label_dir = "run_dump"
############################

sinn.config.set_floatX()

"""
Expected parameters format:
{ 'input': {input_params},
  'model': {model_params},
  ['t0', 'tn', 'dt']
}
"""

def generate_spikes(mgr):
    # Temporarily unload Theano since it isn't supported by spike history
    use_theano = shim.config.use_theano
    shim.load(load_theano=False)

    params = mgr.params
    seed = params.seed
    rndstream = core.get_random_stream(seed)

    #Ihist = iotools.loadraw(mgr.get_pathname(params.input))
    # Ihist = mgr.load(input_filename,
    #                  calc='input',
    #                  #cls=Series.from_raw,
    #                  recalculate=False)
    input_filename = core.add_extension(
        core.get_pathname(data_dir = core.data_dir,
                         params=params.input.params,
                         subdir=params.input.dir,
                         suffix=params.input.name,
                         label_dir=core.label_dir,
                         label=''))
    Ihist = iotools.load(input_filename)
    if isinstance(Ihist, np.lib.npyio.NpzFile):
        # Support older data files
        Ihist = Series.from_raw(Ihist)

    # Create the spiking model
    # We check if different run parameters were specified,
    # otherwise those from Ihist will be taken
    runparams = { name: params[name] for name in params.keys()
                   if name in ['t0', 'tn', 'dt'] }
    # TODO: if dt different from Ihist, subsample Ihist
    shist = Spiketrain(Ihist, name='s', pop_sizes = params.model.N, iterative=True,
                       **runparams)
    if shist.t0 < Ihist.t0:
        raise ValueError("You asked to generate spikes starting from {}, but "
                         "the input only starts at {}."
                         .format(shist.t0, Ihist.t0))
    if shist.tn > Ihist.tn:
        raise ValueError("You asked to generate spikes up to {}, but the input "
                         "is only provided up {}.".format(shist.tn, Ihist.tn))
    model_params_sampler = ml.parameters.ParameterSetSampler(params.model)
    model_params = core.get_model_params(model_params_sampler.sample(),
                                         'GIF_spiking')
        # Needed for now because fsgif_model does not yet use ParameterSet
    # HACK: Casting to PopTerm should be automatic
    model_params = model_params._replace(
        τ_θ=shist.PopTerm(model_params.τ_θ),
        τ_m=shist.PopTerm(model_params.τ_m))
    spiking_model = gif.GIF_spiking(model_params,
                                    shist, Ihist,
                                    params.initializer,
                                    set_weights=True,
                                    random_stream=rndstream)
    #w = gif.GIF_spiking.expand_param(np.array(model_params.w), model_params.N) * model_params.Γ
    #    # w includes both w and Γ from Eq. 20
    #shist.set_connectivity(w)

    # Generate the spikes
    # We could just call mfmodel.advance('end'), but doing it sequentially allows the progress bar
    # And since we know that variables have to be computed iteratively anyway, there's not much
    # cost to doing this.
    logger.info("Generating new spike data...")
    #shist.set()
    for i in tqdm(range(spiking_model.t0idx, spiking_model.tnidx+1),
                  position=mgr.args.threadidx):
        spiking_model.advance(i)
    logger.info("Done.")

    # Reload Theano if it was loaded when we entered the function
    shim.load(load_theano=use_theano)

    return spiking_model

if __name__ == "__main__":
    core.init_logging_handlers()
    mgr = core.RunMgr(description="Generate spikes", calc='spikes')
    mgr.load_parameters()
    params = mgr.params
    spikename_params = params.copy()
    # HACK because we changed spike.params format; reverts to old format if
    # input is a parameterized history function
    # if 'type' in spikename_params.input:
    #     spikename_params.input = spikename_params.input.params
    # END HACK
    def get_filename(label, suffix):
        return core.add_extension(
            core.get_pathname(data_dir  = data_dir,
                              label_dir = core.label_dir,
                              params    = spikename_params,
                              subdir    = 'spikes',
                              suffix    = suffix,
                              label     = label) )

    spike_filename = get_filename(label='', suffix='')
    spike_activity_filename = get_filename(label='', suffix='activity')

    # Check if we are simulating a heterogeneous population
    # Definitely HACK-y: the model should not need the `homo` flag in the first place
    for pname, pval in mgr.params.model.items():
        if isinstance(pval, dict):
            # Homogeneous variables should not be distributions
            # Essentially we are making the assumption that heteregeneous populations
            # are never set by hand, but always through a sampler
            assert('shape' in pval) # Sanity check
            gif.homo = False
            break

    generate_data = False
    try:
        # Try to load data to see if it's already been calculated
        shist = ml.iotools.load(spike_filename)
    except (FileNotFoundError, core.FileNotFound):
        # FIXME: ml.iotools.load does not raise `core.FileNotFound`
        generate_data = True
    except core.FileRenamed:
        # FIXME Will never happen: ml.iotools.load does not raise FileRenamed
        # The --recalculate flag was passed and the original data file renamed
        # Do the same with the associated activity file
        generate_data = True
        activity_path = mgr.find_path(spike_activity_filename)
        if activity_path is not None:
            mgr.rename_to_free_file(activity_path)
                # Warning: If there are missing activity traces, the rename could
                #   suffix the spikes and activity with a different number, as in
                #   both cases the first free suffix is used
                # Warning #2: If the data filename is free, but not the filename
                #   of the derived data (e.g. because only the first was renamed),
                #   the derived data is NOT renamed
    if generate_data:
        # Get new filenames with the run label
        if mgr.args.debug:
            spike_filename = "shist_debug.npr"
            spike_activity_filename = "shist_activity_debug.npr"
            spike_a_filename = "shist_E_activity_debug.npr"
        else:
            spike_filename = get_filename(suffix='', label=mgr.label)
            spike_activity_filename = get_filename(suffix='activity',
                                                   label=mgr.label)
            spike_a_filename = get_filename(suffix='expected_activity',
                                            label=mgr.label)

        # Generate spikes
        model = generate_spikes(mgr)
        shist = model.s

        # Save to file
        iotools.save(spike_filename, shist, format='npr')

        # Compute the associated activity trace
        logger.debug("Computing activity from spike data")
        Ahist = core.compute_spike_activity(shist)
        iotools.save(spike_activity_filename, Ahist, format='npr')

        # Compute the expected activity (i.e. effective firing rate)
        logger.debug("Computing effective firing rate from spike data")
        # ahist = Series(Ahist, iterative=False)
        # slcs = shist.pop_slices
        # def afunc(t):
        #     λ = np.array(model.λ[t])
        #     a = np.empty(λ.shape[:-1] + ahist.shape)
        #     for i, slc in enumerate(shist.pop_slices):
        #         a[..., i] = λ[..., slc].mean(axis=-1)
        #     return a
        # ahist.set_update_function(afunc)
        # ahist.set()
        ahist = anlz.mean(model.λ, shist.pop_slices)
        iotools.save(spike_a_filename, ahist, format='npr')
    else:
        logger.info("Simulation already computed; skipping. Location: {}."
                    .format(spike_filename) )
