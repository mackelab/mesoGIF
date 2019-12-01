import os.path
import numpy as np
#from tqdm import tqdm
import multiprocessing as mp

import mackelab_toolbox as ml
import mackelab_toolbox.iotools

import theano_shim as shim
from sinn.histories import Series, Spiketrain

from fsGIF import core
logger = core.logger
############################
# Model import
from fsGIF import fsgif_model as gif
############################

def get_data(params):

    # Safer to redo imports in subprocess, when multiprocessing
    from fsGIF import core
    import mackelab as ml
    import mackelab.iotools
    data_dir = "data"

    # params = mgr.params
    # seed = params.seed

    input_pathname = core.get_pathname(data_dir = data_dir,
                                       params = params.input,
                                       subdir = core.RunMgr.subdirs['input'],
                                       label = '',
                                       suffix = '')
    input_pathname = core.add_extension(input_pathname)

    Ihist = ml.iotools.load(input_pathname)
    if isinstance(Ihist, np.lib.npyio.NpzFile):
        # Support older data files
        Ihist = Series.from_repr_np(Ihist)

    Ihist = core.subsample(Ihist, params.dt,
                           max_len = int(params.tn / params.dt))

    if params.t0 != Ihist.t0:
        raise ValueError("The input t0 must match that of the data to generate.")

    # spikedata_pathname = mgr.get_pathname(params = params.data.params,
    #                                       subdir = params.data.dir,
    #                                       suffix = '',
    #                                       label='')
    # spikedata_pathname = core.add_extension(spikedata_pathname)
    activitydata_pathname = core.get_pathname(data_dir = data_dir,
                                              params = params,
                                              subdir = core.RunMgr.subdirs['spikes'],
                                              suffix = 'activity',
                                              label = '')
    activitydata_pathname = core.add_extension(activitydata_pathname)

    # shist = ml.iotools.load(spikedata_pathname)
    spikeAhist = ml.iotools.load(activitydata_pathname)
    spikeAhist = core.subsample(spikeAhist, params.dt,
                                max_len = int(params.tn / params.dt))
    spikeAhist.lock()

    return Ihist, spikeAhist

def compute_nbar(params):
    """
    Parameters
    ----------
    params: ParameterSet
        ParameterSet used to generate the spikes.
    """

    # Safer to redo imports in subprocess, when multiprocessing
    from fsGIF import core
    from fsGIF import fsgif_model as gif
    from tqdm import tqdm

    Ihist, spikeAhist = get_data(params)

    # # Create the activity model
    # # We check if different run parameters were specified,
    # # otherwise those from Ihist will be taken
    # runparams = { name: params[name] for name in params
    #               if name in ['t0', 'tn', 'dt'] }

    model_params = core.get_model_params(params.model, 'GIF_mean_field')
        # Needed for now because fsgif_model does not yet use ParameterSet

    # Ahist = Series(Ihist, name='A_look-ahead', shape=(len(model_params.N),), iterative=True,
    #                **runparams)

    # GIF activity model
    mfmodel = gif.GIF_mean_field(model_params, spikeAhist, Ihist,
                                 params.initializer)
        # Don't need a random number generator, since we already have the activity trace

    #logger.info("Generating new activity data...")

    # Try to get the process index
    pname = mp.current_process().name
    pint = ''.join(c for c in pname if c.isnumeric())
    if len(pint) > 0:
        pidx = int(pint)
    else:
        pidx = 0

    # Generate the activity trace
    # We could just call mfmodel.advance('end'), but doing it sequentially allows the progress bar
    # And since we know that variables have to be computed iteratively anyway, there's not much
    # cost to doing this.
    for i in tqdm(range(mfmodel.t0idx, mfmodel.tnidx), position=pidx):
        mfmodel.advance(i)

    return mfmodel

def add_suffix(filename, suffix, sep='_'):
    """Add a suffix before the file extension."""
    base, ext = os.path.splitext(filename)
    if sep != '' and base[-1] != sep and suffix[0] != sep:
        base += sep
    return base + suffix + ext

if __name__ == "__main__":
    core.init_logging_handlers()
    mgr = core.RunMgr(description="Generate activity", calc='spike_nbar')
    mgr.load_parameters()
    nbar_filename = mgr.get_pathname(label='')

    try:
        ml.iotools.load(nbar_filename)#, cls=Series.from_raw)
    except FileNotFoundError:
        # Get pathname with run label
        if mgr.args.debug:
            nbar_filename = "spike-nbar_debug.npr"
            #expected_nbar_filename = "activity_debug_nbar.npr"
        else:
            nbar_filename = core.add_extension(mgr.get_pathname(label=None))
            #expected_nbar_filename = core.add_extension(mgr.get_pathname(label=None, suffix='nbar'))
        # Create mean-field model and generate activity
        #threadidx = getattr(mgr.args, 'threadidx', 0)
        mfmodel = compute_nbar(mgr.params)
        # Save to file
        iotools.save(add_suffix(nbar_filename, 'nbar'), mfmodel.nbar, format='npr')
        #iotools.save(add_suffix(nbar_filename, 'u'), mfmodel.u, format='npr')
        #iotools.save(add_suffix(nbar_filename, 'vartheta'), mfmodel.varθ, format='npr')

