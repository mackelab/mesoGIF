import os.path
import copy
import numpy as np
from collections import Iterable, OrderedDict
import glob

from parameters import ParameterSet
import theano_shim as shim
import sinn
import sinn.histories as histories
import sinn.optimize.gradient_descent as gd
from sinn.histories import Series, Spiketrain
import mackelab as ml
import mackelab.iotools
import mackelab.parameters

from fsGIF import core
from fsGIF.mcmc import get_pymc_model_new as get_pymc_model  # FIXME: Move function to avoid sibling import
logger = core.logger
############################
# Model import
from fsGIF import fsgif_model as gif
data_dir = "data"
label_dir = "run_dump"
############################

# >>>FIXME<<<<<
# Current implementation constructs the computational graph for the cost at least 4 times.


def do_gradient_descent(params, prev_run=None):

    # Create the sgd object
    sgd = get_sgd_new(params, prev_run)

    # Iterate for the desired number of steps
    if sgd.step_i >= params.sgd.max_iterations:
        # TODO Add convergence check (don't compile if converged, even if step_i < max_iterations
        logger.info("Precomputed gradient descent found. Skipping fit.")
        return None, sgd.step_i

    # Compile the optimizer
    logger.info("Compiling {} optimizer...".format(params.optimizer))
    sgd.compile( lr = params.learning_rate )
    logger.info("Done.")

    # Do the fit
    logger.info("Starting gradient descent fit...")
    sgd.iterate(Nmax=params.sgd.max_iterations,
                cost_calc=params.cost_calc,
                **params.cost_calc_params)

    return sgd, sgd.step_i

def load_latest(pathname):
    # TODO: Move to RunMgr
    # First check if there are any previous runs
    _pathname, ext = os.path.splitext(pathname)
    prev_runs = glob.glob(_pathname + '_*iterations*' + ext)
    prev_runs = [run for run in prev_runs
                 if not core.isarchived(run)
                    and os.path.isfile(run)]
        # Filter out archived runs and directories
    if len(prev_runs) > 0:
        # There has been a previous run
        # Find the latest one
        def get_run_N(_pname):
            suffixes = core.get_suffixes(_pname)
            if 'iterations' in suffixes:
                numstr = suffixes['iterations']
                assert(numstr is not None)
            return int(numstr)
        latest = prev_runs[0]
        latest_N = get_run_N(latest)
        for fname in prev_runs[1:]:
            N = get_run_N(fname)
            if N > latest_N:
                latest = fname
                latest_N = N
    else:
        raise core.FileNotFound

    return ml.iotools.load(latest), latest

def get_model(params):
    global data_dir, label_dir

    params = ml.parameters.params_to_arrays(params)

    post_params = params.posterior

    if 'init_vals' not in params or params.init_vals is None:
        init_vals = None
    else:
        # TODO: check if init_vals is a list
        init_vals = params.init_vals

    #output_filename = core.get_pathname(core.likelihood_subdir, params)
    #data_filename = core.get_pathname(params.data.dir, params.data.params, params.data.name)
    sgd_filename = get_sgd_pathname(params, label='')
    data_filename = core.get_pathname(data_dir=data_dir,
                                      params=post_params.data.params,
                                      subdir=post_params.data.dir,
                                      suffix=post_params.data.name,
                                      label_dir=label_dir,
                                      label=''
                                      )
    input_filename = core.get_pathname(data_dir=data_dir,
                                       params=post_params.input.params,
                                       subdir=post_params.input.dir,
                                       suffix=post_params.input.name,
                                       label_dir=label_dir,
                                       label=''
                                       )
    data_filename = core.add_extension(data_filename)
    input_filename = core.add_extension(input_filename)


    # Load the data and model to fit. This is required whether or not
    # sgd was loaded from a previous run file
    data_history = ml.iotools.load(data_filename)
                    #cls=getattr(histories, post_params.data.type).from_raw,
                    #calc='activity',
                    #recalculate=False)
    if isinstance(data_history, np.lib.npyio.NpzFile):
        # Support older data files
        data_history = Series.from_raw(data_history)
    data_history = core.subsample(data_history, post_params.model.dt)
    data_history.lock()

    input_history = ml.iotools.load(input_filename)
                             #cls=getattr(histories, post_params.input.type).from_raw,
                             #calc='input',
                             #recalculate=False)
    if isinstance(input_history, np.lib.npyio.NpzFile):
        # Support older data files
        input_history = Series.from_raw(input_history)
    input_history = core.subsample(input_history, post_params.model.dt)
    input_history.lock()

    prior_sampler = core.get_sampler(post_params.model.prior)

    model = core.construct_model(gif, post_params.model, data_history, input_history,
                                 initializer=post_params.model.initializer)

    return model, prior_sampler

def get_sgd_new(params, prev_run=None):
    model, prior_sampler = get_model(params)
    pymc_model, pymc_priors = get_pymc_model(params.posterior, model)
        # TODO: Put into get_model() when legacy is dropped

    if prev_run is None:
        def cost(tidx, batch_size):
            logL = model.loglikelihood(tidx, batch_size)[0]
            prior_logp = shim.sum([shim.cast_floatX(v.logpt) for v in pymc_model.vars])
                # FIXME: prior logp's still have dtype='float64',
                # no matter the value of floatX.
                # This is probably due to some internal constants
                # which are double precision.
            return logL + prior_logp
        # Get the variables to track
        # We track the non-transformed variables of the prior. For each variable,
        # we extract the corresponding PyMC3 symbolic quantity, which we transform
        # back.
        track_vars = OrderedDict(
            (name, prior.transform.back(pymc_model.named_vars[prior.transform.names.new]))
            for name, prior in pymc_priors.items() )
        start = model.t0idx + model.index_interval(params.posterior.burnin)
        datalen = model.index_interval(params.posterior.datalen)
        burnin = model.index_interval(params.sgd.batch_burnin)
        batch_size = model.index_interval(params.sgd.batch_size)

        if getattr(params.sgd, 'mode', None) == 'sequential':
            def model_initialize(t):
                model.initialize(t=0.)
                    # NOTE: <int> 0 is first time bin, no matter padding
                    #       <float> 0. is time bin of t0
                model.advance(t)
        else:
            def model_initialize(t):
                model.initialize(t=t)

        model.theano_reset() # TODO: deprecated ?
        model.clear_unlocked_histories()
        sgd = gd.SeriesSGD(
            cost = cost,
            cost_format = 'logL',
            optimize_vars = pymc_model.vars,
            track_vars = track_vars,
            advance = model.advance_updates,
            initialize = model_initialize,
            start = start,
            datalen = datalen,
            burnin = burnin,
            batch_size = batch_size,
            optimizer = params.sgd.optimizer,
            lr = params.sgd.learning_rate,
            mode = params.sgd.mode,
            cost_track_freq = params.sgd.cost_track_freq,
            var_track_freq = params.sgd.var_track_freq
        )
        model.clear_unlocked_histories()
        model.theano_reset() # TODO: deprecated ?

    return sgd

def get_sgd(mgr):
    get_sgd_legacy(mgr.params)

def get_sgd_legacy(params, prev_run=None):

    model, prior_sampler = get_model(params)

    post_params = params.posterior
    prior_descs = params.posterior.model.prior

    fitmask = get_fitmask(model, post_params.mask)

    if prev_run is None:
        # For a new run, a new sgd instance must be created and initialized

        sgd = gd.SGD_old(
            cost = model.loglikelihood,
            cost_format = 'logL',
            optimizer = params.optimizer,
            model = model,
            start = params.posterior.burnin,
            datalen = params.posterior.datalen,
            burnin = params.batch_burnin,
            mbatch_size = params.batch_size
        )
        sgd.set_fitparams(fitmask)

        transform_descs = [desc.transform for desc in prior_descs.values() if hasattr(desc, 'transform')]
        # TODO: Change SGD to use the TransformedVar class, than remove the following
        transforms = []
        for desc in transform_descs:
            origname, newname = desc.name.split('->')
            transforms.append([origname.strip(), newname.strip(), desc.to, desc.back])
        for transform_desc in transforms:
            try:
                var = sgd.get_param(transform_desc[0])
            except KeyError:
                pass
            else:
                sgd.transform(var, *transform_desc[1:])
        # for time_cst in [getattr(model.params, tc_name) for tc_name in post_params.time_constants]:
        #     if time_cst in fitmask:
        #         sgd.transform( time_cst, 'log' + time_cst.name,
        #                        'τ -> shim.log10(τ)', 'logτ -> 10**logτ' )

        sgd.verify_transforms(trust_automatically=True)
            # FIXME: Use TransformedVar, and then this whole verification thing won't be needed

        # If the parameters which generated the data are known, set them as ground truth
        if ( 'params' in post_params.data and isinstance(post_params.data, ParameterSet)
             and 'model' in post_params.data.params
             and isinstance(post_params.data.params, ParameterSet) ):
            #trueparams = core.get_model_params(post_params.data.params.model,
            #                                   post_params.model.type)
            # >>>> HACK <<<<<< Above doesn't work e.g. for hetero parameters, which
            #                  aren't compatible with the meso model
            trueparams = core.get_model_params(post_params.model.params,
                                               post_params.model.type)
            sgd.set_ground_truth(trueparams)

        if init_vals is not None:
            init_vals_format = getattr(init_vals, 'format', 'cartesian')
            init_vals_random = getattr(init_vals, 'random', False)
            if init_vals_random:
                if ( 'seed' in init_vals and init_vals.seed is not None ):
                    np.random.seed(init_vals.seed)
                logger.debug("RNG state: {}".format(np.random.get_state()[1][0]))

            if init_vals_format == 'prior':
                _init_vals_dict = {}
                # TODO: Checking for substitutions would be cleaner if using TransformedVar
                for name in post_params.variables:
                    var = sgd.get_param(name)
                    if var in sgd.substitutions:
                        _init_var = sgd.substitutions[var][0]
                        _init_vals_dict[_init_var] = prior_sampler(var)
                    else:
                        _init_vals_dict[var] = prior_sampler(var)

            elif init_vals_format == 'cartesian':
                _init_vals_dict = { sgd.get_param(pname): init_vals[pname]
                                    for pname in init_vals.variables }

            elif init_vals_format in ['polar', 'spherical']:

                # The total number of variables is the sum of each variable's number of elements
                curvals = OrderedDict( (varname, sgd.get_param(varname).get_value())
                                       for varname in init_vals.variables )
                nvars = sum( np.prod(var.shape) if hasattr(var, 'shape') else 1
                             for var in curvals.values() )

                # Get the coordinate angles
                if init_vals_random:
                    # All angles except last [0, π)
                    # Last angle [0, 2π)
                    angles = np.uniform(0, np.pi, nvars - 1)
                    angles[-1] = 2*angles[-1]
                else:
                    # The angles may be given with nested structure; this is just to help
                    # legibility, so flatten everything.
                    angles = np.concatenate([np.array(a).flatten() for a in init_vals.angles])
                    if len(angles) != nvars - 1:
                        raise ValueError("Number of coordinate angles (currently {}) must be "
                                         "one less than the number of variables. (currently {})."
                                         .format(len(init_vals.angles), len(init_vals.variables)))

                # Compute point on the unit sphere
                sines = np.concatenate(([1], np.sin(angles)))
                cosines = np.concatenate((np.cos(angles), [1]))
                unit_vals_flat = np.cumprod(sines) * cosines
                # "unflatten" the coordinates
                unit_vals = []
                i = 0
                for name, val in curvals.items():
                    if shim.isscalar(val):
                        unit_vals.append(unit_vals_flat[i])
                        i += 1
                    else:
                        varlen = np.prod(val.shape)
                        unit_vals.append(unit_vals_flat[i:i+varlen].reshape(val.shape))
                        i += varlen

                # rescale coords
                radii = []
                for varname, var in curvals.items():
                    radius = init_vals.radii[varname]
                    if shim.isscalar(var):
                        radii.append(radius)
                    else:
                        if shim.isscalar(radius):
                            radii.append( np.ones(var.shape) * radius )
                        else:
                            if radius.shape != var.shape:
                                raise ValueError("The given radius has shape '{}'. It should "
                                                 "either be scalar, or of shape '{}'."
                                                 .format(radius.shape, var.shape))
                            radii.append(radius)
                rescaled_vals = [val * radius for val, radius in zip(unit_vals, radii)]

                # add the centre
                centre = []
                for varname, var in curvals.items():
                    comp = init_vals.centre[varname]
                    if shim.isscalar(var):
                        centre.append(comp)
                    else:
                        if shim.isscalar(comp):
                            centre.append( np.ones(var.shape) * comp )
                        else:
                            if comp.shape != var.shape:
                                raise ValueError("The given centre component has shape '{}'. It should "
                                                 "either be scalar, or of shape '{}'."
                                                 .format(comp.shape, var.shape))
                            centre.append(comp)
                _init_vals_values = [c + r for c, r in zip(centre, rescaled_vals)]

                # construct the initial value dictionary
                _init_vals_dict = { sgd.get_param(name): val
                                    for name, val in zip(curvals.keys(),
                                                         _init_vals_values) }

            sgd.initialize(_init_vals_dict, fitmask)
                # Specifying fitmask ensures that parameters which
                # are not being fit are left at the ground truth value

    else:
        # Create the SGD from the loaded raw data
        sgd = gd.SGD_old(
            cost = model.loglikelihood,
            cost_format = 'logL',
            optimizer = params.optimizer,
            model = model,
            sgd_file = prev_run,
            set_params = False  # Have to call verify_transforms first
                # FIXME: Use default (True) once verify_transforms no longer needed
        )
        sgd.verify_transforms(trust_automatically=True)
        sgd.set_params_to_evols()

    # SGD instance has been either created or reloaded
    # It's not compiled yet: caller may find that the fit is already done, and not need to be compiled

    return sgd

def get_fitmask(model, mask_desc):
    return { getattr(model.params, name) : mask for name, mask in mask_desc.items() }

def get_param_hierarchy(params):
    """
    Return a list of mutually exclusive ParameterSets, each subsequent set corresponding to
    a nested directory.
    """
    # SGD parameters
    sgd_params = copy.deepcopy(params.sgd)
    sgd_params.pop('max_iterations', None)  # Don't use max_iterations for filename
    # Initial parameter values
    init_params = copy.deepcopy(params.init_vals)
    # Dataset parameters
    data_keys = ['data', 'input', 'model']
    data_params = copy.deepcopy(
        ParameterSet({key: params.posterior[key] for key in data_keys}))

    return [data_params, sgd_params, init_params]

def get_previous_run(params, resume=True):
    # Now load the sgd file
    sgd_filename = get_sgd_pathname(params, label='')
    try:
        prev_run, _ = load_latest(sgd_filename)
    except (core.FileNotFound, core.FileRenamed):
        # There are no previous runs
        prev_run = None
    else:
        if not resume:
            # Don't resume: just reload the previous run
            prev_run = None

def get_sgd_pathname(params, iterations=None, **kwargs):
    """
    Calculate the filename with a reduced parameter set, where the parameters
    defining the number of iterations are removed.
    This allows continuing a fit on a subsequent run.
    Pathnames are organized hierarchically as [fitdir]/[data]/[sgd]/[init_vals], where
    [init_vals] is the filename. [model], [sgd] and [init_vals] are computed by hashing
    the appropriate parameters.
    """
    global data_dir, label_dir

    suffix = kwargs.pop('suffix', '')
    if iterations is not None:
        assert(isinstance(iterations, int))
        suffix += '_iterations' + str(iterations)

    param_hierarchy = get_param_hierarchy(params)
    datahash = ml.parameters.get_filename(param_hierarchy[0])
    sgdhash = ml.parameters.get_filename(param_hierarchy[1])
    init_params = param_hierarchy[2]

    return core.get_pathname(data_dir=data_dir,
                             params=init_params,
                             subdir='+' + datahash + '/' + sgdhash,
                             suffix=suffix,
                             label_dir=label_dir,
                             **kwargs)

if __name__ == "__main__":
    core.init_logging_handlers()
    mgr = core.RunMgr(description="Gradient descent", calc='sgd')
    mgr.parser.add_argument('--resume', action='store_false',
                            help='Indicates whether to resume iterating the gradient descent '
                              'from a previous run, if one exists, or start a new fit. '
                              'Default is to continue from an existing fit.')
    mgr.load_parameters()

    shim.load_theano()

    # Check if we can continue a previous run
    resume = getattr(mgr.args, 'resume', True)
    prev_run = get_previous_run(mgr.params, resume)

    #sgd, n_iterations = do_gradient_descent(mgr.params, prev_run)

    # Load the gradient descent class
    sgd = get_sgd_new(mgr.params, prev_run)

    # Check if the fit has already been done
    if sgd.step_i >= mgr.params.sgd.max_iterations:
        # TODO Add convergence check (don't compile if converged, even if step_i < max_iterations
        skipped = True
        logger.info("Precomputed gradient descent found. Skipping fit.")
        #return None, sgd.step_i
    else:
        # Do the fit
        skipped = False
        logger.info("Starting gradient descent fit...")
        sgd.fit(Nmax=mgr.params.sgd.max_iterations)

    if not skipped:
        if mgr.args.debug:
            output_filename = 'gd_debug.npr'
        else:
            output_filename = get_sgd_pathname(mgr.params, sgd.step_i, label=None)
        ml.iotools.save(output_filename, sgd)
