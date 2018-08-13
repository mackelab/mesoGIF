import os.path
import copy
import numpy as np
from collections import Iterable, OrderedDict
import glob

from parameters import ParameterSet
import pymc3 as pm
import theano_shim as shim
import sinn
import sinn.histories as histories
import sinn.optimize.gradient_descent as gd
from sinn.histories import Series, Spiketrain
import mackelab as ml
import mackelab.iotools
import mackelab.parameters
import mackelab.pymc3

ml.parameters.Transform.namespaces.update({'shim': shim})

from fsGIF import core
#from fsGIF.mcmc import get_pymc_model_new as get_pymc_model  # FIXME: Move function to avoid sibling import
logger = core.logger
############################
# Model import
from fsGIF import fsgif_model as gif
data_dir = "data"
label_dir = "run_dump"
############################

debugprint = False
gif.debugprint = debugprint


# import mackelab.optimizers
# mackelab.optimizers.debug_flags['nanguard'] = True
# mackelab.optimizers.debug_flags['print grads'] = True

# >>>FIXME<<<<<
# Current implementation constructs the computational graph for the cost at least 4 times.


# def do_gradient_descent(params, prev_run=None):
#
#     # Create the sgd object
#     sgd = get_sgd_new(params, prev_run)
#
#     # Iterate for the desired number of steps
#     if sgd.step_i >= params.sgd.max_iterations:
#         # TODO Add convergence check (don't compile if converged, even if step_i < max_iterations
#         logger.info("Precomputed gradient descent found. Skipping fit.")
#         return None, sgd.step_i
#
#     # Compile the optimizer
#     logger.info("Compiling {} optimizer...".format(params.optimizer))
#     sgd.compile( lr = params.learning_rate )
#     logger.info("Done.")
#
#     # Do the fit
#     logger.info("Starting gradient descent fit...")
#     sgd.iterate(Nmax=params.sgd.max_iterations,
#                 cost_calc=params.cost_calc,
#                 **params.cost_calc_params)
#
#     return sgd, sgd.step_i

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
    # TODO: Use core.get_meso_model

    global data_dir, label_dir

    params = ml.parameters.params_to_arrays(params)

    post_params = params.posterior.copy()

    # HACK because we changed spike.params format; reverts to old format if
    # input is a parameterized history function
    # if 'type' in post_params.data.params.input:
    #     post_params.data.params.input = post_params.data.params.input.params
    # END HACK

    # if 'init_vals' not in params or params.init_vals is None:
    #     init_vals = None
    # else:
    #     # TODO: check if init_vals is a list
    #     init_vals = params.init_vals

    #output_filename = core.get_pathname(core.likelihood_subdir, params)
    #data_filename = core.get_pathname(params.data.dir, params.data.params, params.data.name)
    # sgd_filename = get_sgd_pathname(params, label='')
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

    # prior_sampler = core.get_sampler(post_params.model.prior)

    model = core.construct_model(gif, post_params.model, data_history, input_history,
                                 initializer=post_params.model.initializer)

    return model #, prior_sampler

def get_prior_params(params):
    varnames = getattr(params, 'variables',
                       list(params.mask.keys()))
    varnames = list(varnames)
        # Easier to dynamically remove elements from a list than an array
    masks = getattr(params, 'mask', {})
    #priorparams = params.model.prior
    priorparams = ParameterSet(
        {key: value for key, value in params.model.prior.items()
         if key in varnames})
    for varname in varnames[:]:
        if varname in masks:
            assert(varname in priorparams)
            mask = masks[varname]
            if not np.any(mask):
                varnames.remove(varname)
                del masks[varname]
                del priorparams[varname]
            else:
                priorparams[varname].mask = mask

    return priorparams

def get_model_vars(params, model, prune=None):
    """
    Return the list of parameters in `model` which are defined in `params`.
    The order of the list is determined by `params.variables`.
    If `prune` is given, it should be either a list of strings or of symbolic
    variables; only parameters whose name matches thate of one of the elements
    of `prune` are returned.
    """
    varnames = getattr(params, 'variables',
                       list(params.mask.keys()))
    modelvars = [getattr(model.params, varname) for varname in varnames]
    if prune is not None:
        # Only return vars which match one of the elements in prune.
        # Replace symbolic variables by their names
        prune = [getattr(v, 'name', v) for v in prune]
        modelvars = [v for v in modelvars if v.name in prune]
    return modelvars

def get_pymc_model(params, model, batch_size):
    """

    Parameters
    ----------
    params: ParameterSet (posterior format)
    model:  Sinn Model
    batch_size: int
        Size of batches.
        FIXME: This is only required here because PyMC3 requires an int for the
        data shape. If we could use a symbolic, than this wouldn't be needed.
    """
    # TODO: Use pymc3.data.Minibatch for the n data
    priorparams = get_prior_params(params)
        # Only returns variables which are not completely masked
    modelvars = get_model_vars(params, model, prune=priorparams)
        # `prune` removes variables which don't have a matching priorparam

    burnin_idx = model.index_interval(params.burnin)
    datalen = model.index_interval(params.datalen)
    def setup():
        model.clear_unlocked_histories()
        model.initialize(params.model.initializer)
        model.advance(model.t0idx + burnin_idx)

    with ml.pymc3.InitializableModel(setup=setup) as pymc_model:
        priors = ml.pymc3.PyMCPrior(priorparams, modelvars)
        start = shim.symbolic.scalar('tidx', dtype=model.n.tidx_dtype)
        batch_size_var = shim.symbolic.scalar('batch_size', dtype=model.n.tidx_dtype)
        start.tag.test_value = 1
            # Must be large enough so that test_value slices are not empty
        batch_size_var.tag.test_value = 2
        logL_model = model.loglikelihood(start, batch_size_var, avg=False)[0]
        logL = shim.graph.clone(logL_model, priors.subs) / batch_size_var
            # Use average logL, so increasing batch size actually decreases variance
        def logL_fn(data):
            # Since logL depends on data before the minibatch, we can't just
            # compute the log-likelihood from `data`. Instead it depends on
            # `start` and `batch_size_var`, which we change ourselves.
            return logL
        batch = model.n[start:start+batch_size_var]
        # FIXME: Shape cannot be symbolic, so batch sizes are fixed
        #shape = shim.concatenate((shim.add_axes(batch_size_var), model.n.shape))
        #batch.tag.test_value = np.zeros(shape.tag.test_value, dtype=batch.dtype)
        shape = (batch_size,) + model.n.shape
        batch.tag.test_value = np.zeros(shape, dtype=batch.dtype)
        total_size = (datalen,)+model.n.shape
        total_size = tuple(np.asscalar(s) for s in total_size)
            # total_size must have pure Python types, not np.int_
        n = pm.DensityDist('n', logp=logL_fn, shape=shape,
                           dtype=model.n.dtype, testval=batch,
                           observed=batch, total_size=total_size)

    return pymc_model, priors, start, batch_size_var

def get_sgd(params, model, pymc_model, start_var, batch_size_var):

    # def cost(tidx, batch_size):
    #     logL = model.loglikelihood(tidx, batch_size)[0]
    #         # Can't use PyMC3 model here because we need cost on mini-batch
    #     prior_logp = shim.sum([shim.cast_floatX(v.logpt) for v in pymc_model.vars])
    #         # FIXME: prior logp's still have dtype='float64',
    #         # no matter the value of floatX.
    #         # This is probably due to some internal constants
    #         # which are double precision.
    #     return logL + prior_logp

    # Get the variables to track
    # We track the non-transformed variables of the prior. For each variable,
    # we extract the corresponding PyMC3 symbolic quantity, which we transform
    # back.
    track_vars = OrderedDict(
        (name, prior.transform.back(pymc_model.named_vars[prior.transform.names.new]))
         for name, prior in pymc_priors.items() )
    if not debugprint:
        start = model.t0idx + model.index_interval(params.posterior.burnin)
        datalen = model.index_interval(params.posterior.datalen)
        burnin = model.index_interval(params.sgd.batch_burnin)
        batch_size = model.index_interval(params.sgd.batch_size)
    else:
        # Replace the cost by a cost over a single step and a set starting
        # point. By fixing the start and length of the batch, it allows to
        # compare with a cost computed externally.
        # The batch start point is set to the value of `debugprint` and can
        # be changed with `self.sstart.set_value(np.int32())`.
        start = shim.shared(np.int32(0))  # Attached to SGD below
        datalen = np.int32(1)
        burnin = np.int32(0)
        batch_size = np.int32(1)

    if getattr(params.sgd, 'mode', None) == 'sequential':
        def model_initialize(t):
            model.initialize(t=0.)
                # NOTE: <int> 0 is first time bin, no matter padding
                #       <float> 0. is time bin of t0
            model.advance(t)
    else:
        def model_initialize(t):
            model.initialize(t=t)

    if params.sgd.cost in ('loglikelihood', 'logL', 'log likelihood', 'log L'):
        cost = shim.cast_floatX(pymc_model.n.logpt)
    elif params.sgd.cost == 'posterior':
        cost = shim.cast_floatX(pymc_model.logpt)
    else:
        raise ValueError("Unrecognized cost descriptor '{}'"
                         .format(params.sgd.cost))

    # model_reset is probably not required
    def model_reset(**kwargs):
        values = copy.copy(kwargs)
        # HACK to invert transformations
        for name, v in kwargs.items():
            if name[:3] == 'log':
                nwname = name[3:]; assert(nwname not in values)
                values[nwname] = 10**v
                del values[name]
            if name[-6:] == '_log__':
                nwname = name[:-6]; assert(nwname not in values)
                values[nwname] = shim.exp(v)
                del values[name]
        return model.update_params(values)

    model.theano_reset() # TODO: deprecated ?
    model.clear_unlocked_histories()
    sgd = gd.SeriesSGD(
        # cost = shim.cast_floatX(pymc_model.logpt),
        cost = pymc_model.logpt,
            # FIXME: prior logp's still have dtype='float64', no matter the
            # value of floatX. This is probably due to some internal
            # constants which are double precision.
        start_var = start_var,
        batch_size_var = batch_size_var,
        cost_format = 'logL',
        optimize_vars = pymc_model.vars,
        track_vars = track_vars,
        advance = model.advance_updates,
        reset = None,
        initialize = model_initialize,
        start = start,
        datalen = datalen,
        burnin = burnin,
        batch_size = batch_size,
        optimizer = params.sgd.optimizer,
        optimizer_kwargs = params.sgd.optimizer_kwargs,
        # lr = params.sgd.learning_rate,
        mode = params.sgd.mode,
        mode_params = params.sgd.get('mode_params', None),
        cost_track_freq = params.sgd.cost_track_freq,
        var_track_freq = params.sgd.var_track_freq
    )
    model.clear_unlocked_histories()
    model.theano_reset() # TODO: deprecated ?

    if debugprint:
        sgd.dbgstart = start  # Handle to help debugging

    return sgd

def old_get_param_hierarchy(params):
    sgd_params = copy.deepcopy(params)
    # Parameters we never care about saving
    del sgd_params['max_iterations']
    # Initial value parameters - deepest level
    init_params = sgd_params.init_vals
    del sgd_params['init_vals']
    # Dataset parameters
    data_keys = ['data', 'input', 'model']
    post_params = getattr(sgd_params, 'posterior', sgd_params)
        # Really old scripts didn't have a 'posterior' key
    data_params = ParameterSet({key: post_params[key] for key in data_keys})
    for key in data_keys:
        del post_params[key]

    return [data_params, sgd_params, init_params]

def get_param_hierarchy(params):
    """
    Return a list of mutually exclusive ParameterSets, each subsequent set corresponding to
    a nested directory.
    """
    if 'sgd' not in params:
        return old_get_param_hierarchy(params)
    else:
        # SGD parameters
        sgd_params = copy.deepcopy(params.sgd)
        sgd_params.pop('max_iterations', None)  # Don't use max_iterations for filename
        # Initial parameter values
        init_params = copy.deepcopy(params.init_vals)
        # Dataset parameters
        data_keys = ['data', 'input', 'model', 'mask']
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

def get_init_vals(params, pymc_model, priors):
    init_params = params.init_vals
    init_vals_format = getattr(init_params, 'format', 'fixed')
    init_vals_random = getattr(init_params, 'random', False)
    # if init_vals_random:
    #     if ( 'seed' in init_params and init_params.seed is not None ):
    #         np.random.seed(init_params.seed)
    #     logger.debug("RNG state: {}".format(np.random.get_state()[1][0]))

    init_vals = {}

    if init_vals_format in ('prior', 'dist'):
        # TODO: Only construct sampler once when function is called multiple times
        seed = getattr(init_params, 'seed', None)
        if init_vals_format == 'prior':
            dist_params = params.posterior.model.prior
        else:
            assert(init_vals_format == 'dist')
            dist_params = init_params.dist
        sampler = ml.parameters.ParameterSetSampler(dist_params, seed)
        sampled_init_vals = sampler.sample(priors.keys())
        # Cast to floatX
        sampled_init_vals = {key: shim.cast_floatX(val)
                     for key, val in sampled_init_vals.items()}
        # Attach the sampled variables to their corresponding variables –
        # these are the variables created for the prior.
        for name, val in sampled_init_vals.items():
            prior = priors[name]
            # Get the transformed variable, which is the one used in the cost graph.
            # If there is no transform these are no-ops
            name = prior.transform.names.new
            val = prior.transform.to(val)
            try:
                pymc_var = getattr(pymc_model, name)
            except AttributeError:
                # Sampler returns values for all parameters, including fixed ones.
                # Fixed parameters have no PyMC3 variables
                # TODO: Should we still initialize these parameters ?
                pass
            else:
                if isinstance(pymc_var, pm.model.TransformedRV):
                    val = pymc_var.transformation.forward(val).eval()
                        # PyMC3 transformations use theano.tensor operations
                    pymc_var = pymc_var.transformed
                if prior.mask is None:
                    mask = np.ones(val.shape, dtype=bool)
                        # Basically just serves to flatten `val`
                else:
                    mask = prior.mask
                init_vals[pymc_var] = val[mask]

    elif init_vals_format in ['fixed', 'cartesian']:
        raise NotImplementedError
        _init_vals_dict = { sgd.get_param(pname): init_params[pname]
                            for pname in init_params.variables }

    elif init_vals_format in ['polar', 'spherical']:
        raise NotImplementedError

        # The total number of variables is the sum of each variable's number of elements
        curvals = OrderedDict( (varname, sgd.get_param(varname).get_value())
                               for varname in init_params.variables )
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
            angles = np.concatenate([np.array(a).flatten() for a in init_params.angles])
            if len(angles) != nvars - 1:
                raise ValueError("Number of coordinate angles (currently {}) must be "
                                 "one less than the number of variables. (currently {})."
                                 .format(len(init_params.angles), len(init_params.variables)))

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
            radius = init_params.radii[varname]
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
            comp = init_params.centre[varname]
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

    return init_vals


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
                             subdir='fits/' + datahash + '/' + sgdhash,
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

    shim.config.floatX = mgr.params.sgd.floatX
    sinn.config.set_floatX()  # FIXME: Workaround because it sets sinn.config.rel_tolerance
    shim.load_theano()
    shim.gettheano().config.compute_test_value = 'raise'

    # Check if we can continue a previous run
    resume = getattr(mgr.args, 'resume', True)
    prev_run = get_previous_run(mgr.params, resume)

    #sgd, n_iterations = do_gradient_descent(mgr.params, prev_run)

    # Load the gradient descent class
    model = get_model(mgr.params)
    # Use double precision for priors. Avoids failures when sampling from tails
    #shim.config.floatX = 'float64'
    pymc_model, pymc_priors, start_var, batch_size_var = \
        get_pymc_model(mgr.params.posterior, model, mgr.params.sgd.batch_size)
    if prev_run is None:
        sgd = get_sgd(mgr.params, model, pymc_model, start_var, batch_size_var)

    # Check if the fit has already been done
    skipped = False
    if sgd.step_i >= mgr.params.sgd.max_iterations:
        # TODO Add convergence check (don't compile if converged, even if step_i < max_iterations
        skipped = True
        logger.info("Precomputed gradient descent found. Skipping fit.")
        #return None, sgd.step_i
    else:
        # Do the fit
        init_vals = get_init_vals(mgr.params,
                                  pymc_model,
                                  pymc_priors)
        sgd.initialize_vars(init_vals)
        logger.info("Starting gradient descent fit...")
        if 'floatX' in mgr.params:
            shim.config.floatX = mgr.params.floatX
        sgd.fit(Nmax=mgr.params.sgd.max_iterations, threadidx=mgr.args.threadidx)

    if not skipped:
        if mgr.args.debug:
            output_filename = 'gd_debug.npr'
        else:
            output_filename = get_sgd_pathname(mgr.params, sgd.step_i,
                                               label=mgr.label)

        ml.iotools.save(output_filename, sgd)
