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
import sinn.iotools as iotools

import core
from core import logger
############################
# Model import
import fsgif_model as gif
############################

def do_gradient_descent(mgr):

    # Create the sgd object
    sgd = get_sgd(mgr)

    # Iterate for the desired number of steps

    logger.info("Starting gradient descent fit...")
    sgd.iterate(Nmax=mgr.params.max_iterations,
                cost_calc=mgr.params.cost_calc,
                **mgr.params.cost_calc_params)

    return sgd, sgd.step_i

def get_sgd(mgr, check_previous_runs=True):
    params = mgr.params

    if 'init_vals' not in params or params.init_vals is None:
        init_vals = None
    else:
        # TODO: check if init_vals is a list
        init_vals = params.init_vals

    #output_filename = core.get_pathname(core.likelihood_subdir, params)
    #data_filename = core.get_pathname(params.data.dir, params.data.params, params.data.name)
    sgd_filename = get_sgd_pathname(mgr, label='')
    data_filename = mgr.get_pathname(params=params.data.params,
                                     subdir=params.data.dir,
                                     suffix=params.data.name,
                                     label='')
    input_filename = mgr.get_pathname(params=params.input.params,
                                      subdir=params.input.dir,
                                      suffix=params.input.name,
                                      label='')

    if check_previous_runs:
        # First check if there are any previous runs
        pathname, ext = os.path.splitext(sgd_filename)
        prev_runs = glob.glob(pathname + '_*iterations*' + ext)
        prev_runs = [run for run in prev_runs if not core.isarchived(run)] # Filter out archived runs
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

            # Now load the sgd file
            try:
                sgdraw = mgr.load(latest)
            except (core.FileDoesNotExist, core.FileRenamed):
                new_run = True
            else:
                new_run = False

            try:
                if not mgr.args.resume:
                    # Don't resume: just reload the previous run
                    return None, latest_N
            except AttributeError:
                # Default is not to resume
                return None, latest_N
        else:
            # There are no previous runs
            new_run = True
    else:
        new_run = True


    # Load the data and model to fit. This is required whether or not
    # sgd was loaded from a previous run file
    data_history = mgr.load(data_filename,
                    cls=getattr(histories, params.data.type).from_raw,
                    calc='activity',
                    recalculate=False)
    data_history = core.subsample(data_history, mgr.params.model.dt)
    data_history.lock()

    input_history = mgr.load(input_filename,
                             cls=getattr(histories, params.input.type).from_raw,
                             calc='input',
                             recalculate=False)
    input_history = core.subsample(input_history, mgr.params.model.dt)
    input_history.lock()

    prior_sampler = core.get_sampler(params.model.prior)

    model = core.construct_model(gif, params.model, data_history, input_history,
                                 initializer=params.model.initializer)

    fitmask = get_fitmask(model, mgr.params)

    if new_run:
        # For a new run, a new sgd instance must be created and initialized

        sgd = gd.SGD(
            cost = model.loglikelihood,
            optimizer = params.optimizer,
            model = model,
            start = params.start,
            datalen = params.datalen,
            burnin = params.burnin,
            mbatch_size = params.batch_size
        )
        sgd.set_fitparams(fitmask)

        for transform_desc in params.transforms:
            try:
                var = sgd.get_param(transform_desc[0])
            except KeyError:
                pass
            else:
                sgd.transform(var, *transform_desc[1:])
        # for time_cst in [getattr(model.params, tc_name) for tc_name in params.time_constants]:
        #     if time_cst in fitmask:
        #         sgd.transform( time_cst, 'log' + time_cst.name,
        #                        'τ -> shim.log10(τ)', 'logτ -> 10**logτ' )

        sgd.verify_transforms(trust_automatically=True)
            # FIXME: Use simple eval, and then this whole verification thing won't be needed

        # If the parameters which generated the data are known, set them as ground truth
        if ( 'params' in params.data and isinstance(params.data, ParameterSet)
             and 'model' in params.data.params
             and isinstance(params.data.params, ParameterSet) ):
            sgd.set_ground_truth(core.get_model_params(params.data.params.model))

        if init_vals is not None:
            if init_vals.random:
                if ( 'seed' in init_vals and init_vals.seed is not None ):
                    np.random.seed(init_vals.seed)
                logger.debug("RNG state: {}".format(np.random.get_state()[1][0]))

            if init_vals.format == 'prior':
                _fitparams_lst = [ sgd.get_param(name) for name in init_vals.variables ]
                _init_vals_dict = { p: prior_sampler(p) for p in _fitparams_lst }

            elif init_vals.format == 'cartesian':
                _init_vals_dict = { sgd.get_param(pname): init_vals[pname]
                               for pname in init_vals.variables }

            elif init_vals.format in ['polar', 'spherical']:

                # The total number of variables is the sum of each variable's number of elements
                curvals = OrderedDict( (varname, sgd.get_param(varname).get_value())
                                       for varname in init_vals.variables )
                nvars = sum( np.prod(var.shape) if hasattr(var, 'shape') else 1
                             for var in curvals.values() )

                # Get the coordinate angles
                if init_vals.random:
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
        sgd = gd.SGD(
            cost = model.loglikelihood,
            optimizer = params.optimizer,
            model = model,
            sgd_file = sgdraw,
            set_params = False  # Have to call verify_transforms first
                # FIXME: Use default (True) once verify_transforms no longer needed
        )
        sgd.verify_transforms(trust_automatically=True)
        sgd.set_params_to_evols()

    # SGD instance has been either created or reloaded
    # Now compile the optimizer

    logger.info("Compiling {} optimizer...".format(params.optimizer))
    sgd.compile( lr = params.learning_rate )
    logger.info("Done.")

    # Finally return the ready-to-go sgd instance

    return sgd

def get_fitmask(model, fit_params):
    return { getattr(model.params, name) : mask for name, mask in fit_params.fitmask.items() }

def get_param_hierarchy(mgr):
    """
    Return a list of mutually exclusive ParameterSets, each subsequent set corresponding to
    a nested directory.
    """
    sgd_params = copy.deepcopy(mgr.params)
    # Parameters we never care about saving
    del sgd_params['max_iterations']
    # Initial value parameters - deepest level
    init_params = sgd_params.init_vals
    del sgd_params['init_vals']
    # Dataset parameters
    data_keys = ['data', 'input', 'model']
    data_params = ParameterSet({key: sgd_params[key] for key in data_keys})
    for key in data_keys:
        del sgd_params[key]

    return [data_params, sgd_params, init_params]


def get_sgd_pathname(mgr, iterations=None, **kwargs):
    """
    Calculate the filename with a reduced parameter set, where the parameters
    defining the number of iterations are removed.
    This allows continuing a fit on a subsequent run.
    Pathnames are organized hierarchically as [fitdir]/[data]/[sgd]/[init_vals], where
    [init_vals] is the filename. [model], [sgd] and [init_vals] are computed by hashing
    the appropriate parameters.
    """

    suffix = kwargs.pop('suffix', '')
    if iterations is not None:
        assert(isinstance(iterations, int))
        suffix += '_iterations' + str(iterations)

    param_hierarchy = get_param_hierarchy(mgr)
    datahash = mgr.get_filename(param_hierarchy[0])
    sgdhash = mgr.get_filename(param_hierarchy[1])
    init_params = param_hierarchy[2]

    return mgr.get_pathname(init_params,
                            subdir='+' + datahash + '/' + sgdhash,
                            suffix=suffix,
                            **kwargs)

if __name__ == "__main__":
    core.init_logging_handlers()
    mgr = core.RunMgr(description="Gradient descent", calc='sgd')
    mgr.parser.add_argument('--resume', action='store_true',
                            help='Indicate to resume iterating the gradient descent '
                              'from a previous run, if one can be found. Default is '
                              'to return a found previous run, without continuing.')
    mgr.load_parameters()

    shim.load_theano()

    sgd, n_iterations = do_gradient_descent(mgr)

    if sgd is not None:
        output_filename = get_sgd_pathname(mgr, n_iterations, label=None)
        iotools.saveraw(output_filename, sgd)
