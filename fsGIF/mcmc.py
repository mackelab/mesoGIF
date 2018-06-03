import os.path
import numpy as np
import pymc3 as pymc
import mackelab as ml
from mackelab.pymc3 import PyMCPrior, export_multitrace
from mackelab.parameters import Transform
import mackelab.parameters
import mackelab.theano

from parameters import ParameterSet
import theano_shim as shim
import sinn
import sinn.histories as histories
import sinn.optimize.gradient_descent as gd
from sinn.histories import Series, Spiketrain
import sinn.iotools as iotools

from fsGIF import core
logger = core.logger
############################
# Model import
import fsGIF.fsgif_model as fsgif_model
from fsGIF import fsgif_model as gif
data_dir = "data"
############################

############################
# Exceptions
class ModelSpecificationError(Exception):
    pass

shim.load_theano()
sinn.config.load_theano()
    # Synchronizes sinn and theano config, notably the floatX setting
    # TODO: Move setting of floatX into shim
shim.gettheano().config.compute_test_value = 'raise'
#sinn.config.floatX = 'float32'
#shim.gettheano().config.floatX = sinn.config.floatX

Transform.namespaces.update({'shim': shim})

class nDist(pymc.distributions.Continuous):
    # FIXME Make logp() more robust
    # logp() takes n as input, but nothing ensures that n aligns with the index
    # of self.start.
    # We could consider providing only logp_sum (see distributions.Distribution.logp_sum)

    def __init__(self, start, batch_size, model, variables, **kwargs):
        """
        variables: dictionary of {model_var: PyMC3_var} pairs.
        """
        nvar = model.n._data
        super().__init__(nvar.get_value().shape, nvar.dtype, **kwargs)
        self.model = model
        self.start = start
        self.batch_size = batch_size
        self.variables = variables
        # Basic checks on the variables
        # This catches some errors which allows us to exit with a more friendly
        # error message than the one Theano would otherwise print
        shape_mismatch = [(key,val) for key, val in variables.items()
                          if key.broadcastable != val.broadcastable]
        if len(shape_mismatch) > 0:
            # TODO: Custom error type: ParameterError ?
            # TODO: Print name (e.g. 'row', 'column') instead of broadcast pattern when possible
            msg = "The following random variables don't match their shape in the model:"
            msg += "\n  [var]: [model broadcast pattern] vs [random variable broadcast pattern]\n"
            msg += '\n'.join(["  - {}: {} vs {}"
                              .format(key.name, key.broadcastable, val.broadcastable)
                              for key, val in shape_mismatch])
            raise ValueError(msg)

    def logp(self, n):
        model_graph = self.model.loglikelihood(self.start, self.batch_size)[0]
        logp = shim.graph.clone(model_graph, self.variables)
        # Sanity check on logp: ensure all variable names are unique
        varnames = [v.name for v in shim.graph.inputs([logp]) if v.name is not None]
        counts = {name: varnames.count(name) for name in set(varnames)}
        if any(c > 1 for c in counts.values()):
            names = [name for name, c in counts.items() if c > 1]
            namestr = "name" if len(names) == 1 else "names"
            raise ModelSpecificationError("There are multiple variables with the {} {}. "
                                          "Since PyMC3 calls Theano functions with keyword "
                                          "arguments, this may make these calls ambiguous. "
                                          "Please ensure all variable names are unique."
                                          .format(namestr, ', '.join(names)))
        return logp

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        if name is None:
            name = "n"
        return r"${} \sim \text{{fsGIF}}(…)$".format(name)

class Model(pymc.model.Model):
    """
    Add to PyMC3 models the ability to specify a setup
    function which is run every time just before any compiled
    function is called.

    Parameters
    ----------
    setup: callable
        Function taking no arguments. Will be called just before evaluating
        any compiled function.
    """
    def __init__(self, name='', model=None, theano_config=None, setup=None):
        self.setup = setup
        super().__init__(name=name, model=model, theano_config=theano_config)

    def makefn(self, outs, mode=None, *args, **kwargs):
        f = super().makefn(outs, mode, *args, **kwargs)
        def makefn_wrapper(*args, **kwargs):
            self.setup()
            return f(*args, **kwargs)
        return makefn_wrapper

def get_pymc_model(mgr, model):
    return get_pymc_model_new(mgr.params.posterior, model)

def get_pymc_model_new(params, model):
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

    modelvars = [getattr(model.params, varname) for varname in varnames]

    burnin_idx = model.index_interval(params.burnin)
    def setup():
        model.clear_unlocked_histories()
        model.initialize(params.model.initializer)
        model.advance(burnin_idx)

    with Model(setup=setup) as pymc_model:
        priors = PyMCPrior(priorparams, modelvars)
        ndata = model.n._data.get_value()   # temp: nDist should be generic Dist
        n = nDist('n', params.burnin, params.datalen,
                  model = model,
                  variables = {getattr(model.params, varname) : prior.pymc_var
                               for varname, prior in priors.items()},
                  observed = ndata)

    return pymc_model, priors

def run_mcmc(mgr, model):
    pymc_model, priors = get_pymc_model(mgr, model)
    kwds = mgr.params.sampler
    if 'start' not in kwds:
        start = get_mle_start(mgr.params.posterior, priors)
        if start is not None:
            kwds['start'] = start
    if 'trace' in kwds:
        # Load a trace to start from
        start_trace_params = kwds['trace']
        start_trace_params.chain = kwds.chain  # We must continue the same chain number
        start_trace_filename = mgr.get_pathname(params = start_trace_params,
                                                subdir = core.RunMgr.subdirs['mcmc'],
                                                suffix = '',
                                                label = '')
        start_trace_filename = "mcmc_debug.dill"
        kwds['trace'] = ml.pymc3.import_multitrace(ml.iotools.load(start_trace_filename))

    with pymc_model:
        trace = pymc.sample(**kwds)

    return trace

def get_mle_start(posterior_params, priors):
    dataroot = "data"
        # HACK/TODO: Get this from sumatra project
    filename = ml.parameters.get_filename(posterior_params) + '.repr'
    pathname = os.path.join(dataroot, "MLE", filename)
    try:
        mlestr = iotools.load(pathname)
    except FileNotFoundError:
        return None
    else:
        mles = ParameterSet(eval(mlestr, {'array': np.array}))
            # FIXME: Don't use eval()
        start = {}
        for prior in priors.values():
            mle = mles[prior.transform.names.orig]
            if prior.mask is not None:
                mle = mle[prior.mask]
            start[prior.transform.names.new] = prior.transform.to(mle)
        return start

def get_model(mgr):
    postparams = mgr.params.posterior
    data_filename  = mgr.get_pathname(params = postparams.data.params,
                                      subdir = postparams.data.dir,
                                      suffix = postparams.data.name,
                                      label = '')
    data_filename = core.add_extension(data_filename)
    # flat_params = mackelab.parameters.params_to_arrays(postparams.data.params).flatten()
    # f, _ = mackelab.iotools.get_free_file("debug_dump", bytes=False)
    # f.write('\n'.join(str(key) + ', ' + str(flat_params[key]) for key in sorted(flat_params)))
    # f.close
    input_filename = mgr.get_pathname(params = postparams.input.params,
                                      subdir = postparams.input.dir,
                                      suffix = postparams.input.name,
                                      label = '')
    input_filename = core.add_extension(input_filename)

    data_history = mgr.load(data_filename,
                            cls=getattr(histories, postparams.data.type).from_raw,
                            calc='activity',
                            recalculate=False)
    # TODO: cast data_history to float32 instead of relying on subsample
    data_history = core.subsample(data_history, postparams.model.dt)

    input_history = mgr.load(input_filename,
                             cls=getattr(histories, postparams.input.type).from_raw,
                             calc='input',
                             recalculate=False)
    # TODO: cast input_history to float32 instead of relying on subsample
    input_history = core.subsample(input_history, postparams.model.dt)

    model = core.construct_model(gif, postparams.model, data_history, input_history,
                                 initializer=postparams.model.initializer)
    return model

def get_model_new(params):
    """
    Parameters
    ----------
    params: ParameterSet
        Parameters describing the posterior
    """
    global data_dir
    data_filename = core.get_pathname(data_dir = data_dir,
                                      params = params.data.params,
                                      subdir = params.data.dir,
                                      suffix = params.data.name,
                                      label = '')
    data_filename = core.add_extension(data_filename)
    input_filename = core.get_pathname(data_dir = data_dir,
                                      params = params.input.params,
                                      subdir = params.input.dir,
                                      suffix = params.input.name,
                                      label = '')
    input_filename = core.add_extension(input_filename)

    data_history = ml.iotools.load(data_filename, input_format='npr')
    if isinstance(data_history, np.lib.npyio.NpzFile):
        # Support older data files
        data_history = Series.from_raw(data_history)
    data_history = core.subsample(data_history, params.model.dt)

    input_history = ml.iotools.load(input_filename, input_format='npr')
    if isinstance(input_history, np.lib.npyio.NpzFile):
        # Support older data files
        input_history = Series.from_raw(input_history)
    input_history = core.subsample(input_history, params.model.dt)

    model = core.construct_model(gif, params.model, data_history, input_history,
                                 initializer=params.model.initializer)
    return model

if __name__ == "__main__":
    core.init_logging_handlers()

    mgr = core.RunMgr(description="MCMC sampler", calc='mcmc')
    # TODO: mgr.parser.add_argument('--resume' ...

    mgr.load_parameters()

    if ml.theano.using_gpu():
        logger.info("Theano using GPU")
    else:
        logger.info("Theano using only CPU")

    model = get_model(mgr)
    trace = run_mcmc(mgr, model)

    if mgr.args.debug:
        mcmc_filename = "mcmc_debug.dill"
    else:
        mcmc_filename = mgr.get_pathname(label=None)
    ml.iotools.save(mcmc_filename, export_multitrace(trace), format='dill')
