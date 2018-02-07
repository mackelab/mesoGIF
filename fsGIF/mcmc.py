import numpy as np
import pymc3 as pymc
import mackelab as ml
from mackelab.pymc3 import PyMCPrior, export_multitrace
from mackelab.parameters import Transform
import mackelab.parameters

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
shim.gettheano().config.compute_test_value = 'raise'
sinn.config.floatX = 'float64'
shim.gettheano().config.floatX = sinn.config.floatX

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
        # This catches some errors allows us to exit with a more friendly
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
        logp = shim.gettheano().clone(model_graph, self.variables)
        # Sanity check on logp: ensure all variable names are unique
        varnames = [v.name for v in shim.graph.inputs([logp]) if v.name is not None]
        counts = {name: varnames.count(name) for name in np.unique(varnames)}
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

    with pymc.Model() as pymc_model:
        priors = PyMCPrior(priorparams, modelvars)
        ndata = model.n._data.get_value()   # temp: nDist should be generic Dist
        n = nDist('n', params.burnin, params.datalen,
                  model = model,
                  variables = {getattr(model.params, varname) : prior
                               for varname, prior in priors.items()},
                  observed = ndata)

    return pymc_model

def run_mcmc(mgr, model):
    pymc_model = get_pymc_model(mgr, model)
    with pymc_model:
        trace = pymc.sample(**mgr.params.sampler)

    return trace

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
    data_history = core.subsample(data_history, postparams.model.dt)

    input_history = mgr.load(input_filename,
                             cls=getattr(histories, postparams.input.type).from_raw,
                             calc='input',
                             recalculate=False)
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

    model = get_model(mgr)
    trace = run_mcmc(mgr, model)

    mcmc_filename = mgr.get_pathname(label=None)
    iotools.save(mcmc_filename, export_multitrace(trace), format='dill')


