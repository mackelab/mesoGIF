import numpy as np
import pymc3 as pymc
from mackelab.pymc3 import PyMCPrior, export_multitrace
from mackelab.parameters import Transform

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
from fsGIF import fsgif_model as gif
############################

shim.load_theano()
shim.gettheano().config.compute_test_value = 'raise'
sinn.config.floatX = 'float64'
shim.gettheano().config.floatX = sinn.config.floatX

Transform.namespaces.update({'shim': shim})

class nDist(pymc.distributions.Continuous):

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

    def logp(self, n):
        model_graph = self.model.loglikelihood(self.start, self.batch_size, n)[0]
        return shim.gettheano().clone(model_graph, self.variables)

    def _repr_latex_(self, name=None, dist=None):
        if dist is None:
            dist = self
        if name is None:
            name = "n"
        return r"${} \sim \text{{fsGIF}}(…)$".format(name)

def run_mcmc(mgr, model):

    varnames = getattr(mgr.params.posterior, 'variables',
                       list(mgr.params.posterior.mask.keys()))
    masks = getattr(mgr.params.posterior, 'mask', {})
    priorparams = mgr.params.posterior.model.prior
    priorparams = ParameterSet(
        {key: value for key, value in priorparams.items()
         if key in varnames})
    for varname in varnames[:]:
        assert(varname in priorparams)
        mask = masks[varname]
        if varname in masks:
            if not np.any(mask):
                varnames.remove(varname)
                del masks[varname]
            else:
                priorparams[varname].mask = mask

    modelvars = [getattr(model.params, varname) for varname in varnames]

    with pymc.Model() as pymc_model:
        priors = PyMCPrior(priorparams, modelvars)
        ndata = model.n._data.get_value()   # temp: nDist should be generic Dist
        n = nDist('n', mgr.params.posterior.burnin, mgr.params.posterior.datalen,
                  model = model,
                  variables = {getattr(model.params, varname) : prior
                               for varname, prior in priors.items()},
                  observed = ndata)

        trace = pymc.sample(**mgr.params.sampler)

    return trace

def get_model(mgr):
    postparams = mgr.params.posterior
    data_filename  = mgr.get_pathname(params = postparams.data.params,
                                      subdir = postparams.data.dir,
                                      suffix = postparams.data.name,
                                      label = '')
    input_filename = mgr.get_pathname(params = postparams.input.params,
                                      subdir = postparams.input.dir,
                                      suffix = postparams.input.name,
                                      label = '')

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

if __name__ == "__main__":
    core.init_logging_handlers()

    mgr = core.RunMgr(description="MCMC sampler", calc='mcmc')
    # TODO: mgr.parser.add_argument('--resume' ...

    mgr.load_parameters()

    model = get_model(mgr)
    trace = run_mcmc(mgr, model)

    mcmc_filename = mgr.get_pathname(label=None)
    iotools.save(mcmc_filename, export_multitrace(trace), format='plain')


