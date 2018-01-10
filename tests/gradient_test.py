"""
Note on the test:
The 'nutscost' (and thus the 'nutsgrad') quantities are not exactly comparable
to the plain cost, since they are for the posterior and therefore include priors.
"""
# TODO: Take 'plain' gradients wrt same variables as pymc/nuts gradients
# TODO: Run test with ntests ~ 1000, checking equality of pmyc and plain
#       and that nuts gradient never has any NaNs

import numpy as np
import theano
import theano.tensor as T
from collections import namedtuple
import time

import pymc3.sampling

import theano_shim as shim
import fsGIF.mcmc as mcmc
import fsGIF.core as core

np.set_printoptions(precision=3)

ntests = 1
profile_samples = 5 # Number of different parameter values to use when profiling

mgr = core.RunMgr(calc='mcmc')
mgr.load_parameters(param_file="params/mcmc.params")
model = mcmc.get_model(mgr)
pymcmodel = mcmc.get_pymc_model(mgr, model)
testvalues = {'tidx': model.n.get_t_idx(mgr.params.posterior.burnin),
              'batch_size': model.index_interval(mgr.params.posterior.datalen)}

# tidx_var = theano.tensor.lscalar('tidx')
# mbatch_var = theano.tensor.lscalar('batch_size')
# tidx_var.tag.test_value = testvalues['tidx']
# mbatch_var.tag.test_value = testvalues['batch_size']
# cost = model.loglikelihood(tidx_var, mbatch_var)
cost = model.loglikelihood(mgr.params.posterior.burnin, mgr.params.posterior.datalen)
logL = cost[0]

gradvars = [model.params.w, model.params.τ_θ, model.params.τ_m]
grads = T.grad(logL, gradvars)

# costfn = None
# gradfn = None
pymc_costfn = None
pymc_gradfn = None

VarPair = namedtuple("VarPair", ['param', 'RV', 'back', 'to'])
    # 'back' = transformation back to original variable
varpairs = None

def hasnan(*args):
    return any( np.isnan(arg).any() for arg in args )

def plain_cost_grad():
    return costfn(), gradfn()
    # param_vals = [vp.to(vp.param.get_value()).flatten() for vp in varpairs.values()]
    # return costfn(*param_vals), gradfn(*param_vals)
    # return (costfn(testvalues['tidx'], testvalues['batch_size']),
    #         gradfn(testvalues['tidx'], testvalues['batch_size']))

def pymc_cost_grad():
    param_vals = [vp.to(vp.param.get_value()).flatten() for vp in varpairs.values()]
    return (pymc_costfn(*param_vals), pymc_gradfn(*param_vals))

def get_varpairs():
    varpairs = {}
    for rv in pymcmodel.unobserved_RVs:
        name = rv.name
        if name[:3] == 'log':
            back = lambda x: 10**x
            to = lambda x: shim.log10(x)
            paramname = name[3:]
        else:
            back = lambda x: x
            to = lambda x: x
            paramname = name
        param = None
        for p in model.params:
            if p.name == paramname:
                param = p
                break
        assert(param is not None)
        varpairs[paramname] = VarPair(param, rv, back, to)
    return varpairs

def sample():
    for vp in varpairs.values():
        shape = vp.param.get_value().shape
        vp.param.set_value( vp.back(vp.RV.random()).reshape(shape) )

def nan_test():
    global costfn, gradfn, varpairs, pymc_costfn, pymc_gradfn
    # tidx_var.tag.test_value = testvalues['tidx']
    # mbatch_var.tag.test_value = testvalues['batch_size']

    # costfn = theano.function([tidx_var, mbatch_var], logL)
    # gradfn = theano.function([tidx_var, mbatch_var], grads)

    varpairs = get_varpairs()

    #start, nutsstep = pymc3.sampling.init_nuts(init='jitter+adapt_diag',
    #                                    model=pymcmodel)
    #nuts_gradfn = nutsstep._logp_dlogp_func._theano_function

    # pymc_cost should be exactly the same as plain_cost
    varinputs = [vp.RV for vp in varpairs.values()]
    # plain_inputs = {vp.param: vp.back(vp.RV).reshape(vp.param.get_value().shape)
    #                 for vp in varpairs.values()}
    # logL_w_inputs = theano.clone(logL, plain_inputs)
    costfn = theano.function([], logL)
    gradfn = theano.function([], grads)
    pymc_costfn = theano.function(varinputs, pymcmodel.n.logpt)
    pymc_grad = T.grad(pymcmodel.n.logpt, varinputs)
    pymc_gradfn = theano.function(varinputs, pymc_grad)

    # theano.printing.pydotprint(costfn, "plain_cost.svg", format='svg',
    #                            compact=True,
    #                            scan_graphs=True, var_with_name_simple=True)
    # theano.printing.pydotprint(pymc_costfn, "pymc_cost.svg", format='svg',
    #                            compact=True,
    #                            scan_graphs=True, var_with_name_simple=True)

    for i in range(ntests):
        sample()

        cost, grad = plain_cost_grad()
        param_vals = [vp.to(vp.param.get_value()).flatten() for vp in varpairs.values()]
        param_vals_concat = np.concatenate( [val for val in param_vals] )
        #nutscost, nutsgrad = nuts_gradfn(param_vals_concat)
        pymccost, pymcgrad = pymc_cost_grad()
        # pymccost = pymc_costfn(*param_vals)
        # pymcgrad = pymc_gradfn(*param_vals)
        print(param_vals_concat)
        print( "plain", cost, grad, hasnan(*grad))
        print( "pymc", pymccost, pymcgrad, hasnan(*pymcgrad))
        #print( "nuts", nutscost, nutsgrad, hasnan(nutsgrad) )

def profile():
    # [TODO?]: Measure integration time of pymc3.step_methods.hmc.nuts.[NUTS,_Tree].integrator
    #       Used in nuts.py:332, nuts.py:178
    cost_fn = theano.function([], logL)
    costgrad_fn = theano.function([], [logL] + grads)
    start, nutsstep = pymc3.sampling.init_nuts(init='jitter+adapt_diag',
                                               model=pymcmodel)
    nutsgrad_fn = nutsstep._logp_dlogp_func._theano_function
    param_vals = [vp.to(vp.param.get_value()).flatten() for vp in varpairs.values()]
    param_vals_concat = np.concatenate( [val for val in param_vals] )
    cost_times = []
    costgrad_times = []
    nutscostgrad_times = []
    print("Profiling cost and grad functions...")
    for i in range(profile_samples):
        output = []
        sample()
        t1 = time.perf_counter()
        cost_fn()
        t2 = time.perf_counter()
        cost_times.append(t2 - t1)
        output.append("Cost: {:.4f}s".format(cost_times[-1]))
        t1 = time.perf_counter()
        costgrad_fn()
        t2 = time.perf_counter()
        costgrad_times.append(t2 - t1)
        output.append("Cost+Grad: {:.4f}s".format(costgrad_times[-1]))
        t1 = time.perf_counter()
        nutsgrad_fn(param_vals_concat)
        t2 = time.perf_counter()
        nutscostgrad_times.append(t2 - t1)
        output.append("NUTS C+G: {:.4f}s".format(nutscostgrad_times[-1]))
        print('\t'.join(output))
    print("----------")
    print("Avg cost evaluation time: {}s".format(np.mean(cost_times)))
    print("Avg cost+grad evaluation time: {}s".format(np.mean(costgrad_times)))
    print("Avg NUTS c+g evaluation time: {}s".format(np.mean(nutscostgrad_times)))

varpairs = get_varpairs()

if __name__ == '__main__':
    profile()
