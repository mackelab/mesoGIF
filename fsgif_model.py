# -*- coding: utf-8 -*-
"""
Created Wed May 24 2017

author: Alexandre René
"""

import numpy as np
import scipy as sp
from scipy.optimize import root
from collections import namedtuple, OrderedDict, Iterable
import logging
import copy

import theano_shim as shim
import sinn
import sinn.config as config
from sinn.histories import Series, Spiketrain
import sinn.kernels as kernels
import sinn.models as models

logger = logging.getLogger("fsgif_model")

# HACK
shim.cf.inf = 1e12
    # Actual infinity doesn't play nice in kernels, because inf*0 is undefined

class Kernel_ε(models.ModelKernelMixin, kernels.ExpKernel):
    @staticmethod
    def get_kernel_params(model_params):
        return kernels.ExpKernel.Parameters(
            height      = 1/model_params.τ_s,
            decay_const = model_params.τ_s,
            t_offset    = model_params.Δ)

# The θ kernel is separated in two: θ_1 is the constant equal to ∞ over (0, t_ref)
# θ_2 is the exponentially decaying adaptation kernel
class Kernel_θ1(models.ModelKernelMixin, kernels.Kernel):
    Parameter_info = OrderedDict( ( ( 'height', config.floatX ),
                                    ( 'start',  config.floatX ),
                                    ( 'stop',   config.floatX ) ) )
    Parameters = kernels.com.define_parameters(Parameter_info)
    @staticmethod
    def get_kernel_params(model_params):
        Npops = len(model_params.N.get_value())
        return Kernel_θ1.Parameters(
            height = (shim.cf.inf,)*Npops,
            start  = 0,
            stop   = model_params.t_ref
        )

    def __init__(self, name, params=None, shape=None, **kwargs):
        kern_params = self.get_kernel_params(params)
        memory_time = shim.asarray(kern_params.stop.get_value()
                                   - kern_params.start).max()
            # FIXME: At present, if we don't set memory_time now, tn is not set
            #        properly
        super().__init__(name, params, shape,
                         t0 = 0,
                         memory_time = memory_time,  # FIXME: memory_time should be optional
                         **kwargs)

    def _eval_f(self, t, from_idx=slice(None,None)):
        if shim.isscalar(t):
            return self.params.height
        else:
            # t has already been shaped to align with the function output in Kernel.eval
            return shim.ones(t.shape, dtype=sinn.config.floatX) * self.params.height

class Kernel_θ2(models.ModelKernelMixin, kernels.ExpKernel):
    @staticmethod
    def get_kernel_params(model_params):
        if model_params.t_ref.ndim == 1:
            t_offset = model_params.t_ref[np.newaxis,:]
        else:
            t_offset = model_params.t_ref
        return kernels.ExpKernel.Parameters(
            height      = model_params.J_θ / model_params.τ_θ,
            decay_const = model_params.τ_θ,
            t_offset    = t_offset
        )


class GIF_spiking(models.Model):

    # Entries to Parameter_info: ( 'parameter name',
    #                              (dtype, default value, shape_flag) )
    # If the shape_flag is True, the parameter will be reshaped into a 2d
    # matrix, if it isn't already. This is ensures the parameter is
    # consistent with kernel methods which assume inputs which are at least 2d
    # The last two options can be omitted; default flag is 'False'
    # Typically if a parameter will be used inside a kernel, shape_flag should be True.
    Parameter_info = OrderedDict( (( 'N',      'int32' ),
                                   ( 'R',      config.floatX ),    # membrane resistance (
                                   ( 'u_rest', (config.floatX, None, False) ),
                                   ( 'p',      config.floatX ),   # Connection probability between populations
                                   ( 'w',      config.floatX ),         # matrix of *population* connectivity strengths
                                   ( 'Γ',      'int32' ),               # binary connectivity between *neurons*
                                   ( 'τ_m',    config.floatX  ), # membrane time constant (s)
                                   ( 't_ref',  config.floatX  ), # absolute refractory period (s)
                                   ( 'u_th',   config.floatX  ),    # non-adapting threshold (mV)
                                   ( 'u_r',    config.floatX  ),    # reset potential (mV)
                                   ( 'c',      config.floatX  ),   # escape rate at threshold (Hz)
                                   ( 'Δu',     config.floatX  ),    # noise level (mV)
                                   ( 'Δ',      (config.floatX, None, True)), # transmission delay (s) (kernel ε)
                                   ( 'τ_s',    (config.floatX, None, True)), # synaptic time constant (mV) (kernel ε)
                                   # Adaptation parameters (θ-kernel dependent)
                                   ( 'J_θ',    (config.floatX, None, True)), # Integral of adaptation (mV s)
                                   ( 'τ_θ',    (config.floatX, None, True))
                                   ) )
    Parameters = sinn.define_parameters(Parameter_info)
    State = namedtuple('State', ['u', 't_hat'])


    def __init__(self, params, spike_history, input_history,
                 initializer='stationary', random_stream=None, memory_time=None):

        self._refhist = spike_history
        self.s = spike_history
        self.I_ext = input_history
        self.rndstream = random_stream
        if not isinstance(self.s, Spiketrain):
            raise ValueError("Spike history must be an instance of sinn.Spiketrain.")
        if not isinstance(self.I_ext, Series):
            raise ValueError("External input history must be an instance of sinn.Series.")
        # This runs consistency tests on the parameters
        # models.Model.same_shape(self.s, self.I)
        models.Model.same_dt(self.s, self.I_ext)
        models.Model.output_rng(self.s, self.rndstream)

        super().__init__(params)
        # NOTE: Do not use `params` beyond here. Always use self.params.

        N = self.params.N.get_value()
        assert(N.ndim == 1)
        self.Npops = len(N)

        # Model variables
        self.RI_syn = Series(self.s, 'RI_syn',
                             shape = (N.sum(), ))
        self.λ = Series(self.RI_syn, 'λ')
        self.varθ = Series(self.RI_syn, 'ϑ')
        self.u = Series(self.RI_syn, 'u')
        # Surrogate variables
        self.t_hat = Series(self.RI_syn, 't_hat')
            # time since last spike

        # Kernels
        shape2d = (self.Npops, self.Npops)
        self.ε = Kernel_ε('ε', self.params, shape=shape2d)
        self.θ1 = Kernel_θ1('θ1', self.params, shape=(self.Npops,))
        self.θ2 = Kernel_θ2('θ2', self.params, shape=(self.Npops,))

        self.add_history(self.s)
        self.add_history(self.I_ext)
        self.add_history(self.λ)
        self.add_history(self.varθ)
        self.add_history(self.u)
        self.add_history(self.RI_syn)
        self.add_history(self.t_hat)

        self.s.set_update_function(self.s_fn)
        self.λ.set_update_function(self.λ_fn)
        self.varθ.set_update_function(self.varθ_fn)
        self.u.set_update_function(self.u_fn)
        self.RI_syn.set_update_function(self.RI_syn_fn)
        self.t_hat.set_update_function(self.t_hat_fn)

        self.s.add_inputs([self.λ])
        self.λ.add_inputs([self.u, self.varθ])
        self.varθ.add_inputs([self.s])
        self.u.add_inputs([self.t_hat, self.u, self.I_ext, self.RI_syn])
        self.RI_syn.add_inputs([self.s])
        self.t_hat.add_inputs([self.s, self.t_hat])

        # Pad to allow convolution
        # FIXME Check with GIF_mean_field to see if memory_time could be better / more consistently treated
        if memory_time is None:
            memory_time = 0
        self.memory_time = max(memory_time,
                               max( kernel.memory_time
                                    for kernel in [self.ε, self.θ1, self.θ2] ) )
        self.s.pad(self.memory_time)
        # Pad because these are ODEs (need initial condition)
        #self.u.pad(1)
        #self.t_hat.pad(1)

        # Expand the parameters to treat them as neural parameters
        # Original population parameters are kept as a copy
        # self.pop_params = copy.copy(self.params)
        ExpandedParams = namedtuple('ExpandedParams', ['u_rest', 't_ref', 'u_r'])
        self.expanded_params = ExpandedParams(
            u_rest = self.expand_param(self.params.u_rest, self.params.N),
            t_ref = self.expand_param(self.params.t_ref, self.params.N),
            u_r = self.expand_param(self.params.u_r, self.params.N)
        )

        self.init_state_vars(initializer)

        # # Initialize last spike time such that it was effectively "at infinity"
        # idx = self.t_hat.t0idx - 1; assert(idx >= 0)
        # data = self.t_hat._data.get_value(borrow=True)
        # data[idx, :] = shim.ones(self.t_hat.shape) * self.memory_time #* self.t_hat.dt
        # self.t_hat._data.set_value(data, borrow=True)

        # # Initialize membrane potential to u_rest
        # idx = self.u.t0idx - 1; assert(idx >= 0)
        # data = self.t_hat._data.get_value(borrow=True)
        # data[idx, :] = self.expanded_params.u_rest
        # self.u._data.set_value(data, borrow=True)

        # self.params = Parameters(
        #     N = self.params.N,
        #     u_rest = self.params.u_rest,
        #     w = self.params.w,
        #     Γ = self.params.Γ,
        #     τ_m = self.expand_param(self.params.τ_m),
        #     t_ref = self.params.t_ref,
        #     u_th = self.expand_param(self.params.u_th),
        #     u_r = self.expand_param(self.params.u_r),
        #     c = self.expand_param(self.params.c),
        #     Δu =

    def init_state_vars(self, initializer='stationary'):

        if initializer == 'stationary':
            K = np.rint( self.memory_time / self.dt ).astype(int)
            θ_dis, θtilde_dis = GIF_mean_field.discretize_θkernel(
                [self.θ1, self.θ2], self._refhist, self.params)
            init_A = GIF_mean_field.get_stationary_activity(
                self, K, θ_dis, θtilde_dis)
            init_state = self.get_stationary_state(init_A)
        elif initializer == 'silent':
            init_A = np.zeros((len(self.s.pop_slices),))
            init_state = self.get_silent_state()
        else:
            raise ValueError("Initializer string must be one of 'stationary', 'silent'")

        for varname in self.State._fields:
            hist = getattr(self, varname)
            initval = getattr(init_state, varname)
            hist.pad(1)
            idx = hist.t0idx - 1; assert(idx >= 0)
            data = hist._data.get_value(borrow=True)
            data[idx,:] = initval
            hist._data.set_value(data, borrow=True)

        # TODO: Combine the following into the loop above
        nbins = self.s.t0idx
        self.s.update(np.arange(nbins), [ np.nonzero(timebin)[0] for timebin in init_state.s ] )
        #data = self.s._data
        #data[:nbins,:] = init_state.s
        #self.s._data.set_value(data, borrow=True)

    def get_silent_state(self):
        # TODO: include spikes in model state, so we don't need this custom 'Stateplus'
        Stateplus = namedtuple('Stateplus', self.State._fields + ('s',))
        state = Stateplus(
            u = self.expanded_params.u_rest,
            t_hat = shim.ones(self.t_hat.shape) * self.memory_time,
            s = np.zeros((self.s.t0idx, self.s.shape[0]))
            )
        return state

    def get_stationary_state(self, Astar):
        # TODO: include spikes in model state, so we don't need this custom 'Stateplus'
        Stateplus = namedtuple('Stateplus', self.State._fields + ('s',))
        # Initialize the spikes
        # We treat that as a Bernouilli process, with firing rate
        # given by Astar; this means ISI statistics will be off as
        # we ignore refractory effects, but the overall rate will be
        # correct.
        p = self.expand_param(Astar, self.params.N) * self.dt
        nbins = self.s.t0idx
        nneurons = self.s.shape[0]
        s = np.random.binomial(1, p, (nbins, nneurons))
        # argmax returns first occurrence; by flipping s, we get the
        # index (from the end) of the last spike, i.e. number of bins - 1
        t_hat = (s[::-1].argmax(axis=0) + 1) * self.dt
        # u is initialized by integrating the ODE from the last spike
        # See documentation for details (TODO: not yet in docs)
        τm_exp = self.expand_param(self.params.τ_m, self.params.N)
        τmT = self.params.τ_m.flatten()[:, np.newaxis]
        η1 = τmT * self.params.p * self.params.N * self.params.w
            # As in GIF_mean_field.get_η_csts
        u = np.where(t_hat <= self.expanded_params.t_ref,
                     self.expanded_params.u_r,
                     (1 - np.exp(-t_hat/τm_exp)) * self.expand_param((self.params.u_rest + η1.dot(Astar)),
                                                                     self.params.N)
                       + self.expanded_params.u_r * np.exp(-t_hat/τm_exp)
                     )
        state = Stateplus(
            u = u,
            t_hat = t_hat,
            s = s
        )
        return state

    @staticmethod
    def make_connectivity(N, p):
        """
        Construct a binary neuron connectivity matrix, for use as this class'
        Γ parameter.

        Parameters
        ----------
        N: array of ints
            Number of neurons in each population
        p: 2d array of floats between 0 and 1
            Connection probability. If connection probabilities are symmetric,
            this array should also be symmetric.

        Returns
        -------
        2d binary matrix
        """
        # TODO: Use array to pre-allocate memory
        # TODO: Does it make sense to return a sparse matrix ? For low connectivity
        #       yes, but with the assumed all-to-all connectivity, we can't really
        #       have p < 0.2.
        Γrows = []
        for Nα, prow in zip(N, p):
            Γrows.append( np.concatenate([ np.random.binomial(1, pαβ, size=(Nα, Nβ))
                                        for Nβ, pαβ in zip(N, prow) ],
                                        axis = 1) )

        Γ = np.concatenate( Γrows, axis=0 )
        return Γ

    @staticmethod
    def expand_param(param, N):
        """
        Expand a population parameter such that it can be multiplied directly
        with the spiketrain.

        Parameters
        ----------
        param: ndarray
            Parameter to expand

        N: tuple or ndarray
            Number of neurons in each population
        """

        Npops = len(N)
        if param.ndim == 1:
            return shim.concatenate( [ param[i]*np.ones((N[i],))
                                       for i in range(Npops) ] )

        elif param.ndim == 2:
            # i, j = 0, 0
            # A = [ param[i, j]* np.ones((N[i], N[j]))
            #                           for j in range(Npops) ]
            # B = shim.concatenate( [ param[i, j]* np.ones((N[i], N[j]))
            #                           for j in range(Npops) ],
            #                         axis = 1 )
            # C = [ shim.concatenate( [ param[i, j]* np.ones((N[i], N[j]))
            #                           for j in range(Npops) ],
            #                         axis = 1 )
            #       for i in range(Npops) ]
            # D = shim.concatenate(
            #     [ shim.concatenate( [ param[i, j]* np.ones((N[i], N[j]))
            #                           for j in range(Npops) ],
            #                         axis = 1 )
            #       for i in range(Npops) ],
            #     axis = 0 )
            return shim.concatenate(
                [ shim.concatenate( [ param[i, j]* np.ones((N[i], N[j]))
                                      for j in range(Npops) ],
                                    axis = 1 )
                  for i in range(Npops) ],
                axis = 0 )
        else:
            raise ValueError("Parameter {} has {} dimensions; can only expand "
                             "dimensions of 1d and 2d parameters."
                             .format(param.name, param.ndim))

    def loglikelihood(self, start, batch_size):
        # >>>>>>>>>>>>>> WARNING: Untested, incomplete <<<<<<<<<<<<<

        #######################
        # Some hacks to get around current limitations

        self.remove_other_histories()

        # End hacks
        #####################

        startidx = self.get_t_idx(start)
        stopidx = startidx + batch_size
        N = self.params.N

        def logLstep(tidx, cum_logL):
            # TODO: Don't use private _data variable
            p = sinn.clip_probabilities(self.λ[tidx]*self.s.dt)
            s = shim.cast(self.s._data.tocsr()[tidx+self.s.t0idx], 'int32')

            # L = s*n - (1-s)*(1-p)
            cum_logL += ( s*p - (1-p) + s*(1-p) ).sum()

            return [cum_logL], shim.get_updates()

        if shim.is_theano_object([self.nbar, self.params, self.n]):
            raise NotImplementedError

            logger.info("Producing the likelihood graph.")

            if batch_size == 1:
                # No need for scan
                logL, upds = logLstep(start, 0)

            else:
                # FIXME np.float64 -> shim.floatX or sinn.floatX
                logL, upds = shim.gettheano().scan(logLstep,
                                                sequences = shim.getT().arange(startidx, stopidx),
                                                outputs_info = np.float64(0))
                self.apply_updates(upds)
                    # Applying updates is essential to remove the temporary iteration variable
                    # scan introduces from the shim updates dictionary

            logger.info("Likelihood graph complete")

            return logL[-1], upds
        else:
            # TODO: Remove this branch once shim.scan is implemented
            logL = 0
            for t in np.arange(startidx, stopidx):
                logL = logLstep(t, logL)[0][0]
            upds = shim.get_updates()

            return logL, upd

    def f(self, u):
        """Link function. Maps difference between membrane potential & threshold
        to firing rate."""
        return self.params.c * shim.exp(u/self.params.Δu.flatten())


    def RI_syn_fn(self, t):
        """Incoming synaptic current times membrane resistance. Eq. (20)."""
        return ( self.s.pop_rmul( self.params.τ_m,
                                  self.s.convolve(self.ε, t) ) )
            # s includes the connection weights w, and convolution also includes
            # the sums over j and β in Eq. 20.
            # Need to fix spiketrain convolution before we can use the exponential
            # optimization. (see fixme comments in histories.spiketrain._convolve_single_t
            # and kernels.ExpKernel._convolve_single_t). Weights will then no longer
            # be included in the convolution


    def u_fn(self, t):
        """Membrane potential. Eq. (21)."""
        if not shim.isscalar(t):
            tidx_u_m1 = self.u.time_array_to_slice(t - self.u.dt)
            t_that = self.t_hat.time_array_to_slice(t)
            t_Iext = self.I_ext.time_array_to_slice(t)
            t_RIsyn = self.RI_syn.time_array_to_slice(t)
        else:
            tidx_u_m1 = self.u.get_t_idx(t) - 1
            t_that = t
            t_Iext = t
            t_RIsyn = t
            # Take care using this on another history than u – it may be padded
        # Euler approximation for the integral of the differential equation
        # 'm1' stands for 'minus 1', so it's the previous time bin
        red_factor = shim.exp(-self.u.dt/self.params.τ_m)
        return shim.switch( shim.ge(self.t_hat[t_that], self.expanded_params.t_ref),
                            self.s.pop_mul( self.u[tidx_u_m1],
                                            red_factor )
                            + self.s.pop_mul( self.s.pop_radd( (self.params.u_rest + self.params.R * self.I_ext[t_Iext]) ,
                                                               self.RI_syn[t_RIsyn] ),
                                              (1 - red_factor) ),
                            self.expanded_params.u_r )

    def varθ_fn(self, t):
        """Dynamic threshold. Eq. (22)."""
        if t > 1:
            sinn.flag = True
        return self.s.pop_radd(self.params.u_th,
                               self.s.convolve(self.θ1, t) + self.s.convolve(self.θ2, t))
            # Need to fix spiketrain convolution before we can use the exponential
            # optimization. (see fixme comments in histories.spiketrain._convolve_single_t
            # and kernels.ExpKernel._convolve_single_t)

    def λ_fn(self, t):
        """Hazard rate. Eq. (23)."""
        # TODO: Use self.f here (requires overloading of ops to remove pop_rmul & co.)
        return self.s.pop_rmul(self.params.c,
                               shim.exp( self.s.pop_div( ( self.u[t] - self.varθ[t] ),
                                                       self.params.Δu ) ) )

    def s_fn(self, t):
        """Spike generation"""
        return ( self.rndstream.binomial( size = self.s.shape,
                                          n = 1,
                                          p = sinn.clip_probabilities(self.λ[t]*self.s.dt) )
                 .nonzero()[0] )
            # nonzero returns a tuple, with one element per axis

    def t_hat_fn(self, t):
        """Update time since last spike"""
        if shim.isscalar(t):
            s_tidx_m1 = self.s.get_t_idx(t) - 1
            t_tidx_m1 = self.t_hat.get_t_idx(t) - 1
            cond_tslice = 0
        else:
            s_tidx_m1 = self.s.time_array_to_slice(t - self.s.dt)
            t_idx_m1 = self.t_hat.time_array_to_slice(t - self.t.dt)
            cond_tslice = slice(None)
        # If the last bin was a spike, set the time to dt (time bin length)
        # Otherwise, add dt to the time
        return shim.switch( (self.s[s_tidx_m1] == 0)[cond_tslice],
                            self.t_hat[t_tidx_m1] + self.t_hat.dt,
                            self.t_hat.dt )




# TODO: Implement pop_xxx functions as methods of Spiketrain
def pop_add(pop_slices, neuron_term, summand):
    if not shim.is_theano_object(neuron_term, summand):
        assert(len(pop_slices) == len(summand))
        return shim.concatenate([neuron_term[pop_slice] + sum_el
                                 for pop_slice, sum_el in zip(pop_slices, summand)],
                                axis=0)
    else:
        raise NotImplementedError

def pop_radd(pop_slices, summand, neuron_term):
    return pop_add(pop_slices, neuron_term, summand)

def pop_mul(pop_slices, neuron_term, multiplier):
    if not shim.is_theano_object(neuron_term, multiplier):
        assert(len(pop_slices) == len(multiplier))
        return shim.concatenate([neuron_term[pop_slice] * mul_el
                                 for pop_slice, mul_el in zip(pop_slices, multiplier)],
                                axis=0)
    else:
        raise NotImplementedError

def pop_rmul(pop_slices, multiplier, neuron_term):
    return pop_mul(pop_slices, neuron_term, multiplier)

def pop_div(pop_slices, neuron_term, divisor):
    if not shim.is_theano_object(neuron_term, divisor):
        assert(len(pop_slices) == len(divisor))
        return shim.concatenate( [ neuron_term[pop_slice] / div_el
                                   for pop_slice, div_el in zip(pop_slices, divisor)],
                                 axis = 0)
    else:
        raise NotImplementedError



class GIF_mean_field(models.Model):
    Parameter_info = GIF_spiking.Parameter_info
    Parameters = sinn.define_parameters(Parameter_info)

    # 'State' is an irreducible set of variables which uniquely define the model's state
    State = namedtuple('State',
                       ['h', 'u',
                        'λ', 'λfree',
                        'g', 'm', 'v', 'x', 'y', 'z'])
        # TODO: Some way of specifying how much memory is needed for each variable
        #       Or get this directly from the update functions, by some kind of introspection ?
    # HACK For propagating gradients without scan
    #      Order must be consistent with return value of symbolic_update
    #statevars = [ 'λfree', 'λ', 'g', 'h', 'u', 'v', 'm', 'x', 'y', 'z' ]

    def __init__(self, params, activity_history, input_history,
                 initializer='stationary', random_stream=None, memory_time=None):

        self._refhist = activity_history
        self.A = activity_history
        self.I_ext = input_history
        self.rndstream = random_stream
        if not isinstance(self.A, Series):
            raise ValueError("Activity history must be an instance of sinn.Series.")
        if not isinstance(self.I_ext, Series):
            raise ValueError("External input history must be an instance of sinn.Series.")
        # This runs consistency tests on the parameters
        models.Model.same_shape(self.A, self.I_ext)
        models.Model.same_dt(self.A, self.I_ext)
        models.Model.output_rng(self.A, self.rndstream)

        super().__init__(params)
        # NOTE: Do not use `params` beyond here. Always use self.params.

        N = self.params.N.get_value()
        assert(N.ndim == 1)
        self.Npops = len(N)

        # TODO: Move to History.index_interval
        self.Δ_idx = shim.concatenate(
            [ shim.concatenate (
                [ shim.asarray(self.A.index_interval(self.params.Δ.get_value()[α,β])).reshape(1,1)
                  for β in range(self.Npops) ],
                axis = 1)
              for α in range(self.Npops) ],
            axis = 0)
        # A will be indexed by 0 - Δ_idx
        self.A.pad(self.Δ_idx.max() + 1) # +1 HACK; not sure if required

        # Hyperparameters ?
        self.Nθ = 1  # Number of exponentials in threshold kernel
                     # Code currently does not allow for Nθ > 1

        # Kernels
        shape2d = (self.Npops, self.Npops)
        self.ε = Kernel_ε('ε', self.params, shape=shape2d)
        self.θ1 = Kernel_θ1('θ1', self.params, shape=(self.Npops,))
        self.θ2 = Kernel_θ2('θ2', self.params, shape=(self.Npops,))
            # Temporary; created to compute its memory time
        self.memory_time, self.K = self.get_memory_time(self.θ2); del self.θ2
        # HACK
        #self.memory_time = 0.02; self.K = self.A.index_interval(0.02)# - 1
            # DEBUG
        self.θ2 = Kernel_θ2('θ2', self.params, shape=(self.Npops,),
                            memory_time=self.memory_time)

        # Histories
        self.n = Series(self.A, 'n')
        self.h = Series(self.A, 'h')
        self.h_tot = Series(self.A, 'h_tot')
        self.u = Series(self.A, 'u', shape=(self.K, self.Npops))
            # self.u[t][0] is the array of membrane potentials at time t, at lag Δt, of each population
            # TODO: Remove +1: P_λ_fn doesn't need it anymore
        self.varθ = Series(self.u, 'varθ')
        self.λ = Series(self.u, 'λ')

        # Temporary variables
        self.nbar = Series(self.n, 'nbar')
        self.A_Δ = Series(self.A, 'A_Δ', shape=(self.Npops, self.Npops))
        #self.A_Δ.pad(1)  # +1 HACK (safety)
        #self.g = Series(self.A, 'g', shape=(self.Npops, self.Nθ,))
        self.g = Series(self.A, 'g', shape=(self.Npops,))  # HACK: Nθ = 1    # auxiliary variable(s) for the threshold of free neurons. (avoids convolution)

        # Free neurons
        self.x = Series(self.A, 'x')                                         # number of free neurons
        self.y = Series(self.A, 'y', shape=(self.Npops, self.Npops))         # auxiliary variable for the membrane potential of free neurons (avoids convolution)
        self.z = Series(self.x, 'z')                                         # variance function integrated over free neurons
        self.varθfree = Series(self.A, 'varθfree', shape=(self.Npops,))  # HACK: Nθ = 1
        #self.λtilde = Series(self.u, 'λtilde')
            # In pseudocode, same symbol as λtildefree
        #self.λtildefree = Series(self.A, 'λtildefree')
        self.λfree = Series(self.A, 'λfree')
            #TODO: Either just take λtilde in the past, or make λtilde & λfree variables
        self.Pfree = Series(self.λfree, 'Pfree')

        # Refractory neurons
        self.m = Series(self.u, 'm', shape=(self.K, self.Npops))          # Expected no. neurons for each last-spike bin
            # One more than v, because we need the extra spill-over bin to compute how many neurons become 'free' (Actually, same as v)
        self.P_λ = Series(self.m, 'P_λ')
        self.v = Series(self.m, 'v', shape=(self.K, self.Npops))
        self.P_Λ = Series(self.Pfree, 'P_Λ')
        self.X = Series(self.A, 'X')
        self.Y = Series(self.X, 'Y')
        self.Z = Series(self.X, 'Z')
        self.W = Series(self.X, 'W')

        # HACK For propagating gradients without scan
        #      Order must be consistent with return value of symbolic_update
        self.statehists = [ getattr(self, varname) for varname in self.State._fields ]


        # Initialize the variables
        self.init_state_vars(initializer)
        # self.θtilde                                                     # QR kernel, computed from θ
        # self.θhat                                                       # convolution of θtilde with n up to t_n (for t_n > t - self.K)


        #####################################################
        # Create the loglikelihood function
        # FIXME: Doesn't work with Theano histories because they only support updating tidx+1
        #        Need to create a Variable(History) type, which doesn't
        #        trigger 'compute_up_to'.
        # TODO: Use op and write as `self.nbar / self.params.N`
        #phist = Series(self.nbar, 'p')
        #phist.set_update_function(lambda t: self.nbar[t] / self.params.N)
        #phist.add_inputs([self.nbar])
        #phist = self.nbar / self.params.N
        #self.loglikelihood = self.make_binomial_loglikelihood(
        #    self.n, self.params.N, phist, approx='low p')
        #####################################################

        self.add_history(self.A)
        self.add_history(self.I_ext)
        self.add_history(self.θ_dis)
        self.add_history(self.θtilde_dis)
        self.add_history(self.n)
        self.add_history(self.h)
        self.add_history(self.h_tot)
        self.add_history(self.u)
        self.add_history(self.varθ)
        self.add_history(self.λ)
        self.add_history(self.A_Δ)
        self.add_history(self.g)
        self.add_history(self.x)
        self.add_history(self.y)
        self.add_history(self.z)
        self.add_history(self.varθfree)
        #self.add_history(self.λtildefree)
        self.add_history(self.λfree)
        #self.add_history(self.λtilde)
        self.add_history(self.Pfree)
        self.add_history(self.v)
        self.add_history(self.m)
        self.add_history(self.P_λ)
        self.add_history(self.P_Λ)
        self.add_history(self.X)
        self.add_history(self.Y)
        self.add_history(self.Z)
        self.add_history(self.W)
        self.add_history(self.nbar)

        self.A.set_update_function(self.A_fn)
        self.n.set_update_function(self.n_fn)
        self.h.set_update_function(self.h_fn)
        self.h_tot.set_update_function(self.h_tot_fn)
        self.u.set_update_function(self.u_fn)
        self.varθ.set_update_function(self.varθ_fn)
        self.λ.set_update_function(self.λ_fn)
        self.A_Δ.set_update_function(self.A_Δ_fn)
        self.g.set_update_function(self.g_fn)
        self.x.set_update_function(self.x_fn)
        self.y.set_update_function(self.y_fn)
        self.z.set_update_function(self.z_fn)
        self.varθfree.set_update_function(self.varθfree_fn)
        #self.λtildefree.set_update_function(self.λtildefree_fn)
        self.λfree.set_update_function(self.λfree_fn)
        #self.λtilde.set_update_function(self.λtilde_fn)
        self.Pfree.set_update_function(self.Pfree_fn)
        self.v.set_update_function(self.v_fn)
        self.m.set_update_function(self.m_fn)
        self.P_λ.set_update_function(self.P_λ_fn)
        self.P_Λ.set_update_function(self.P_Λ_fn)
        self.X.set_update_function(self.X_fn)
        self.Y.set_update_function(self.Y_fn)
        self.Z.set_update_function(self.Z_fn)
        self.W.set_update_function(self.W_fn)
        self.nbar.set_update_function(self.nbar_fn)


        # FIXME: At present, sinn dependencies don't support lagged
        #        inputs (all inputs are assumed to need the same time point t),
        #        while some of the dependencies below are on previous time points
        self.A.add_inputs([self.n])
        self.n.add_inputs([self.nbar])
        self.h.add_inputs([self.h, self.h_tot])
        #self.h.add_inputs([self.h_tot])
        self.h_tot.add_inputs([self.I_ext, self.A_Δ, self.y])
        self.u.add_inputs([self.u, self.h_tot])
        #self.u.add_inputs([self.h_tot])
        self.varθ.add_inputs([self.varθfree, self.n, self.θtilde_dis])
        #self.varθ.add_inputs([self.varθfree, self.θtilde_dis])
        self.λ.add_inputs([self.u, self.varθ])
        self.A_Δ.add_inputs([self.A])
        self.g.add_inputs([self.g, self.n])
        #self.g.add_inputs([])
        self.x.add_inputs([self.Pfree, self.x, self.m])
        #self.x.add_inputs([])
        self.y.add_inputs([self.y, self.A_Δ])
        #self.y.add_inputs([self.A_Δ])
        self.z.add_inputs([self.Pfree, self.z, self.x, self.v])
        #self.z.add_inputs([self.Pfree])
        self.varθfree.add_inputs([self.g])
        self.λfree.add_inputs([self.h, self.varθfree])
        self.Pfree.add_inputs([self.λfree, self.λfree])
        self.v.add_inputs([self.v, self.m, self.P_λ])
        #self.v.add_inputs([self.m, self.P_λ])
        self.m.add_inputs([self.n, self.m, self.P_λ])
        #self.m.add_inputs([])
        self.P_λ.add_inputs([self.λ])
        self.P_Λ.add_inputs([self.z, self.Z, self.Y, self.Pfree])
        #self.P_Λ.add_inputs([self.Z, self.Y, self.Pfree])
        self.X.add_inputs([self.m])
        self.Y.add_inputs([self.P_λ, self.v])
        #self.Y.add_inputs([self.P_λ])
        self.Z.add_inputs([self.v])
        #self.Z.add_inputs([])
        self.W.add_inputs([self.P_λ, self.m])
        self.nbar.add_inputs([self.W, self.Pfree, self.x, self.P_Λ, self.X])

        #if self.A._original_tidx.get_value() >= self.A.t0idx + len(self.A) - 1:
        if self.A.locked:
            self.given_A()


        # Used to fill the data by iterating
        # FIXME: This is only required because of our abuse of shared variable updates
        # if shim.config.use_theano:
        #     logger.info("Compiling advance function.")
        #     tidx = shim.getT().iscalar()
        #     self.remove_other_histories()  # HACK
        #     self.clear_unlocked_histories()
        #     self.theano_reset()
        #     self.nbar[tidx + self.nbar.t0idx]  # Fills updates
        #     self._advance_fn = shim.gettheano().function([tidx], [], updates=shim.get_updates())
        #     self.theano_reset()
        #     logger.info("Done.")


    def given_A(self):
        """Run this function when A is given data. It reverses the dependency
        n -> A to A -> n and fills the n array
        WARNING: We've hidden the dependency on params.N here.
        """
        assert(self.A._original_tidx.get_value() >= self.A.t0idx + len(self.A) - 1)
        self.n.clear_inputs()
        # TODO: use op
        #self.n.set_update_function(lambda t: self.A[t] * self.params.N * self.A.dt)
        self.n.pad(self.A.t0idx)
        self.A.pad(self.n.t0idx)  # Increase whichever has less padding
        ndata = self.A._data * self.params.N * self.A.dt
        self.n.add_input(self.A)
        self.n.set(ndata.eval())
            # TODO: Don't remove dependence on self.param.N
        self.n.lock()

        # HACK Everything below
        self.A_Δ.set()
        self.A_Δ._original_data.set_value(self.A_Δ._data.eval())
        self.A_Δ._data = self.A_Δ._original_data
        self.A_Δ._original_tidx.set_value(self.A_Δ._cur_tidx.eval())
        self.A_Δ._cur_tidx = self.A_Δ._original_tidx
        self.A_Δ.lock()

    def get_t_idx(self, t, allow_rounding=False):
        """
        Returns the time index corresponding to t, with 0 corresponding to t0.
        """
        if shim.istype(t, 'int'):
            return t
        else:
            return self.A.get_t_idx(t, allow_rounding) - self.A.t0idx
    def index_interval(self, Δt):
        return self.A.index_interval(Δt)


    def get_memory_time(self, kernel, max_time=10):
        """
        Based on GetHistoryLength (p. 52). We set a global memory_time, rather
        than a population specific one; this is much easier to vectorize.

        Parameters
        ----------
        max_time: float
            Maximum allowable memory time, in seconds.
        """
        # def evalT(x):
        #     return (x.get_value() if shim.isshared(x)
        #             else x.eval() if shim.is_theano_variable(x)
        #             else x)

        if shim.is_theano_object(kernel.eval(0)):
            t = shim.getT().dscalar('t')
            kernelfn = shim.gettheano().function([t], kernel.eval(t))
        else:
            kernelfn = lambda t: kernel.eval(t)

        T = float(max_time // self.A.dt * self.A.dt)  # make sure T is a multiple of dt
        #while (evalT(kernel.eval(T)) < 0.1 * self.Δ_idx).all() and T > self.A.dt:
        while (kernelfn(T) < 0.1 * self.Δ_idx).all() and T > self.A.dt:
            T -= self.A.dt

        T = max(T, 5*self.params.τ_m.get_value().max(), self.A.dt)
        K = self.index_interval(T)
        return T, K

    def init_state_vars(self, initializer='stationary'):
        """
        Originally based on InitPopulations (p. 52)

        Parameters
        ----------
        initializer: str
            One of
              - 'stationary': (Default) Stationary state under no input conditions.
              - 'silent': The last firing time of each neuron is set to -∞. Very artificial
                condition, that may require a long burnin time to remove the transient.

        TODO: Call this every time the model is updated
        """
        # FIXME: Initialize series' to 0

        self.θ_dis, self.θtilde_dis = self.discretize_θkernel(
            [self.θ1, self.θ2], self.A, self.params)

        # Pad the the series involved in adaptation
        max_mem = self.u.shape[0]
            # The longest memories are of the size of u
        self.n.pad(max_mem)
        #self.θtilde_dis.pad(max_mem)
        self.varθ.pad(max_mem)
        self.varθfree.pad(max_mem)

        # >>>>> Extreme HACK, remove ASAP <<<<<
        self.θ_dis.locked = True
        self.θtilde_dis.locked = True
        # <<<<<

        # ====================
        # Initialize state variables

        if initializer == 'stationary':
            init_A = self.get_stationary_activity(self, self.K, self.θ_dis, self.θtilde_dis)
            init_state = self.get_stationary_state(init_A)
        elif initializer == 'silent':
            init_A = np.zeros(self.A.shape)
            init_state = self.get_silent_state()
        else:
            raise ValueError("Initializer string must be one of 'stationary', 'silent'")

        # Set initial values (make sure this is done after all padding is added)

        # ndata = self.n._data.get_value(borrow=True)
        # ndata[0] = self.params.N
        # self.n._data.set_value(ndata, borrow=True)
        # mdata = self.m._data.get_value(borrow=True)
        # mdata[0, -1, :] = self.params.N
        # self.m._data.set_value(mdata, borrow=True)

        for varname in self.State._fields:
            hist = getattr(self, varname)
            initval = getattr(init_state, varname)
            hist.pad(1)  # Ensure we have at least one bin for the initial value
                # TODO: Allow longer padding
            idx = hist.t0idx - 1; assert(idx >= 0)
            data = hist._data.get_value(borrow=True)
            data[idx,:] = initval
            hist._data.set_value(data, borrow=True)

        # TODO: Make A a state variable and just use the above code
        data = self.A._data.get_value(borrow=True)
        data[:self.A.t0idx,:] = init_A
        self.A._data.set_value(data, borrow=True)

        # # Make all neurons free neurons
        # idx = self.x.t0idx - 1; assert(idx >= 0)
        # data = self.x._data.get_value(borrow=True)
        # data[idx,:] = self.params.N.get_value()
        # self.x._data.set_value(data, borrow=True)

        # # Set refractory membrane potential to u_rest
        # idx = self.u.t0idx - 1; assert(idx >= 0)
        # data = self.u._data.get_value(borrow=True)
        # data[idx,:] = self.params.u_rest.get_value()
        # self.u._data.set_value(data, borrow=True)

        # # Set free membrane potential to u_rest
        # idx = self.h.t0idx - 1; assert(idx >= 0)
        # data = self.h._data.get_value(borrow=True)
        # data[idx,:] = self.params.u_rest.get_value()
        # self.h._data.set_value(data, borrow=True)

        #self.g_l.set_value( np.zeros((self.Npops, self.Nθ)) )
        #self.y.set_value( np.zeros((self.Npops, self.Npops)) )

        # =============================
        # Set the refractory mask

        # TODO: Use a switch here, so ref_mask can have a symbolic dependency on t_ref
        # Create the refractory mask
        # This mask is zero for time bins within the refractory period,
        # such that it can be multiplied element-wise with arrays of length K
        self.ref_mask = np.ones(self.u.shape)
        for l in range(self.ref_mask.shape[0]):
            # Loop over lags. l=0 corresponds to t-Δt, l=1 to t-2Δt, etc.
            for α in range(self.ref_mask.shape[1]):
                # Loop over populations α
                if (l+1)*self.dt < self.params.t_ref.get_value()[α]:
                    self.ref_mask[l, α] = 0
                else:
                    break

    # # FIXME: This is only required because of our abuse of shared variable
    # # updates
    # def advance(self, stop):
    #     stopidx = self.nbar.get_t_idx(stop + self.nbar.t0idx)
    #     for i in range(self.nbar._original_tidx.get_value() + 1, stopidx):
    #         self._advance_fn(i)

    @staticmethod
    def discretize_θkernel(θ, reference_hist, params):
        """
        Parameters
        ----------
        θ: kernel, or iterable of kernels
            The kernel to discretize. If an iterable, its elements are summed.
        reference_hist: History
            The kernel will be discretized to be compatible with this history.
            (E.g. it will use the same time step.)
        params: namedtuple-like
            Must have the following attributes: Δu, N
        """
        ## Create discretized kernels
        # TODO: Once kernels can be combined, can just
        #       use A's discretize_kernel method
        # FIXME: Check with p.52, InitPopulations – pretty sure the indexing isn't quite right
        if not isinstance(θ, Iterable):
            θ = [θ]
        memory_time = max(kernel.memory_time for kernel in θ)
        dt = reference_hist.dt

        θ_dis = Series(reference_hist, 'θ_dis',
                       t0 = dt,
                       tn = memory_time+reference_hist.dt)
            # Starts at dt because memory buffer does not include current time
        θ_dis.set_update_function(
            lambda t: sum( kernel.eval(t) for kernel in θ ) )
        # HACK Currently we only support updating by one histories timestep
        #      at a time (Theano), so for kernels (which are fully computed
        #      at any time step), we index the underlying data tensor
        θ_dis.set()
        # HACK θ_dis updates should not be part of the loglikelihood's computational graph
        #      but 'already there'
        if shim.is_theano_object(θ_dis._data):
            if θ_dis._original_data in shim.config.theano_updates:
                del shim.config.theano_updates[θ_dis._original_data]
            if θ_dis._original_tidx in shim.config.theano_updates:
                del shim.config.theano_updates[θ_dis._original_tidx]

        # TODO: Use operations
        θtilde_dis = Series(θ_dis, 'θtilde_dis')
        # HACK Proper way to ensure this would be to specify no. of bins (instead of tn) to history constructor
        if len(θ_dis) != len(θtilde_dis):
            θtilde_dis._tarr = copy.copy(θ_dis._tarr)
            θtilde_dis._original_data.set_value(shim.zeros_like(θ_dis._original_data.get_value()))
            θtilde_dis._data = θtilde_dis._original_data
            θtilde_dis.tn = θtilde_dis._tarr[-1]
            θtilde_dis._unpadded_length = len(θtilde_dis._tarr)
        # HACK θ_dis._data should be θ_dis; then this can be made a lambda function
        def θtilde_upd_fn(t):
            tidx = θ_dis.get_t_idx(t)
            return params.Δu * (1 - shim.exp(-θ_dis._data[tidx]/params.Δu) ) / params.N
        θtilde_dis.set_update_function(θtilde_upd_fn)
        # self.θtilde_dis.set_update_function(
        #     lambda t: self.params.Δu * (1 - shim.exp(-self.θ_dis._data[t]/self.params.Δu) ) / self.params.N )
        θtilde_dis.add_inputs([θ_dis])
        # HACK Currently we only support updating by one histories timestep
        #      at a time (Theano), so for kernels (which are fully computed
        #      at any time step), we index the underlying data tensor
        θtilde_dis.set()
        # HACK θ_dis updates should not be part of the loglikelihood's computational graph
        #      but 'already there'
        if shim.is_theano_object(θtilde_dis._data):
            if θtilde_dis._original_data in shim.config.theano_updates:
                del shim.config.theano_updates[θtilde_dis._original_data]
            if θtilde_dis._original_tidx in shim.config.theano_updates:
                del shim.config.theano_updates[θtilde_dis._original_tidx]

        return θ_dis, θtilde_dis

    @staticmethod
    def get_stationary_activity(model, K, θ, θtilde):
        """
        Determine the stationary activity for these parameters by solving a
        self-consistency equation. For details see the notebook
        'docs/Initial_condition.ipynb'

        We make this a static method to allow external calls. In particular,
        this allows us to use this function to use this function in the
        initialization of GIF_spiking.

        TODO: Use get_η_csts rather than accessing parameters directly.

        Parameters  (not up to date)
        ----------
        params: Parameters instance
            Must be compatible with GIF_mean_field.Parameters
        dt: float
            Time step. Typically [mean field model].dt
        K: int
            Size of the memory vector. Typically [mean field model].K
        θ, θtilde: Series
            Discretized kernels θ and θtilde.
        """

        params = model.params
        dt = model.dt
        # TODO: Find something less ugly than the following. Maybe using np.vectorize ?
        class F:
            def __init__(self, model):
                self.model = model
            def __getitem__(self, α):
                def _f(u):
                    if isinstance(u, Iterable):
                        return np.array([self.model.f(ui)[α] for ui in u])
                    else:
                        return self.model.f(u)[α]
                return _f
        f = F(model)

        # Define the equation we need to solve
        k_refs = np.rint(params.t_ref / dt).astype('int')
        jarrs = [np.arange(k0, K) for k0 in k_refs]
        memory_time = K * dt
        def rhs(A):
            a = lambda α: ( np.exp(-(jarrs[α]-k_refs[α]+1)*dt/params.τ_m[α]) * (params.u_r[α] - params.u_rest[α])
                + params.u_rest[α] - params.u_th[α] - θ.get_trace()[k_refs[α]-1:K-1,α] )

            b = lambda α: ( (1 - np.exp(-(jarrs[α]-k_refs[α]+1)*dt/params.τ_m[α]))[:,np.newaxis]
                * params.τ_m[α] * params.p[α] * params.N * params.w[α] )

            # TODO: remove params.N factor once it's removed in model
            θtilde_dis = lambda α: θtilde.get_trace()[k_refs[α]-1:K-1,α] * params.N[α] # starts at j+1, ends at K incl.
            c = lambda α: params.J_θ[0,α] * np.exp(-memory_time/params.τ_θ[0,α]) + dt * np.cumsum(θtilde_dis(α)[::-1])[::-1]

            ap = lambda α: params.u_rest[α] - params.u_th[α]

            bp = lambda α: (1 - np.exp(-dt/params.τ_m[α])) * params.τ_m[α] * params.p[α] * params.N * params.w[α]

            cp = lambda α: params.J_θ[0,α] * np.exp(-memory_time/params.τ_θ[0,α])

            return ( (k_refs + 1).astype(float)
                    + np.array( [ np.exp(- f[α](a(α) + (b(α) * A).sum(axis=-1) - c(α)*A[α])[:-1].cumsum()*dt).sum()
                                for α in range(len(params.N))])
                    + np.array( [ ( np.exp(- f[α](a(α) + (b(α) * A).sum(axis=-1) - c(α)*A[α]).sum()*dt)
                                / (1 - np.exp(-f[α](ap(α) + (bp(α)*A).sum(axis=-1) - cp(α)*A[α])*dt)) )
                                for α in range(len(params.N)) ] ).flatten()
                ) * A * dt - 1

        # Solve the equation for A*
        Aguess = np.ones(len(params.N)) * 10
        res = root(rhs, Aguess)

        if not res.success:
            raise RuntimeError("Root-finding algorithm was unable to find a stationary activity.")
        else:
            return res.x

    @staticmethod
    def get_η_csts(model, K, θ, θtilde):
        """
        Returns the tensor constants which, along with the stationary activity,
        allow calculating the stationary value of each state variable. See the notebook
        'docs/Initial_condition.ipynb' for their definitions.

        Parameters (not up to date)
        ----------
        params: Parameters instance
            Must be compatible with GIF_mean_field.Parameters
        dt: float
            Time step. Typically [mean field model].dt
        K: int
            Size of the memory vector. Typically [mean field model].K
        θ, θtilde: Series
            Discretized kernels θ and θtilde.
        """

        params = model.params
        dt = model.dt

        # There are a number of factors K-1 below because the memory vector
        # doesn't include the first (current) bin
        Npop = len(params.N)
        τm = params.τ_m.flatten()
        τmT = τm[:,np.newaxis]  # transposed τ_m
        k_refs = np.rint(params.t_ref / dt).astype('int')
        jarrs = [np.arange(k0, K+1) for k0 in k_refs]
        memory_time = K*dt
        η = []
        η.append(τmT * params.p * params.N * params.w)  # η1
        η.append( (1 - np.exp(-dt/τmT)) * η[0] )        # η2
        η3 = np.empty((K, Npop))
        for α in range(Npop):
            red_factor = np.exp(-(jarrs[α]-k_refs[α]+1)*dt/params.τ_m[α])
            η3[:k_refs[α]-1, α] = params.u_r[α]
            η3[k_refs[α]-1:, α] = red_factor * (params.u_r[α] - params.u_rest[α]) + params.u_rest[α]
        η.append(η3)
        η4 = np.zeros((K, Npop))
        for α in range(Npop):
            η4[k_refs[α]-1:, α] = (1 - np.exp(- (jarrs[α] - k_refs[α] + 1)*dt / τm[α])) / (1 - np.exp(- dt / τm[α]))
        η.append(η4)
        η.append( params.u_th + θ.get_trace()[:K] )   # η5
        # TODO: remove params.N factor once it's removed in model
        η.append( params.J_θ * np.exp(-memory_time/params.τ_θ.flatten())
                  + dt * params.N*np.cumsum(θtilde.get_trace()[K-1::-1], axis=0)[::-1] )   # η6
        η.append( params.J_θ * np.exp(-memory_time/params.τ_θ.flatten()) )  # η7
        η.append( params.u_rest - params.u_th )  # η8
        η.append( η[2] - η[4] )  # η9
        η.append( η[3][..., np.newaxis] * η[1] )  # η10

        return η

    def get_silent_state(self):
        K = self.varθ.shape[0]
        state = self.State(
            h = self.params.u_rest.get_value(),
            #h_tot = np.zeros(self.h_tot.shape),
            u = self.params.u_rest.get_value(),
            #varθ = self.params.params.u_th + self.θ_dis.get_trace()[:K],
            #varθfree = self.params.u_th,
            λ = np.zeros(self.λ.shape),         # Clamp the rates to zero
            λfree = np.zeros(self.λfree.shape), # idem
            g = np.zeros(self.g.shape),
            m = np.zeros(self.m.shape),
            v = np.zeros(self.v.shape),
            x = self.params.N.get_value(),
            y = np.zeros(self.y.shape),
            z = np.zeros(self.z.shape)          # Clamped to zero
            )
        return state

    def get_stationary_state(self, Astar):

        η = self.get_η_csts(self, self.K,
                            self.θ_dis, self.θtilde_dis)
        state = self.State(
            h = self.params.u_rest + η[0].dot(Astar),
            #h_tot = η[1].dot(Astar),
            u = η[2] + η[9].dot(Astar),
            #varθ = η[4] + η[5]*Astar,
            #varθfree = self.params.u_th + η[6]*Astar,
            λ = np.stack(
                  [ self.f(u) for u in η[8] + (η[9]*Astar).sum(axis=-1) - η[5]*Astar ] ),
            λfree = self.f(η[7] + η[0].dot(Astar) - η[6]*Astar),
            # The following quantities are set below
            #P = 0,
            #Pfree = 0,
            g = Astar,
            m = 0,
            v = 0,
            x = 0,
            y = Astar,
            z = 0
            )

        λprev = np.concatenate(
            ( np.zeros((1,) + state.λ.shape[1:]),
              state.λ[:-1] )  )
        P = 1 - np.exp(-(state.λ+λprev)/2 *self.dt)
        Pfree = 1 - np.exp(-state.λfree*self.dt)

        m = np.empty((self.K, self.Npops))
        v = np.empty((self.K, self.Npops))
        m[0] = Astar * self.params.N * self.dt
        v[0, ...] = 0
        for i in range(1, self.K):
            m[i] = (1 - P[i])*m[i-1]
            v[i] = (1 - P[i])**2 * v[i-1] + P[i] * m[i-1]
        x = m[-1] / Pfree
        z = (x + v[-1]/Pfree) / (2 - Pfree)
        state = state._replace(
            m = m,
            v = v,
            x = x,
            z = z
            )

        return state

    def advance(self, stop):

        if stop == 'end':
            stop = self.nbar.tnidx

        stopidx = self.nbar.get_t_idx(stop + self.nbar.t0idx)

        if not shim.config.use_theano:
            self.nbar[stopidx]

        else:
            if not hasattr(self, '_advance_fn'):
                logger.info("Compiling advance function.")
                curtidx_var = shim.getT().lscalar()
                stopidx_var = shim.getT().lscalar()
                self.remove_other_histories()  # HACK
                self.clear_unlocked_histories()
                self.theano_reset()

                def onestep(tidx, *args):
                    statevar_upds, input_vars, output_vars = self.symbolic_update(tidx, args)
                    return list(statevar_upds.values()), {}

                outputs_info = []
                for hist in self.statehists:
                    outputs_info.append( hist._data[curtidx_var + hist.t0idx])
                    # HACK-y !!
                    if hist.name == 'v':
                        outputs_info[-1] = shim.getT().unbroadcast(outputs_info[-1], 1)
                    elif hist.name == 'z':
                        outputs_info[-1] = shim.getT().unbroadcast(outputs_info[-1], 0)

                outputs, upds = shim.gettheano().scan(onestep,
                                                      sequences = shim.getT().arange(curtidx_var+1, stopidx_var),
                                                      outputs_info = outputs_info)
                self.apply_updates(upds)
                    # Applying updates ensures we remove the iteration variable
                    # scan introduces from the shim updates dictionary

                self._advance_fn = shim.gettheano().function([curtidx_var, stopidx_var],
                                                             outputs)

                self.theano_reset()
                logger.info("Done.")

            curtidx = min( hist._original_tidx.get_value() - hist.t0idx
                           for hist in self.statehists )

            if curtidx+1 < stopidx:
                newvals = self._advance_fn(curtidx, stopidx)
                # HACK: We change the history directly to avoid dealing with updates
                for hist, newval in zip(self.statehists, newvals):
                    valslice = slice(curtidx+1+hist.t0idx, stopidx+hist.t0idx)

                    data = hist._original_data.get_value(borrow=True)
                    data[valslice] = newval
                    hist._original_data.set_value(data, borrow=True)
                    hist._data = hist._original_data

                    hist._original_tidx.set_value( valslice.stop - 1 )
                    hist._cur_tidx = hist._original_tidx


    def remove_other_histories(self):
        """HACK: Remove histories from sinn.inputs that are not in this model."""
        histnames = [h.name for h in self.history_set]
        dellist = []
        for h in sinn.inputs:
            if h.name not in histnames:
                dellist.append(h)
        for h in dellist:
            del sinn.inputs[h]

    def loglikelihood(self, start, batch_size):

        ####################
        # Some hacks to get around current limitations

        self.remove_other_histories()

        # End hacks
        #####################

        startidx = self.get_t_idx(start)
        stopidx = startidx + batch_size
        N = self.params.N

        # Windowed test
        #windowlen = 5
        #stopidx -= windowlen

        def logLstep(tidx, *args):
            if shim.is_theano_object(tidx):
                statevar_updates, input_vars, output_vars = self.symbolic_update(tidx, args[1:])
                nbar = output_vars[self.nbar]
            else:
                nbar = self.nbar[tidx+self.nbar.t0idx]
                #nbar = self.nbar[tidx+self.nbar.t0idx-windowlen:tidx+self.nbar.t0idx].sum(axis=0)
                statevar_updates = {}
                updates = shim.get_updates()
            p = sinn.clip_probabilities(nbar / self.params.N)
            n = shim.cast(self.n[tidx+self.n.t0idx], 'int32')
            #n = shim.cast(self.n[tidx+self.n.t0idx-windowlen:tidx+self.n.t0idx].sum(axis=0), 'int32')

            cum_logL = args[0] + ( -shim.gammaln(n+1) - shim.gammaln(N-n+1)
                                   + n*shim.log(p)
                                   + (N-n)*shim.log(1-p)
                                  ).sum()

            return [cum_logL] + list(statevar_updates.values()), {}
            # return [cum_logL], shim.get_updates()

        if shim.is_theano_object([self.nbar._data, self.params, self.n._data]):
            logger.info("Producing the likelihood graph.")

            # Create the outputs_info list
            # First element is the loglikelihood, subsequent are aligned with input_vars
            outputs_info = [shim.cast(0, sinn.config.floatX)]
            for hist in self.statehists:
                outputs_info.append( hist._data[startidx + hist.t0idx - 1] )
                # HACK !!
                if hist.name == 'v':
                    outputs_info[-1] = shim.getT().unbroadcast(outputs_info[-1], 1)
                elif hist.name == 'z':
                    outputs_info[-1] = shim.getT().unbroadcast(outputs_info[-1], 0)
            if batch_size == 1:
                # No need for scan
                outputs, upds = logLstep(start, *outputs_info)
                outputs[0] = [outputs[0]]

            else:
                # FIXME np.float64 -> shim.floatX or sinn.floatX
                outputs, upds = shim.gettheano().scan(logLstep,
                                                      sequences = shim.getT().arange(startidx, stopidx),
                                                      outputs_info = outputs_info)
                                                      #outputs_info = np.float64(0))
                # HACK Since we are still using shared variables for data
                #for hist, new_data in outputs[1:]:
                #    hist.update(slice(startidx+hist.t0idx, stopidx+hist.t0idx),
                #                new_data)

                self.apply_updates(upds)
                    # Applying updates is essential to remove the iteration variable
                    # scan introduces from the shim updates dictionary

            logger.info("Likelihood graph complete")

            return outputs[0][-1], outputs[1:], upds
                # logL = outputs[0]; outputs[1:] => statevars
        else:
            # TODO: Remove this branch once shim.scan is implemented
            logL = 0
            for t in np.arange(startidx, stopidx):
                logL = logLstep(t, logL)[0][0]
            upds = shim.get_updates()

            return logL, upds

    def f(self, u):
        """Link function. Maps difference between membrane potential & threshold
        to firing rate."""
        return self.params.c * shim.exp(u/self.params.Δu.flatten())

    def A_fn(self, t):
        """p. 52"""
        return self.n[t] / (self.params.N * self.A.dt)

    def h_fn(self, t):
        """p.53, also Eq. 92 p. 48"""

        tidx_h = self.h.get_t_idx(t)
        #tidx_h_tot = self.h_tot.get_t_idx(t)
        red_factor = shim.exp(-self.h.dt/self.params.τ_m.flatten() )
        return ( (self.h[tidx_h-1] - self.params.u_rest) * red_factor
                 + self.h_tot[t] )

    def A_Δ_fn(self, t):
        """p.52, line 9"""
        tidx_A = self.A.get_t_idx(t)
        # a = lambda α: [ self.A[tidx_A - self.Δ_idx[α, β]][β:β+1, np.newaxis]
        #                 for β in range(self.Npops) ]
        # b = lambda α: shim.concatenate( a(α), axis=1)
        # c = [ b(α) for α in range(self.Npops) ]
        # d = shim.concatenate( c, axis = 0)

        return shim.concatenate(
            [ shim.concatenate(
                [ self.A[tidx_A - self.Δ_idx[α, β]][..., β:β+1, np.newaxis]  # make scalar 2d
                  for β in range(self.Npops) ],
                axis = -1)
              for α in range(self.Npops) ],
            axis=-2)

    def h_tot_fn(self, t):
        """p.52, line 10, or Eq. 94, p. 48
        Note that the pseudocode on p. 52 includes the u_rest term, whereas in Eq. 94
        this term is instead included in the equation for h (Eq. 92). We follow the pseudocode here.
        """
        # FIXME: Check again that indices are OK (i.e. should they be ±1 ?)
        τ_m = self.params.τ_m.flatten()[:,np.newaxis]
           # We have τ_sβ, but τ_mα. This effectively transposes τ_m
        red_factor_τm = shim.exp(-self.h_tot.dt/self.params.τ_m)
        red_factor_τmT = shim.exp(-self.h_tot.dt/τ_m)
        red_factor_τs = shim.exp(-self.h_tot.dt/self.params.τ_s)
        return ( self.params.u_rest + self.params.R*self.I_ext[t] * (1 - red_factor_τm)
                 + ( τ_m * (self.params.p * self.params.w) * self.params.N
                       * (self.A_Δ[t]
                          + ( ( self.params.τ_s * red_factor_τs * ( self.y[t] - self.A_Δ[t] )
                                - red_factor_τmT * (self.params.τ_s * self.y[t] - τ_m * self.A_Δ[t]) )
                              / (self.params.τ_s - τ_m) ) )
                   ).sum(axis=-1) )

    def y_fn(self, t):
        """p.52, line 11"""
        tidx_y = self.y.get_t_idx(t)
        red_factor = shim.exp(-self.y.dt/self.params.τ_s)
        return self.A_Δ[t] + (self.y[tidx_y-1] - self.A_Δ[t]) * red_factor

    # TODO: g and varθ: replace flatten by sum along axis=1

    def g_fn(self, t):
        """p. 53, line 5, also p. 45, Eq. 77b"""
        tidx_g = self.g.get_t_idx(t)
        tidx_n = self.n.get_t_idx(t)
        # TODO: cache the reduction factor
        # FIXME: Not sure if this should be tidx_n-self.K-1
        red_factor = shim.exp(- self.g.dt/self.params.τ_θ)  # exponential reduction factor
        return ( self.g[tidx_g-1] * red_factor
                 + (1 - red_factor) * self.n[tidx_n-self.K] / (self.params.N * self.g.dt)
                ).flatten()

    def varθfree_fn(self, t):
        """p. 53, line 6 and p. 45 Eq. 77a"""
        #tidx_varθ = self.varθ.get_t_idx(t)
        # TODO: cache reduction factor
        red_factor = (self.params.J_θ * shim.exp(-self.memory_time/self.params.τ_θ)).flatten()
        return self.params.u_th + red_factor * self.g[t]
            # TODO: sum over exponentials (l) of the threshold kernel

    def λfree_fn(self, t):
        """p. 53, line 8"""
        # FIXME: 0 or -1 ?
        return self.f(self.h[t] - self.varθfree[t][0])

    def Pfree_fn(self, t):
        """p. 53, line 9"""
        tidx_λ = self.λfree.get_t_idx(t,)
        self.λfree.compute_up_to(tidx_λ)
            # HACK: force Theano to compute up to tidx_λ first
            #       This is required because of the hack in History.compute_up_to
            #       which assumes only one update per history is required
        return 1 - shim.exp(-0.5 * (self.λfree[tidx_λ-1] + self.λfree[tidx_λ]) * self.Pfree.dt )

    def X_fn(self, t):
        """p. 53, line 12"""
        #tidx_m = self.m.get_t_idx(t)
        return (self.m[t]).sum(axis=-2)
            # axis 0 is for lags, axis 1 for populations
            # FIXME: includes the absolute ref. lags

    def varθ_fn(self, t):
        """p.53, line 11, 15 and 16, and Eqs. 97, 98 (p.48)"""
        # FIXME: does not correctly include cancellation from line 11
        # FIXME: t-self.K+1:t almost certainly wrong
        tidx_n = self.n.get_t_idx(t)
        K = self.u.shape[0]
        # DEBUG
        #if shim.cf.use_theano:
        #    K = shim.print(K, "K : ")
        #θtilde = shim.print(self.θtilde_dis._data, "θtilde data")  # DEBUG
        # HACK: use of ._data to avoid indexing θtilde (see comment where it is created)
        # TODO: Exclude the last element from the sum, rather than subtracting it.
        varθref = ( shim.cumsum(self.n[tidx_n-K:tidx_n]*self.θtilde_dis._data[:K][...,::-1,:],
                                axis=-2)
                    - self.n[tidx_n-K:tidx_n]*self.θtilde_dis._data[:K][...,::-1,:])[...,::-1,:]
        # FIXME: Use indexing that is robust to θtilde_dis' t0idx
        # FIXME: Check that this is really what line 15 says
        return self.θ_dis._data[:K] + self.varθfree[t] + varθref

    def u_fn(self, t):
        """p.53, line 17 and 35"""
        tidx_u = self.u.get_t_idx(t)
        red_factor = shim.exp(-self.u.dt/self.params.τ_m).flatten()[np.newaxis, ...]
        return shim.concatenate(
            ( self.params.u_r[..., np.newaxis, :],
              ((self.u[tidx_u-1][:-1] - self.params.u_rest[np.newaxis, ...]) * red_factor + self.h_tot[t][np.newaxis,...]) ),
            axis=-2)

    #def λtilde_fn(self, t):
    def λ_fn(self, t):
        """p.53, line 18"""
        return self.f(self.u[t] - self.varθ[t]) * self.ref_mask

    def P_λ_fn(self, t):
        """p.53, line 19"""
        tidx_λ = self.λ.get_t_idx(t)
        self.λ.compute_up_to(tidx_λ)  # HACK: see Pfree_fn
        λprev = np.concatenate(
            ( np.zeros((1,) + self.λ.shape[1:]),
              self.λ[tidx_λ-1][:-1] )  )
        P_λ = 0.5 * (self.λ[tidx_λ][:] + λprev) * self.P_λ.dt
        return shim.switch(P_λ <= 0.01,
                           P_λ,
                           1 - shim.exp(-P_λ))

    # def λ_fn(self, t):
    #     """p.53, line 21 and 36"""
    #     return self.λtilde[t]
    #         # FIXME: check that λ[t] = 0 (line 36)

    def Y_fn(self, t):
        """p.53, line 22"""
        #tidx_Y = self.Y.get_t_idx(t)
        tidx_v = self.v.get_t_idx(t)
        return (self.P_λ[t] * self.v[tidx_v - 1]).sum(axis=-2)
            # FIXME: includes abs. refractory lags

    def Z_fn(self, t):
        """p.53, line 23"""
        #tidx_Z = self.Z.get_t_idx(t)
        tidx_v = self.v.get_t_idx(t)
        return self.v[tidx_v-1].sum(axis=-2)
            # FIXME: includes abs. refractory lags

    def W_fn(self, t):
        """p.53, line 24"""
        #tidx_W = self.W.get_t_idx(t)
        #tidx_m = self.m.get_t_idx(t)
        ref_mask = self.ref_mask[:self.m.shape[0],:]
            # ref_mask is slightly too large, so we truncate it
        return (self.P_λ[t] * self.m[t] * ref_mask).sum(axis=-2)
            # FIXME: includes abs. refractory lags

    def v_fn(self, t):
        """p.53, line 25 and 34"""
        tidx_v = self.v.get_t_idx(t)
        tidx_m = self.m.get_t_idx(t)
        return shim.concatenate(
            ( shim.zeros( (1,) + self.v.shape[1:], dtype=sinn.config.floatX),
              (1 - self.P_λ[t][1:])**2 * self.v[tidx_v-1][:-1] + self.P_λ[t][1:] * self.m[tidx_m-1][:-1]
            ),
            axis=-2)

    def m_fn(self, t):
        """p.53, line 26 and 33"""
        tidx_m = self.m.get_t_idx(t)
        tidx_Pλ = self.P_λ.get_t_idx(t)
        tidx_n = self.n.get_t_idx(t)
        # TODO: update m_0 with n(t)
        return shim.concatenate(
            ( self.n[tidx_n-1][np.newaxis,:],
              ((1 - self.P_λ._data[tidx_Pλ][1:]) * self.m[tidx_m-1][:-1]) ),
            axis=-2 )
            # HACK: Index P_λ data directly to avoid triggering its computational update before v_fn

    def P_Λ_fn(self, t):
        """p.53, line 28"""
        tidx_z = self.z.get_t_idx(t)
        z = self.z[tidx_z-1] # Hack: Don't trigger computation of z 'up to' t-1
        Z = self.Z[t]
        return shim.switch( Z + z > 0,
                            ( (self.Y[t] + self.Pfree[t]*z)
                              / (shim.abs(Z + z) + sinn.config.abs_tolerance) ),
                            0 )

    def nbar_fn(self, t):
        """p.53, line 29"""
        return ( self.W[t] + self.Pfree[t] * self.x[t]
                 + self.P_Λ[t] * (self.params.N - self.X[t] - self.x[t]) )

    def n_fn(self, t):
        """p.53, lines 30 and 33"""
        return self.rndstream.binomial( size = self.n.shape,
                                        n = self.params.N,
                                        p = sinn.clip_probabilities(self.nbar[t]/self.params.N) )

    def z_fn(self, t):
        """p.53, line 31"""
        tidx_x = self.x.get_t_idx(t)
        #tidx_v = self.v.get_t_idx(t)
        tidx_z = self.z.get_t_idx(t)
        return ( (1 - self.Pfree[t])**2 * self.z[tidx_z-1]
                 + self.Pfree[t]*self.x[tidx_x-1]
                 + self.v[t][0] )

    def x_fn(self, t):
        """p.53, line 32"""
        tidx_x = self.x.get_t_idx(t)
        tidx_m = self.m.get_t_idx(t)
        tidx_P = self.Pfree.get_t_idx(t)
        # TODO: ensure that m can be used as single time buffer, perhaps
        #       by merging the second line with m_fn update ?
        return ( (1 - self.Pfree[tidx_P]) * self.x[tidx_x-1]
                 + self.m._data[tidx_m][-1] )
            # HACK: Index P_λ, m _data directly to avoid triggering it's computational udate before v_fn


    def symbolic_update(self, tidx, statevars):
        """
        Temorary fix to get symbolic updates. Eventually sinn should
        be able to do this itself.
        """
        # T = shim.getT()

        # def create_variable(hist):
        #     data_tensor_broadcast = tuple([True if d==1 else 0 for d in hist.shape])
        #     DataType = T.TensorType(sinn.config.floatX, data_tensor_broadcast)
        #     return DataType(hist.name + '_0')

        # λfree0 = create_variable(self.λfree)
        # λ0 = create_variable(self.λ)
        # Pfree0 = create_variable(self.Pfree)
        # P_λ0 = create_variable(self.P_λ)
        # g0 = create_variable(self.g)
        # h0 = create_variable(self.h)
        # u0 = create_variable(self.u)
        # v0 = create_variable(self.v)
        # m0 = create_variable(self.m)
        # x0 = create_variable(self.x)
        # y0 = create_variable(self.y)
        # z0 = create_variable(self.z)

        # λfree0 = statevars[0]
        # λ0 = statevars[1]
        # #Pfree0 = statevars[2]
        # #P_λ0 = statevars[3]
        # g0 = statevars[2]
        # h0 = statevars[3]
        # u0 = statevars[4]
        # v0 = statevars[5]
        # m0 = statevars[6]
        # x0 = statevars[7]
        # y0 = statevars[8]
        # z0 = statevars[9]

        curstate = self.State(*statevars)

        λfree0 = curstate.λfree
        λ0 = curstate.λ
        #Pfree0 = statevars[2]
        #P_λ0 = statevars[3]
        g0 = curstate.g
        h0 = curstate.h
        u0 = curstate.u
        v0 = curstate.v
        m0 = curstate.m
        x0 = curstate.x
        y0 = curstate.y
        z0 = curstate.z

        # shared constants
        tidx_n = tidx + self.n.t0idx

        # yt
        red_factor = shim.exp(-self.y.dt/self.params.τ_s)
        yt = self.A_Δ[tidx+self.A_Δ.t0idx] + (y0 - self.A_Δ[tidx+self.A_Δ.t0idx]) * red_factor

        # htot
        τ_mα = self.params.τ_m.flatten()[:,np.newaxis]
        red_factor_τm = shim.exp(-self.h_tot.dt/self.params.τ_m)
        red_factor_τmT = shim.exp(-self.h_tot.dt/τ_mα)
        red_factor_τs = shim.exp(-self.h_tot.dt/self.params.τ_s)
        h_tot = ( self.params.u_rest + self.params.R*self.I_ext[tidx+self.I_ext.t0idx] * (1 - red_factor_τm)
                 + ( τ_mα * (self.params.p * self.params.w) * self.params.N
                       * (self.A_Δ[tidx+self.A_Δ.t0idx]
                          + ( ( self.params.τ_s * red_factor_τs * ( yt - self.A_Δ[tidx+self.A_Δ.t0idx] )
                                - red_factor_τmT * (self.params.τ_s * yt - τ_mα * self.A_Δ[tidx+self.A_Δ.t0idx]) )
                              / (self.params.τ_s - τ_mα) ) )
                   ).sum(axis=-1) )

        # ht
        red_factor = shim.exp(-self.h.dt/self.params.τ_m.flatten() )
        ht = ( (h0 - self.params.u_rest) * red_factor + h_tot )

        # ut
        red_factor = shim.exp(-self.u.dt/self.params.τ_m).flatten()[np.newaxis, ...]
        ut = shim.concatenate(
            ( self.params.u_r[..., np.newaxis, :],
              ((u0[:-1] - self.params.u_rest[np.newaxis, ...]) * red_factor + h_tot[np.newaxis,...]) ),
            axis=-2)

        # gt
        red_factor = shim.exp(- self.g.dt/self.params.τ_θ)
        gt = ( g0 * red_factor
                 + (1 - red_factor) * self.n[tidx_n-self.K] / (self.params.N * self.g.dt)
                ).flatten()

        # varθfree
        red_factor = (self.params.J_θ * shim.exp(-self.memory_time/self.params.τ_θ)).flatten()
        varθfree =  self.params.u_th + red_factor * gt

        # varθ
        K = self.u.shape[0]
        varθref = ( shim.cumsum(self.n[tidx_n-K:tidx_n] * self.θtilde_dis._data[:K][...,::-1,:],
                                axis=-2)
                              - self.n[tidx_n-K:tidx_n] * self.θtilde_dis._data[:K])[...,::-1,:]
        varθ = self.θ_dis._data[:K] + varθfree + varθref

        # λt
        λt = self.f(ut - varθ) * self.ref_mask

        # λfree
        λfreet = self.f(ht - varθfree[0])

        # Pfreet
        Pfreet = 1 - shim.exp(-0.5 * (λfree0 + λfreet) * self.λfree.dt )

        # P_λt
        λprev = shim.concatenate(
            ( shim.zeros((1,) + self.λ.shape[1:]),
              λ0[:-1] ) )
        P_λ_tmp = 0.5 * (λt + λprev) * self.P_λ.dt
        P_λt = shim.switch(P_λ_tmp <= 0.01,
                           P_λ_tmp,
                           1 - shim.exp(-P_λ_tmp))
        # mt
        mt = shim.concatenate(
            ( self.n[tidx_n-1][np.newaxis,:], ((1 - P_λt[1:]) * m0[:-1]) ),
            axis=-2 )

        # X
        X = mt.sum(axis=-2)

        # xt
        xt = ( (1 - Pfreet) * x0 + mt[-1] )

        # zt
        zt = ( (1 - Pfreet)**2 * z0  +  Pfreet*x0  + vt[0] )

        # vt
        vt = shim.concatenate(
            ( shim.zeros( (1,) + self.v.shape[1:] , dtype=sinn.config.floatX),
              (1 - P_λt[1:])**2 * v0[:-1] + P_λt[1:] * m0[:-1] ),
            axis=-2)

        # W
        Wref_mask = self.ref_mask[:self.m.shape[0],:]
        W = (P_λt * mt * Wref_mask).sum(axis=-2)

        # Y
        Y = (P_λt * v0).sum(axis=-2)

        # Z
        Z = v0.sum(axis=-2)

        # P_Λ
        P_Λ = shim.switch( Z + z0 > 0,
                           ( (Y + Pfreet*z0)
                             / (shim.abs(Z + z0) + sinn.config.abs_tolerance) ),
                           0 )

        # nbar
        nbar = ( W + Pfreet * xt + P_Λ * (self.params.N - X - xt) )

        newstate = State(
            h = ht,
            u = ut,
            λ = λt,
            λfree = λfreet,
            g = gt,
            m = mt,
            v = vt,
            x = xt,
            y = yt,
            z = zt
            )

        updates = OrderedDict( (getattr(curstate, key), getattr(newstate, key))
                               for key in self.State._fields )

        # TODO: use the key string itself
        input_vars = OrderedDict( (getattr(self, key), getattr(curstate, key))
                                  for key in self.State._fields )

        # Output variables contain updates to the state variables, as well as
        # whatever other quantities we want to compute
        output_vars = OrderedDict( (getattr(self, key), getattr(newstate, key))
                                   for key in self.State._fields )
        output_vars[self.nbar] = nbar

        # updates = OrderedDict((
        #     (λfree0, λfreet),
        #     (λ0, λt),
        #     #(Pfree0, Pfreet),
        #     #(P_λ0, P_λt),
        #     (g0, gt),
        #     (h0, ht),
        #     (u0, ut),
        #     (v0, vt),
        #     (m0, mt),
        #     (x0, xt),
        #     (y0, yt),
        #     (z0, zt)
        # ))

        # input_vars = OrderedDict(( (self.λfree, λfree0),
        #                            (self.λ, λ0),
        #                            #(self.Pfree, Pfree0),
        #                            #(self.P_λ, P_λ0),
        #                            (self.g, g0),
        #                            (self.h, h0),
        #                            (self.u, u0),
        #                            (self.v, v0),
        #                            (self.m, m0),
        #                            (self.x, x0),
        #                            (self.y, y0),
        #                            (self.z, z0) ))

        # output_vars = OrderedDict(( (self.λfree, λfreet),
        #                             (self.λ, λt),
        #                             #(self.Pfree, Pfreet),
        #                             #(self.P_λ, P_λt),
        #                             (self.g, gt),
        #                             (self.h, ht),
        #                             (self.u, ut),
        #                             (self.v, vt),
        #                             (self.m, mt),
        #                             (self.x, xt),
        #                             (self.y, yt),
        #                             (self.z, zt),
        #                             (self.nbar, nbar) ))

        return updates, input_vars, output_vars
