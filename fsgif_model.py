# -*- coding: utf-8 -*-
"""
Created Wed May 24 2017

author: Alexandre René
"""

import numpy as np
import scipy as sp
from collections import namedtuple, OrderedDict
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
        return Kernel_θ1.Parameters(
            height = (shim.cf.inf, shim.cf.inf),
            start  = 0,
            stop   = model_params.t_ref
        )

    def __init__(self, name, params=None, shape=None, **kwargs):
        kern_params = self.get_kernel_params(params)
        memory_time = shim.asarray(kern_params.stop - kern_params.start).max()
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


    def __init__(self, params, spike_history, input_history,
                 random_stream=None, memory_time=None):

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
        self.u.pad(1)
        self.t_hat.pad(1)

        # Expand the parameters to treat them as neural parameters
        # Original population parameters are kept as a copy
        # self.pop_params = copy.copy(self.params)
        ExpandedParams = namedtuple('ExpandedParams', ['u_rest', 't_ref', 'u_r'])
        self.expanded_params = ExpandedParams(
            u_rest = self.expand_param(self.params.u_rest, self.params.N),
            t_ref = self.expand_param(self.params.t_ref, self.params.N),
            u_r = self.expand_param(self.params.u_r, self.params.N)
        )

        # Initialize last spike time such that it was effectively "at infinity"
        idx = self.t_hat.t0idx - 1; assert(idx >= 0)
        data = self.t_hat._data.get_value(borrow=True)
        data[idx, :] = shim.ones(self.t_hat.shape) * self.memory_time * self.t_hat.dt
        self.t_hat._data.set_value(data, borrow=True)

        # Initialize membrane potential to u_rest
        idx = self.u.t0idx - 1; assert(idx >= 0)
        data = self.t_hat._data.get_value(borrow=True)
        data[idx, :] = self.expanded_params.u_rest
        self.u._data.set_value(data, borrow=True)

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

    def __init__(self, params, activity_history, input_history,
                 random_stream=None, memory_time=None):

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

        # Set model default attributes
        self.dt = self.A.dt

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
        self.A.pad(self.Δ_idx.max() + 1) # +1 HACK

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
        self.u = Series(self.A, 'u', shape=(self.K+1, self.Npops))
            # self.u[t][0] is the array of membrane potentials at time t, at lag Δt, of each population
            # TODO: Remove +1: P_λ_fn doesn't need it anymore
        self.varθ = Series(self.u, 'varθ')
        self.λ = Series(self.u, 'λ')

        # Temporary variables
        self.nbar = Series(self.n, 'nbar')
        self.A_Δ = Series(self.A, 'A_Δ', shape=(self.Npops, self.Npops))
        self.A_Δ.pad(1)  # +1 HACK (safety)
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

        # Initialize the variables
        self.init_populations()
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
        phist = self.nbar / self.params.N
        self.loglikelihood = self.make_binomial_loglikelihood(
            self.n, self.params.N, phist, approx='low p')
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

    def f(self, u):
        """Link function. Maps difference between membrane potential & threshold
        to firing rate."""
        return self.params.c * shim.exp(u/self.params.Δu.flatten())

    def get_t_idx(self, t):
        """
        Returns the time index corresponding to t such that it is compatible
        with loglikelihood / simulation timestep.
        """
        return self.A.get_t_idx(t)
    def index_interval(self, Δt):
        return self.A.index_interval(Δt)


    def get_memory_time(self, kernel, max_time=10):
        """
        Based on GetHistoryLength (p. 52). We set a global memory_time, rather
        than a population specific one; this is much easier to vecorize.

        Parameters
        ----------
        max_time: float
            Maximum allowable memory time, in seconds.
        """
        T = float(max_time)
        while (kernel.eval(T) < 0.1 * self.Δ_idx).all() and T > self.A.dt:
            T -= self.A.dt

        T = max(T, 5*self.params.τ_m.get_value().max(), self.A.dt)
        K = self.index_interval(T)
        return T, K

    def init_populations(self):
        """
        Based on InitPopulations (p. 52)

        TODO: Call this every time the model is updated
        """
        # FIXME: Initialize series' to 0

        ## Create discretized kernels
        # TODO: Once kernels can be combined, can just
        #       use A's discretize_kernel method
        # FIXME: Check with p.52, InitPopulations – pretty sure the indexing isn't quite right
        self.θ_dis = Series(self.A, 'θ_dis',
                            t0 = self.dt,
                            tn = self.θ2.memory_time+self.dt)
            # Starts at dt because memory buffer does not include current time
        self.θ_dis.set_update_function(
            lambda t: self.θ1.eval(t) + self.θ2.eval(t) )
        # HACK Currently we only support updating by one histories timestep
        #      at a time (Theano), so for kernels (which are fully computed
        #      at any time step), we index the underlying data tensor
        self.θ_dis.set()

        # TODO: Use operations
        self.θtilde_dis = Series(self.θ_dis, 'θtilde_dis',)
        # DEBUG (was lambda)
        # HACK θ_dis._data should be θ_dis
        def θtilde_upd_fn(t):
            tidx = self.θ_dis.get_t_idx(t)
            return self.params.Δu * (1 - shim.exp(-self.θ_dis._data[tidx]/self.params.Δu) ) / self.params.N
        self.θtilde_dis.set_update_function(θtilde_upd_fn)
        # self.θtilde_dis.set_update_function(
        #     lambda t: self.params.Δu * (1 - shim.exp(-self.θ_dis._data[t]/self.params.Δu) ) / self.params.N )
        self.θtilde_dis.add_inputs([self.θ_dis])
        # HACK Currently we only support updating by one histories timestep
        #      at a time (Theano), so for kernels (which are fully computed
        #      at any time step), we index the underlying data tensor
        self.θtilde_dis.set()

        ## Initialize variables that are defined through an ODE
        for series in [self.n, self.m, self.u, self.v, self.λ, self.P_λ]:
            series.pad(1)
            #series._data[0,:] = 0
        for var_series in [self.Pfree, self.P_λ, self.λfree,
                           self.x, self.z, self.h,
                           self.g, self.y]:
            var_series.pad(1)

        # Pad the the series involved in adaptation
        max_mem = self.u.shape[0]
            # The longest memories are of the size of u
        self.n.pad(max_mem)
        #self.θtilde_dis.pad(max_mem)
        self.varθ.pad(max_mem)
        self.varθfree.pad(max_mem)


        # Set initial values (make sure this is done after all padding is added)

        # ndata = self.n._data.get_value(borrow=True)
        # ndata[0] = self.params.N
        # self.n._data.set_value(ndata, borrow=True)
        # mdata = self.m._data.get_value(borrow=True)
        # mdata[0, -1, :] = self.params.N
        # self.m._data.set_value(mdata, borrow=True)

        # Make all neurons free neurons
        idx = self.x.t0idx - 1; assert(idx >= 0)
        data = self.x._data.get_value(borrow=True)
        data[idx,:] = self.params.N.get_value()
        self.x._data.set_value(data, borrow=True)

        # Set refractory membrane potential to u_rest
        idx = self.u.t0idx - 1; assert(idx >= 0)
        data = self.u._data.get_value(borrow=True)
        data[idx,:] = self.params.u_rest.get_value()
        self.u._data.set_value(data, borrow=True)

        # Set free membrane potential to u_rest
        idx = self.h.t0idx - 1; assert(idx >= 0)
        data = self.h._data.get_value(borrow=True)
        data[idx,:] = self.params.u_rest.get_value()
        self.h._data.set_value(data, borrow=True)

        #self.g_l.set_value( np.zeros((self.Npops, self.Nθ)) )
        #self.y.set_value( np.zeros((self.Npops, self.Npops)) )

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
        # FIXME: I'm pretty sure some time indices are wrong (should be ±1)
        #τs_flat = self.params.τ_s.flatten()
        #τm_flat = self.params.τ_m.flatten()
        red_factor_τm = shim.exp(-self.h_tot.dt/self.params.τ_m)
        red_factor_τs = shim.exp(-self.h_tot.dt/self.params.τ_s)
        return ( self.params.u_rest + self.params.R*self.I_ext[t] * (1 - red_factor_τm)
                 + ( self.params.τ_m * (self.params.p * self.params.w) * self.params.N
                       * (self.A_Δ[t]
                          + ( ( self.params.τ_s * red_factor_τs * ( self.y[t] - self.A_Δ[t] )
                                - red_factor_τm * (self.params.τ_s * self.y[t] - self.params.τ_m * self.A_Δ[t]) )
                              / (self.params.τ_s - self.params.τ_m) ) )
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
        tidx_λ = self.λfree.get_t_idx(t)
        return 1 - shim.exp(-0.5 * (self.λfree[tidx_λ-1] + self.λfree[tidx_λ]) * self.Pfree.dt)

    # def λfree_fn(self, t):
    #     """p. 53, line 10"""
    #     return self.λtildefree[t]

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
        varθref = ( shim.cumsum(self.n[tidx_n-K:tidx_n] * self.θtilde_dis._data[:K][...,::-1,:],
                                axis=-2)
                    - shim.addbroadcast(self.n[tidx_n-K]*self.θtilde_dis._data[K-1:K], -2) )[...,::-1,:]
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
        P_λ = 0.5 * (self.λ[tidx_λ][:-1] + self.λ[tidx_λ-1][:-1]) * self.P_λ.dt
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
        #tidx_m = self.m.get_t_idx(t)
        return shim.concatenate(
            ( shim.zeros(self.v.shape[:-1] + (1,), dtype=sinn.config.floatX),
              ((1 - self.P_λ[t])**2 * self.v[tidx_v-1] + self.P_λ[t] * self.m[t])[...,:-1]
            ),
            axis=-1)

    def m_fn(self, t):
        """p.53, line 26 and 33"""
        tidx_m = self.m.get_t_idx(t)
        tidx_Pλ = self.P_λ.get_t_idx(t)
        tidx_n = self.n.get_t_idx(t)
        # TODO: update m_0 with n(t)
        return shim.concatenate(
            ( self.n[tidx_n-1][np.newaxis,:],
              ((1 - self.P_λ[tidx_Pλ-1][:-1]) * self.m[tidx_m-1][:-1]) ),
            axis=-2 )

    def P_Λ_fn(self, t):
        """p.53, line 28"""
        tidx_z = self.z.get_t_idx(t)
        z = self.z[tidx_z-1]
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
        tidx_v = self.v.get_t_idx(t)
        tidx_z = self.z.get_t_idx(t)
        return ( (1 - self.Pfree[t])**2 * self.z[tidx_z-1]
                 + self.Pfree[t]*self.x[tidx_x-1]
                 + self.v[tidx_v - 1][0] )

    def x_fn(self, t):
        """p.53, line 32"""
        tidx_x = self.x.get_t_idx(t)
        tidx_m = self.m.get_t_idx(t)
        tidx_P = self.Pfree.get_t_idx(t)
        tidx_Pλ = self.P_λ.get_t_idx(t)
        # TODO: ensure that m can be used as single time buffer, perhaps
        #       by merging the second line with m_fn update ?
        return ( (1 - self.Pfree[tidx_P-1]) * self.x[tidx_x-1]
                 + (1 - self.P_λ[tidx_Pλ-1][-1])*self.m[tidx_m-1][-1] )


