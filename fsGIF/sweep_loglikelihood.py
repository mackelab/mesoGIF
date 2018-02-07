import sys
import time
from collections import Iterable
import numpy as np

import theano_shim as shim
import sinn
import sinn.histories as histories
from sinn.histories import Series, Spiketrain
from sinn.analyze import sweep
import sinn.iotools as iotools

import core
from core import logger
############################
# Model import
import fsgif_model as gif
############################

"""
Expected parameter format:
{
  param1: str,
  param2: str,
  fineness: int or [int,…],
  burnin: float or int,
  datalen: float or int (must be same type as burnin)
}
"""

def sweep_loglikelihood(model, calc_params, output_filename):

    logger.info("Computing log likelihood...")
    param_sweep = sweep.ParameterSweep(model)

    if isinstance(calc_params.fineness, Iterable):
        if len(calc_params.fineness) == 1:
            fineness = list(calc_params.fineness) * 2
        else:
            fineness = calc_params.fineness
    else:
        fineness = [calc_params.fineness] * 2

    param1 = calc_params.param1
    param2 = calc_params.param2
    param1_stops = get_param_stops(param1,
                                   fineness[0])
    param2_stops = get_param_stops(param2,
                                   fineness[1])
    param_sweep.add_param(param1.name, idx=param1.idx, axis_stops=param1_stops)
    param_sweep.add_param(param2.name, idx=param2.idx, axis_stops=param2_stops)

    if type(calc_params.posterior.burnin) != type(calc_params.posterior.datalen):
        raise ValueError("The 'burnin' and 'datalen' parameters must be of the same type.")
    burnin_idx = model.get_t_idx(calc_params.posterior.burnin, allow_rounding=True)
    stop_idx = model.get_t_idx(calc_params.posterior.burnin+calc_params.posterior.datalen, allow_rounding=True)

    if shim.config.use_theano:
        model.theano_reset()
        model.clear_unlocked_histories()
        tidx = shim.getT().lscalar('tidx')
        logL_graph, statevar_seqs, shared_upds = model.loglikelihood(tidx, stop_idx-tidx)
        logger.info("Compiling Theano loglikelihood")
        if not mgr.args.debug:
            logL_fn = shim.gettheano().function([tidx], logL_graph)
        else:
            # Also return the full sequence of state variables
            logL_fn = shim.gettheano().function(
                [tidx], [logL_graph] + statevar_seqs )
                          # [np.array(statevar_seq) for statevar_seq in statevar_seqs] ] )
        logger.info("Done compilation.")
        model.theano_reset()
        def logL_fn_wrapper(model):
            model.clear_unlocked_histories()
            logger.info("Computing state variable traces...")
            model.init_latent_vars(calc_params.posterior.model.initializer)
            model.advance(burnin_idx)
            logger.info("Computing log likelihood...")
            return logL_fn(burnin_idx)
    else:
        def logL_fn_wrapper(model):
            # if core.match_params(model.params,
            #                      ('w', (0,0), 0.958333),
            #                      ('τ_m', (1,), 0.013907)):
            #     res = model.loglikelihood(burnin_idx, stop_idx-burnin_idx)
            #     print("MLE: ", res[0])
            #     iotools.save("debug_mle", res)
            # if core.match_params(model.params,
            #                      ('w', (0,0), 0.16666),
            #                      ('τ_m', (1,), 0.019214)):
            #     res = model.loglikelihood(burnin_idx, stop_idx-burnin_idx)
            #     print("GT: ", res[0])
            #     iotools.save("debug_gt", res)
            # return 1
            res = model.loglikelihood(burnin_idx, stop_idx-burnin_idx)
            if not mgr.args.debug:
                return res[0]
            else:
                # Also return the full sequence of state variables
                return [res[0]] + res[1]
                   #[ hist[:stop_idx+hist.t0idx] for hist in res[1] ]

    param_sweep.set_function(logL_fn_wrapper, 'log $L$')

    # Compute the likelihood
    if mgr.args.debug:
        output_filename = None
    t1 = time.perf_counter()
    loglikelihood = param_sweep.do_sweep(output_filename, debug=False)
            # This can take a long time
            # The result will be saved in output_filename
    t2 = time.perf_counter()
    logger.info("Calculation of the likelihood took {}s."
                .format((t2-t1)))

    sinn.flush_log_queue()

    return loglikelihood

def get_param_stops(param, fineness):
    if param.range_desc[0][:3] == 'lin':
        return sweep.linspace(param.range_desc[1], param.range_desc[2], fineness)
    elif param.range_desc[0] == 'log':
        return sweep.logspace(param.range_desc[1], param.range_desc[2], fineness)

if __name__ == "__main__":
    core.init_logging_handlers()
    #parser = core.argparse.ArgumentParser(description="Generate activity")
    #params, flags = core.load_parameters(parser)
    mgr = core.RunMgr(description="Sweep loglikelihood", calc='logL_sweep')
    mgr.parser.add_argument('--debug', action='store_true',
                            help="Indicate to run in debug mode: disables checking for "
                            "precomputed data and does not save the result.")
    mgr.load_parameters()
    posparams = mgr.params.posterior
    sweepparams = mgr.params

    #output_filename = core.get_pathname(core.likelihood_subdir, params)
    #data_filename = core.get_pathname(params.data.dir, params.data.params, params.data.name)
    logL_filename = mgr.get_pathname(label='')
    logL_filename = core.add_extension(logL_filename)
    data_filename = mgr.get_pathname(params=posparams.data.params,
                                     subdir=posparams.data.dir,
                                     suffix=posparams.data.name,
                                     label='')
    data_filename = core.add_extension(data_filename)
    input_filename = mgr.get_pathname(params=posparams.input.params,
                                      subdir=posparams.input.dir,
                                      suffix=posparams.input.name,
                                      label='')
    input_filename = core.add_extension(input_filename)

    try:
        if mgr.args.debug:
            # Don't try to load previous data if debugging
            raise core.FileNotFound
        mgr.load(logL_filename)
    except (core.FileNotFound, core.FileRenamed):

        data = mgr.load(data_filename,
                        cls=getattr(histories, posparams.data.type).from_raw,
                        calc='activity',
                        recalculate=False)
        data = core.subsample(data, posparams.model.dt)
        data.lock()

        #input_filename = core.get_pathname(params.input.dir, params.input.params, params.input.name)
        #input_history = getattr(histories, params.input.type).from_raw(
        #    iotools.loadraw(input_filename))
        input_history = mgr.load(input_filename,
                                 cls=getattr(histories, posparams.input.type).from_raw,
                                 calc='input',
                                 recalculate=False)
        input_history = core.subsample(input_history, posparams.model.dt)
        input_history.lock()

        model_params = core.get_model_params(posparams.model.params, 'GIF_mean_field')
        model = getattr(gif, posparams.model.type)(model_params,
                                                data,
                                                input_history,
                                                initializer=posparams.model.initializer)

        # Get output filename with run label
        logL_filename = mgr.get_pathname(label=None)

        # Compute log likelihood
        logL = sweep_loglikelihood(model, sweepparams, logL_filename)


        if mgr.args.debug:
            print("Obtained logL: ", logL[0][-1])

            theanostr = '_theano' if sweepparams.theano else '_numpy'
            iotools.save('logL_debug' + theanostr, logL[0],
                         overwrite=True)

            varnames = ['n'] + list(model.State._fields)
            for varname, varseq in zip(varnames, logL[1:]):
                iotools.save('logL_debug_' + varname + theanostr, varseq,
                             overwrite=True)

            def save(var, varname):
                if isinstance(var, np.ndarray):
                    val = var
                else:
                    try:
                        val = var.get_value()
                    except AttributeError:
                        val = var.eval()
                iotools.save('logL_debug_' + varname + theanostr, val,
                             overwrite=True)

            save(model.θ_dis._data, 'θdis')
            save(model.θtilde_dis._data, 'θtilde_dis')
