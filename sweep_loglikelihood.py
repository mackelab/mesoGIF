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

    if type(calc_params.burnin) != type(calc_params.datalen):
        raise ValueError("The 'burnin' and 'datalen' parameters must be of the same type.")
    burnin_idx = model.get_t_idx(calc_params.burnin, allow_rounding=True)
    stop_idx = model.get_t_idx(calc_params.burnin+calc_params.datalen, allow_rounding=True)

    if shim.config.use_theano:
        model.theano_reset()
        model.clear_unlocked_histories()
        tidx = shim.getT().lscalar('tidx')
        logL_graph, statevar_upds, shared_upds = model.loglikelihood(tidx, stop_idx-tidx)
        logger.info("Compiling Theano loglikelihood")
        logL_step = shim.gettheano().function([tidx], logL_graph)
                                              #updates=upds)
        logger.info("Done compilation.")
        model.theano_reset()
        def logL_fn(model):
            model.clear_unlocked_histories()
            logger.info("Computing state variable traces...")
            model.init_state_vars(params.model.initializer)
            model.advance(burnin_idx)
            logger.info("Computing log likelihood...")
            return logL_step(burnin_idx)
            #return sum(logL_step(i)
            #           for i in range(burnin_idx, stop_idx, mbatch_size))
    else:
        def logL_fn(model):
            return model.loglikelihood(burnin_idx, stop_idx-burnin_idx)[0]

    param_sweep.set_function(logL_fn, 'log $L$')

    # Compute the likelihood
    t1 = time.perf_counter()
    loglikelihood = param_sweep.do_sweep(output_filename)
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
    mgr.load_parameters()
    params = mgr.params

    #output_filename = core.get_pathname(core.likelihood_subdir, params)
    #data_filename = core.get_pathname(params.data.dir, params.data.params, params.data.name)
    logL_filename = mgr.get_pathname(label='')
    data_filename = mgr.get_pathname(params=params.data.params,
                                     subdir=params.data.dir,
                                     suffix=params.data.name,
                                     label='')
    input_filename = mgr.get_pathname(params=params.input.params,
                                      subdir=params.input.dir,
                                      suffix=params.input.name,
                                      label='')

    try:
        mgr.load(logL_filename)
    except (core.FileDoesNotExist, core.FileRenamed):

        data = mgr.load(data_filename,
                        cls=getattr(histories, params.data.type).from_raw,
                        calc='activity',
                        recalculate=False)
        data.lock()

        #input_filename = core.get_pathname(params.input.dir, params.input.params, params.input.name)
        #input_history = getattr(histories, params.input.type).from_raw(
        #    iotools.loadraw(input_filename))
        input_history = mgr.load(input_filename,
                                 cls=getattr(histories, params.input.type).from_raw,
                                 calc='input',
                                 recalculate=False)
        input_history.lock()

        model_params = core.get_model_params(params.model.params)
        model = getattr(gif, params.model.type)(model_params,
                                                data,
                                                input_history,
                                                initializer=params.model.initializer)

        # Get output filename with run label
        logL_filename = mgr.get_pathname()
        sweep_loglikelihood(model, params, logL_filename)

