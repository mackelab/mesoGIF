import os.path
import inspect

import simpleeval
import ast
import operator as op

import mackelab as ml
import mackelab.iotools

import theano_shim as shim
import sinn.history_functions

from fsGIF import core
logger = core.logger

class ParameterError(ValueError):
    pass

hist_types = { histname: histtype
               for histname, histtype in inspect.getmembers(sinn.history_functions)
               if inspect.isclass(histtype)
                 and issubclass(histtype, sinn.history_functions.SeriesFunction)
                 and histtype is not sinn.history_functions.SeriesFunction }

# Can add elements to hist_type here

def generate_input(mgr):

    params = mgr.params
    seed = params.seed
    rndstream = core.get_random_stream(seed)

    hists = {}
    for histname in params.inputs:
        if histname in hists:
            raise ParameterError("Parameter '{}' appears twice in the list of inputs."
                                 .format(histname))
        _hist_params = getattr(params, histname)
        try:
            HistType = hist_types[_hist_params.function]
        except KeyError:
            raise ParameterError("'{}' is not a recognized history function."
                                 .format(_hist_params.function))
        hist_params = { name: param
                        for name, param in _hist_params.items()
                        if name != 'function' }
        if HistType.requires_rng:
            assert('random_stream' not in hist_params)
            hist_params['random_stream'] = rndstream
        hists[histname] = HistType(name=histname,
                                   t0=params.t0, tn=params.tn, dt=params.dt,
                                   **hist_params)

    # Replace the "safe" operators with their standard forms
    # (simpleeval implements safe_add, safe_mult, safe_exp, which test their
    #  input but this does not work with sinn histories)
    operators = simpleeval.DEFAULT_OPERATORS
    operators.update(
        {ast.Add: op.add,
         ast.Mult: op.mul,
         ast.Pow: op.pow}),
    input_hist = simpleeval.simple_eval(params.eval,
                                        operators=operators,
                                        names=hists)
    input_hist.compute_up_to('end')

    return input_hist


if __name__ == "__main__":
    core.init_logging_handlers()
    mgr = core.RunMgr(description="Generate input", calc='input')
    mgr.load_parameters()
    pathname = mgr.get_pathname(label='')

    try:
        mgr.load(pathname)
    except (core.FileNotFound, core.FileRenamed):
        # Get pathname with run label
        if mgr.args.debug:
            pathname = "input_debug.npr"
        else:
            pathname = mgr.get_pathname(label=None)
        # Generate input
        input_hist = generate_input(mgr)
        # Save to file
        ml.iotools.save(pathname, input_hist)
