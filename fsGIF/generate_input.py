import os.path
import inspect

import simpleeval
import ast
import operator as op
import itertools

import mackelab as ml
import mackelab.iotools

import theano_shim as shim
from sinn.histories import Series
import sinn.history_functions
import sinn.models

from fsGIF import core
import fsGIF.history_functions
logger = core.logger

class ParameterError(ValueError):
    pass

possible_hist_types = itertools.chain(
    inspect.getmembers(sinn.history_functions),
    inspect.getmembers(fsGIF.history_functions))
hist_types = { histname: histtype
               for histname, histtype in
               possible_hist_types
               if inspect.isclass(histtype)
                 and issubclass(histtype, sinn.history_functions.SeriesFunction)
                 and histtype is not sinn.history_functions.SeriesFunction }

# Can add elements to hist_type here

def generate_input(params):

    seed = params.seed
    rndstream = core.get_random_stream(seed)

    hists = {}
    for histname in params.inputs:
        if histname in hists:
            raise ParameterError("Parameter '{}' appears twice in the list of inputs."
                                 .format(histname))
        _hist_params = getattr(params, histname).copy()

        if hasattr(_hist_params, 'function') == hasattr(_hist_params, 'model'):
            raise SyntaxError("History description must have either a "
                              " `function` or a `model` entry, but not both.")
        if hasattr(_hist_params, 'function'):
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
                                       dtype=shim.config.floatX,
                                       **hist_params)

        else:
            # Get the model class
            modelname = _hist_params.pop('model')
            Model = sinn.models.get_model(modelname)
            if Model is None:
                raise ParameterError("{} is not a recognized model name."
                                     .format(modelname))
            modelparams = _hist_params.pop('params')
            assert(len(modelparams) == 0)
                # FIXME: Model parameters are not currently supported.
                # Use ParameterSet for model parameters, then we can just
                # pass them along.
            modelparams = Model.Parameters()
                # Remove this line once we use ParameterSets
            shape = tuple(_hist_params.pop('shape'))
            init = _hist_params.pop('init_cond', None)

            # Create the Series history
            # TODO: Allow for 'iterative=False' option ?
            hist = Series(name=histname, shape=shape,
                          t0=params.t0, tn=params.tn, dt=params.dt,
                          dtype=shim.config.floatX)

            # Get random number generator
            if Model.requires_rng:
                _hist_params.random_stream = core.get_random_stream()

            # Create the model
            # FIXME: Only models with one public history are currently supported
            model = Model(modelparams, hist, **_hist_params)

            # Initialize it
            model.initialize(init)

            # Finally, add the appropriate history to the list
            hists[histname] = model.public_histories[0]

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
        input_hist = generate_input(mgr.params)
        # Save to file
        ml.iotools.save(pathname, input_hist)
