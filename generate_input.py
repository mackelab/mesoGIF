import sys
import inspect

import parameters
import simpleeval

import theano_shim as shim
import sinn.history_functions

import core

class ParameterError(ValueError):
    pass

hist_types = { histname: histtype
               for histname, histtype in inspect.getmembers(sinn.history_functions)
               if inspect.isclass(histtype)
                 and issubclass(histtype, sinn.history_functions.SeriesFunction)
                 and histtype is not sinn.history_functions.SeriesFunction }

# Can add elements to hist_type here

def generate_input(params):

    seed = core.resolve_linked_param(params, 'seed')
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

    input_hist = simpleeval.simple_eval(params.eval, names=hists)
    input_hist.compute_up_to('end')

    pathname = core.get_pathname(core.input_subdir, params)
    sinn.iotools.saveraw(pathname, input_hist)


if __name__ == "__main__":
    params = core.load_parameters(sys.argv[1])
    generate_input(params)
