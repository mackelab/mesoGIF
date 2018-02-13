import numpy as np
import pymc3 as pymc
from parameters import ParameterSet
import mackelab as ml
import mackelab.iotools
"""
This test will run a short MCMC chain starting at the maximum likelihood estimate (MLE).
We are testing to see if the log posterior used by PyMC3 corresponds to the one used
to obtain the MLE: if that is the case, the chain should remain in the vicinity of the
starting point. If the chain moves away from the starting point, then there is an issue
with the log posterior that needs to be addressed (otherwise any computed posterior will
be garbage).

The chain is purposely extremely short in order to make the test run in a reasonable time.
It is definitely not suitable for inferring any statistics.
"""

import matplotlib.pyplot as plt
import mackelab.parameters
import mackelab.pymc3
import fsGIF.mcmc as mcmc

mcmc.data_dir = "test-data"

def mcmc_test():
    pathname = "out/mcmc_test.dill"
    try:
        trace = ml.pymc3.import_multitrace(ml.iotools.load(pathname))
    except FileNotFoundError:
        params = ParameterSet("mcmc_test_params/mcmc.params")
        params = ml.parameters.params_to_arrays(params)
        model = mcmc.get_model_new(params.posterior)
        pymc_model, priors = mcmc.get_pymc_model_new(params.posterior, model)

        # Start the chain at the MLE
        mlestr = ml.iotools.load("mcmc_test_params/mle_start.repr")
        mles = ParameterSet(eval(mlestr, {'array': np.array}))
            # FIXME: Don't use eval()
        start = {}
        for prior in priors.values():
            mle = mles[prior.transform.names.orig]
            if prior.mask is not None:
                mle = mle[prior.mask]
            start[prior.transform.names.new] = prior.transform.to(mle)

        # Run the MCMC chain
        kwds = params.sampler
        kwds['start'] = start
        with pymc_model:
            trace = pymc.sample(**kwds)

        ml.iotools.save(pathname, ml.pymc3.export_multitrace(trace), format='dill')
    else:
        print("Precomputed trace found; to rerun test, delete file '{}'."
              .format(pathname))

    pymc.traceplot(trace)
    plt.show()


if __name__ == "__main__":
    mcmc_test()
