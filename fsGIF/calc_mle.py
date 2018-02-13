import os
import numpy as np
from parameters import ParameterSet
import mackelab as ml
from mackelab import smttk
from mackelab import iotools
import mackelab.parameters
import sinn
import sinn.optimize.gradient_descent as gd

try:
    import click
except ImportError:
    click_loaded = False
else:
    click_loaded = True

from fsGIF import core
logger = core.logger
############################
# Model import
from fsGIF import fsgif_model as gif
############################

if click_loaded:

    @click.command()
    @click.argument('params', nargs=1)
    @click.option('--overwrite/--no-overwrite', default=False,
                  help="Whether to overwrite a file containing a previously calculated MLE.")
    def calc_mle(params, overwrite):
        """
        Parameters
        ----------
        params: str URL
            Path to the posterior parameters file.
        overwrite: bool
            Whether to overwrite a file containing a previously calculated MLE.

        TODO: Allow other parameter files, and automatically determine
              the required parameter root (e.g. 'posterior.model.params').
        """
        dataroot = "data"
            # HACK/TODO: Get this from sumatra project
        paramset = ParameterSet(params)
        filename = ml.parameters.get_filename(paramset) + '.repr'
        pathname = os.path.join(dataroot,
                                "MLE",
                                filename)
        if overwrite:
            do_calc = True
        else:
            do_calc = not os.path.exists(pathname)

        if not do_calc:
            logger.info("The maximum likelihood estimate for this parameter set was already calculated. "
                        "To force a computation, use the --overwrite flag.")
        else:
            recordstore = smttk.RecordStore()
            # Add the correct parameter root
            paramset = ParameterSet({'posterior': paramset})  # HACK: hard-coded 'posterior'
            # Grab the gradient descent results which used the same posterior parameters
            records = (smttk.get_records(recordstore, 'fsGIF')
                    .filter.after(2017,11,13)  # HACK
                    .filter.script('gradient')
                    .filter.parameters(paramset)).list

            if len(records) == 0:
                logger.warning("No fits were performed for runs with these parameters. "
                               "Cannot calculate MLE.")
                return

            # Begin hacky stuff
            dt = records[0].parameters.posterior.model.dt
            model_params = core.get_model_params(records[0].parameters.posterior.model.params,
                                                'GIF_mean_field')
                # All records have the same posterior params
            surrogate = sinn.models.Surrogate(gif.GIF_mean_field)(model_params, t0=0, dt=dt)
            # End hacky stuff

            # Load the records into the collection
            fitcoll = gd.FitCollection(model = surrogate)
            fitcoll.load( records.extract("parameters", "outputpath"),
                          input_format='npr' )

            iotools.save(pathname, fitcoll.MLE, format='repr')

if __name__ == '__main__':
    if click_loaded:
        calc_mle()
