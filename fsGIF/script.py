import logging
import multiprocessing
import numbers

import sinn
import sinn.iotools as io

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
def _init_logging_handlers():
    # Only attach handlers if running as a script
    import logging.handlers
    fh = logging.handlers.RotatingFileHandler('script.log', mode='w', maxBytes=5e5, backupCount=5)
    fh.setLevel(sinn.LoggingLevels.MONITOR)
    fh.setFormatter(sinn.config.logging_formatter)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(sinn.config.logging_formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)

seed_lsts = {'sin': [0, 100, 200, 300, 314],
             'flat': [314],
             'no': [314]}

def fit(input_desc, init_seed, nparams):
    import numpy as np
    import main_1pop as main

    np.random.seed(init_seed)
    data_seed = np.random.choice(seed_lsts[input_desc])

    print("fit for data seeded {}".format(data_seed))
    input_filename = "data/short_adap/spikes/fsgif_1pop_{}-input_9s_{:0>3}seed".format(input_desc, data_seed)
    batch = 100
    burnin = 0.5
    datalen = 8.0
    lr = 0.005
    output_filename = ("data/short_adap/fits/random_init/fit_1pop_{}-input_{}s_{}lr_{}batch_{}params_{:0>3}seed.sir"
                       .format(input_desc, io.paramstr(datalen), io.paramstr(lr), io.paramstr(batch), io.paramstr(nparams), io.paramstr(data_seed)))

    main.load_theano()
    if 'derived mf model' in main.loaded:
        del main.loaded['derived mf model']

    model = main.derive_mf_model_from_spikes(input_filename, max_len=burnin+datalen)
    fitmask = main.get_fitmask(model, nparams)
    main.gradient_descent(input_filename, batch, output_filename,
                          burnin, datalen, lr,
                          Nmax=5e3,
                          fitmask=fitmask,
                          init_vals=init_seed)

    print("Done.")

if __name__ == '__main__':
    _init_logging_handlers()

    reslst = []
    with multiprocessing.Pool(6) as pool:
        for nparams in range(1, 9):
            for i in range(0, 8):
                reslst.append(pool.apply_async(fit, ['no', i, nparams]))

        pool.close()
        pool.join() # wait for processes to exit

