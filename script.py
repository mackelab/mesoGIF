import logging
import multiprocessing

import sinn

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


def fit():
    import numpy as np
    import main

    seed = np.random.choice([0, 100, 200, 300])

    print("fit for data seeded {}".format(seed))
    input_filename = "data/short_adap/spikes/fsgif_sin-input_10s_{:0>3}seed".format(seed)
    batch = 100
    burnin = 0.5
    datalen = 8.0
    lr = 0.0005
    output_filename = ("data/short_adap/fits/fit_{}s_{}lr_{}batch_{:0>3}seed.sir"
                       .format(int(datalen), str(lr)[2:], batch, seed))

    main.load_theano()
    main.gradient_descent(input_filename, batch, output_filename,
                          burnin, datalen, lr,
                          Nmax=5e4,
                          init_vals='random')

    print("Done.")

if __name__ == '__main__':

    _init_logging_handlers()
    reslst = []
    with multiprocessing.Pool(12) as pool:
        for i in range(48):
            reslst.append(pool.apply_async(fit))

        pool.close()
        pool.join() # wait for processes to exit

    for res in reslst:
        print(res.get())

