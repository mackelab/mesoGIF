"""
This code sets up a default environment by importing the packages I most
typically used for this project. Typical use is to type

    from fsGIF.init import *
    %matplotlib inline

in the first cell of a Jupyter notebook. The import code will be printed,
so as to document which packages are imported, in the same way writing
all these imports in the first cell documents your namespace. Using this
package just avoids you actually having to type them ;-).
"""

#TODO: Get 'home' path from a global config

init_code = """
import copy
import os.path
import itertools
from collections import Iterable, OrderedDict, namedtuple, deque
import numpy as np
import scipy as sp
import scipy.stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image
from matplotlib.gridspec import GridSpec
import pymc3 as pymc
import pandas as pd
from pandas import DataFrame
from parameters import ParameterSet

from importlib import reload

import theano_shim as shim

import mackelab as ml
import mackelab.smttk as smttk
import mackelab.iotools
import mackelab.pymc3
import mackelab.parameters
import mackelab.plot
ml.parameters.Transform.namespaces['shim'] = shim

import sinn
import sinn.histories as histories
import sinn.models
from sinn.optimize.gradient_descent import FitCollection
import sinn.analyze as anlz
from sinn.analyze.axisdata import LogLikelihood, Likelihood, Probability
from sinn.analyze.axis import Axis

import fsGIF.core as core
import fsGIF.fsgif_model as fsgif_model
from fsGIF.fsgif_model import GIF_mean_field, GIF_spiking
from fsGIF.nblogger import logger

HOME = "/home/alex/Recherche/macke_lab"
DATADIR = "/home/alex/Recherche/data/mackelab/sim/fsGIF/"
DUMPDIR = os.path.join(DATADIR, "run_dump")

recordstore = smttk.RecordStore(os.path.join(HOME, "containers/fsGIF/run/.smt/records"))
#records = smttk.get_records(recordstore, 'fsGIF')
"""

exec(init_code)
print(init_code)
