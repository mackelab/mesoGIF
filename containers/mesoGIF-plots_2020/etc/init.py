fig_init_code = """
from fsGIF.init import *
from odictliteral import odict
from operator import sub
from warnings import warn
from scipy import stats
from tqdm import tqdm as tqdm
from attrdict import AttrDict

import logging
logging.getLogger().setLevel('ERROR')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
pd.set_option('display.max_rows', 25)
"""

exec(fig_init_code)
print(fig_init_code)

del fig_init_code

def inject_vars(input_module, target_namespace):
    for name, val in input_module.__dict__.items():
        if name[:2] != "__":
            target_namespace[name] = val
            
print("Defined:")
print("    inject_vars()")