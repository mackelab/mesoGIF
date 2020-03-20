import matplotlib as mpl
import matplotlib.pyplot
import numpy as np
import mackelab as ml  # TODO: Change to mackelab_toolbox
from . import figureslib as lib

# =========================
# Generic config for papers
# =========================

# Equivalent to
#   %matplotlib inline
#   %config InlineBackend.print_figure_kwargs = {'bbox_inches':None}
# The second line ensures figures are displayed as they will be saved
from IPython import get_ipython
ipython = get_ipython()
#ipython.magic('matplotlib tkagg')
ipython.magic("config InlineBackend.print_figure_kwargs = {'bbox_inches':None}")

# References on getting consistent size with matplotlib:
#   https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-matplotlib
#   https://github.com/matplotlib/matplotlib/issues/4853

from matplotlib.backends.backend_pgf import FigureCanvasPgf
mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
styles = ["mldefault", "mlpublish"]
mpl.style.use(styles)

# ============================
# Specific but reusable config
# ============================

ml.rcParams['plot.subrefformat'] = '{!u}'
ml.rcParams['plot.subrefinside'] = False
ml.rcParams['plot.subrefx'] = 0.
ml.rcParams['plot.subrefy'] = 1.

mpl.rcParams['errorbar.capsize'] = 1

_colours = np.array(['#113f65', '#a43d12', '#42860e', '#c01675', '#8d5ded', '#d7a429'])
_colours_light = np.array(['#658fb2', '#d2896b', '#8cc262', '#d36ca6', '#d1c0f2', '#e3c886'])
colours_very_light = np.array(['#80b4e0', '#e79775', '#acee78', '#f5afd6',])
# Invert green and pink order
#_colours[2:4] = _colours[2:4][::-1]
#_colours_light[2:4] = _colours_light[2:4][::-1]
#colours_very_light = colours_very_light[2:4][::-1]
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=_colours)

# TODO: Add to mackelab.stylelib.colorschemes
black = '#333333'
icyblack = '#4F6066'
blue  = '#113F65'
red   = "#B76948"
transparent = '#FFFFFF00'

# White background settings
def light_bg(black='#333333', white='white', colours=None, colours_light=None, target=None):
    """
    target: namespace dictionary
        If specified, colour definitions will be injected into this namespace.
        E.g. if you are calling from a notebook and you want the colours to be
        defined in that notebook's global namespace, specify `target=globals()`.
        Colours are always injected into the `figureslib` namespace as well.
    """
    if colours is None: colours = _colours
    if colours_light is None: colours_light = _colours_light
    if target is not None:
        try:
            target['black'] = black
            target['white'] = white
            target['colours'] = colours
            target['colours_light'] = colours_light
        except TypeError:
            raise ValueError("`target` argument should be dict-like, like `globals()`.")
    globals()['black'] = black
    globals()['white'] = white
    globals()['colours'] = colours
    globals()['colours_light'] = colours_light
    lib.black = black
    lib.white = white
    lib.colours = colours
    lib.colours_light = colours_light
    mpl.rcParams['axes.edgecolor'] = black
    mpl.rcParams['axes.labelcolor'] = black
    mpl.rcParams['xtick.color'] = black
    mpl.rcParams['ytick.color'] = black
    mpl.rcParams['text.color'] = black
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=colours)

# Dark background settings
def dark_bg(black = '#AAAAAA', white = 'black', colours=None, colours_light=None, target=None):
    if colours is None: colours = _colours_light
    if colours_light is None: colours_light = _colours
    light_bg(black=black, white=white, colours=colours, colours_light=colours_light, target=target)

light_bg()
