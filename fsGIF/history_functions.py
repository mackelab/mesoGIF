from collections import Iterable
import numpy as np
from sinn.history_functions import PiecewiseLinear

class RampSpikes(PiecewiseLinear):
    """
    Produce a sequence of linear ramp spikes.

    Parameters
    ----------
    centers: list of floats
    halfwidth: float | list of floats
    heigths: list of floats
        Must have same length as `centers`.
    """
    def init_params(self, centers, halfwidth, heights):
        L = len(centers)
        if not isinstance(halfwidth, Iterable):
            halfwidth = [halfwidth] * L
        heights = np.array(heights)
        if heights.ndim == 0:
            heights = np.broadcast_to(heights, (L,1))
        elif heights.ndim == 1:
            heights = np.broadcast_to(heights, (L, len(heights)))
        assert(len(halfwidth) == len(heights) == L)

        values = [i*h for t,h in zip(centers, heights)
                      for i in (0, 1, 0)]
        stops = [t+Δ for t, hw in zip(centers, halfwidth)
                     for Δ in (-hw, 0, hw)]
        return super().init_params(stops, values)
