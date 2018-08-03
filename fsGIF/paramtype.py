"""
Defines the parameter format we use.

Actually just defines a 'format' which does no formatting at all, and lets
scripts deal with parsing parameters. This avoids Sumatra trying to guess the
format, which can cause errors.
"""

import abc
import os.path
from sumatra.core import component_type, component
from sumatra.parameters import ParameterSet

# @component_type
# class FreeformParams(metaclass=abc.ABCMeta):
#     required_attributes = ('name',)
#         # name: extension to use (including period)
#

@component
class FsGIFParams(ParameterSet):
    name = '.params'

    def __init__(self, initialiser):
        if os.path.exists(initialiser):
            with open(initialiser) as f:
                self.paramstr = f.read()
        else:
            self.paramstr = initialiser

    def save(self, filename, add_extension=False):
       if add_extension:
           filename += self.name
       with open(filename, "w") as f:
           f.write(self.paramstr)
       return filename

    def update(self, E, **F):
        raise NotImplementedError
