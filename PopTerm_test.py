
# coding: utf-8

# In[1]:


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

import sinn
from sinn.histories import Spiketrain, Series
import sinn.analyze as anlz
import fsGIF.fsgif_model as fsgif
import fsGIF.fsgif_model_old as fsgifold

from parameters import ParameterSet
import mackelab as ml
import mackelab.parameters

import theano_shim as shim

# In[2]:


T = 30
datalen = 0.03
pop_sizes = (3, 10)

data = np.random.binomial(1, p=0.3, size=(T, sum(pop_sizes)))

shist = Spiketrain(name='S', time_array=np.linspace(0, datalen, T), pop_sizes=pop_sizes)
shistold = Spiketrain(shist)

shist.set(data)
shistold.set(data)
shist.lock()
shistold.lock()

Ihist = Series(name='I', time_array=shist.get_time_array(), shape=(len(pop_sizes),),
               iterative=False)
coeffs = np.random.uniform(-1.5, 1.5, size=2)
def Ifn(t):
    if not shim.isscalar(t):
        assert(t.ndim == 1)
        t = t.reshape(t.shape + (1,))
    return coeffs*np.sin(t)
Ihist.set_update_function(Ifn)
Ihist.set()
Ihist.lock()

pset = ml.parameters.params_to_arrays(ParameterSet("../../run/fsGIF/params/model2pop.params"))
pset.N = np.array(pop_sizes)

pset.Γ = fsgif.GIF_spiking.make_connectivity(pset.N, pset.p)


# In[3]:


# In[4]:


params = fsgif.GIF_spiking.Parameters(**pset)
paramsold = fsgifold.GIF_spiking.Parameters(**pset)


# In[5]:


model = fsgif.GIF_spiking(params, shist, Ihist)
modelold = fsgifold.GIF_spiking(params, shist, Ihist)


# In[6]:


model.λ.set()
modelold.λ.set()


# In[19]:


assert((model.λ._data.get_value() == modelold.λ._data.get_value()).all())

