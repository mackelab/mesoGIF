#!/usr/bin/env python
# coding: utf-8

# # Description
# 
# `Sumatra` works best if each run is stored in a separate folder named by the run label. This is great for distinguishing runs, but makes it hard to recover data since we need to first find the date and time at which it was produced.
# 
# A solution to this problem is to have a have separate directories where runs are organized to allow retrieval (e.g. by task), which simply contain links to all the data in `Sumatra`'s data store. By using links, we never touch the actual files, which helps ensure data integrity: we can change the organization later without worrying of losing data or corrupting the `Sumatra` database. In effect, this constructs a file-system level interface to the output files in the `Sumatra`.
# 
# This notebook (re)builds such a file-system interface.

# # Initialization

# In[1]:


import copy
import os.path
import re
from tqdm import tqdm
from parameters import ParameterSet
import mackelab as ml
import mackelab.smttk as smttk
from mackelab.smttk import MoveList, create_links


# In[2]:


# Project-specific import
from fsGIF.gradient_descent import get_sgd_pathname, get_param_hierarchy
HOME = "/home/alex/Recherche/macke_lab"
#DATADIR = "/home/alex/Recherche/data/mackelab/sim/fsGIF/"
DATADIR = "/home/alex/Recherche/macke_lab/run/fsGIF/data/"
#DATADIR = "/home/alex/Recherche/macke_lab/container/fsGIF/run/data/" ?
    # Works best when DATADIR shares as much of its path as possible
    # with records' `outputpath`.
DUMPDIR = os.path.join(DATADIR, "run_dump")


#     # For debugging
#     DATADIR = "/home/alex/tmp/data-test/"
#     DUMPDIR = os.path.join(DATADIR, "run_dump")

# In[3]:


recordstore = smttk.RecordStore(os.path.join(HOME, "containers/fsGIF/run/.smt/records"))


# In[4]:


records = smttk.get_records(recordstore, 'fsGIF')


# Files can be stored in arbitrary folder hierarchies, but we currently assume the following:
# 
# - That the hierarchy is the same for the data dump and access directly, except the 'label' directory which is only present in the data dump and serves to differentiate runs.
# - That each level of the hierarchy is formed as `[id]_[suffix]`, where `id` is a unique ID string computed from the run parameters. If only `id` or `suffix` is present, we assume that the '`_`' is omitted.
# - That the `id` is a 40-character hex representation of a hash.
# 
# The last two assumptions can be changed by adjusting the variable `id_pattern` as needed.
# 
# The link paths are formed as `[link id]_[source suffix]`. This means that simply providing a `link id` is sufficient, but also that any `link suffix` is ignored.

# In[5]:


lbl_pattern = '[0-9]{8}-[0-9]{6}(_[0-9]+)?'
id_pattern = '\A[0-9a-f]{40}'  # Regex format for automatically computed run IDs


# In[6]:


real_dumpdir = os.path.realpath(DUMPDIR)  # Remove symbolic links from data dump directory


# # File name rules

# We define a different *rule* for each script, specifying how it stores its output. It might just hash the run parameters, but e.g. the gradient descent script organizes them into a hierarchy (`[model params]/[SGD params]/[initialization]`).

# In[7]:


def update_params(params):
    """
    Hacky function to still find records using old parameter file format.
    This is specific to our data.
    """
    if ('theano' in params # filters for spike & activity
        and 'input' in params
        and 'type' not in params.input):
        # Only input entries for spike & activity generation were changed
        # Derefence all entries which may link to 'input'
        params.seed = params.seed
        params.t0 = params.t0
        params.tn = params.tn
        params.dt = params.dt
        # Update params.input
        params.input = ParameterSet({
            'type': "Series",
            'dir' : "inputs",
            'params': params.input,
            'name': ""})
    else:
        for key, val in params.items():
            if isinstance(val, ParameterSet):
                update_params(val)
                
def update_params(params):
    return


# In[8]:


# TODO: Move these rules into the modules themselves
def get_link_path(rec):
    if 'generate_input' in rec.main_file:
        file_depth = 1
        link_filename = ml.parameters.get_filename(rec.parameters)
        link_dirpath = 'inputs'
    elif 'generate_spikes' in rec.main_file:
        file_depth = 1
        params = copy.deepcopy(rec.parameters)
        update_params(params)
        link_filename = ml.parameters.get_filename(params)
        link_dirpath = 'spikes'
    elif 'generate_activity' in rec.main_file:
        params = copy.deepcopy(rec.parameters)
        update_params(params)
        file_depth = 1
        link_filename = ml.parameters.get_filename(params)
        link_dirpath = 'activity'
    elif 'gradient_descent' in rec.main_file:
        params = copy.deepcopy(rec.parameters)
        update_params(params)
        file_depth = 3
        prefix, link_dirpath, link_filename = get_sgd_pathname(params).split('/', 2)
        #assert(os.path.realpath(prefix) == os.path.realpath(DATADIR))  # Removed for debugging
        assert(link_dirpath == 'fits')
    elif 'mcmc' in rec.main_file:
        params = copy.deepcopy(rec.parameters)
        update_params(params)
        file_depth = 1
        link_filename = ml.parameters.get_filename(params)
        link_dirpath = 'mcmc_nosync'
    elif 'likelihood' in rec.main_file:
        params = copy.deepcopy(rec.parameters)
        update_params(params)
        file_depth = 1
        link_filename = ml.parameters.get_filename(params)
        link_dirpath = 'likelihood'
    else:
        raise ValueError("There is no rule for python module `{}`.".format(rec.main_file))
    return file_depth, link_dirpath, link_filename


# In[9]:


def subsource(target_part, source_part):
    """Substitute a link id into a data path part"""
    source_match = re.match(id_pattern, source_part)
    target_match = re.match(id_pattern, target_part)
    if (source_match is None) is not (target_match is None):
        # Only one of the two was matched
        raise RuntimeError("Data and link filenames don't have the same format\n"
                           "Data: {}\nLink: {}".format(os.path.join(*source_filename_parts), target_filename))
    # The source may have suffixes not included in the target. Suffixes can indicate run exit state
    # or serve to distinguish multiple files produced by the same run
    sspan = source_match.span()
    target_id = target_part[slice(*target_match.span())]
    return source_part[:sspan[0]] + target_id + source_part[sspan[1]:]


# # Extract links from Sumatra records

# Loop through the records and identify all the links we need to create.<br>
# If the same link is added multiple times, only the one pointing to the latest data is kept

# In[10]:


# Structure to store all links to data
symlinks = MoveList()
dirpaths = set()

# Loop over records.
# Ignore records before 2018 – They were stored differently, and are mostly junk anyway
for rec in tqdm(records.filter.after(2018)):
    # Normalize all data paths for this record, making them relative to the dump directory and removing any links
    realdatapaths = [os.path.realpath(path) for path in rec.outputpath]  # Remove symbolic links
    # HACK
    if (os.path.commonpath([real_dumpdir] + realdatapaths) != real_dumpdir):
        # Some really old output files were put directly into the organized directory
        # We don't do this any more, so no point in dealing with it
        continue
    elif not isinstance(rec.parameters, ParameterSet):
        # Some older records have buggy parameters
        # This is fixed, so no point in dealing with them
        continue
    assert(os.path.commonpath([real_dumpdir] + realdatapaths) == real_dumpdir)
        # Validate assumption that all data is within the dump directory
    
    # Get the link path
    file_depth, target_dirpath, target_filename = get_link_path(rec)
    
    # For every data path:
    #   - Construct an updated link path, where the 'new' ids are merged with the 'source' suffixes
    #   - Add the updated link path to the list of links to add

    for real_source_path, output_source_path in zip(realdatapaths, rec.outputpath):

        # Get a source path without the label, so we can compare with the link path
        source_path = os.path.relpath(real_source_path, start=real_dumpdir)
        label, source_hierarchy_path = source_path.split('/', 1)
            # The label is the first directory after the dump dir
            # source_hierarchy_path: starts from dumpdir/label, so location-agnostic
        assert(re.fullmatch(lbl_pattern, label) is not None)  # Verify that we really extracted a label

        # Split path hierarchy into a list
        dirpath, *source_filename_parts = source_hierarchy_path.rsplit('/', file_depth)
        target_filename_parts = target_filename.split('/')

        # Check that data and link hierarchies are the same
        if dirpath != target_dirpath:
            raise RuntimeError("Data and link files are in different directories.\n"
                               "Data: {}\nLink: {}".format(dirpath, target_dirpath))
        if len(target_filename_parts) != len(source_filename_parts):
            raise ValueError("`link` and `data` paths must have the same depth.")

        # Create the new link path by substituting the link ids into the source path parts
        new_link_parts = [subsource(target_part, source_part)
                            for source_part, target_part in zip(source_filename_parts, target_filename_parts)]
        real_link_path = os.path.join(DATADIR, dirpath, *new_link_parts)
        #if os.path.realpath(DUMPDIR) in real_link_path:
        #    # If there was already a link pointing to the data, `os.path.realpath` will dereference it
        #    assert(real_source_path == real_link_path)
        #    # Since the link is already there, no need to do anything
        #    continue

        if 'run_dump' in real_link_path:
            pdb.set_trace()
            pass
        # Add the link to the list of paths to create
        symlinks.add(output_source_path, real_link_path, rec.timestamp)
        dirpaths.add(dirpath)


# Test that most recent files were found
# 
#     from datetime import datetime
#     for sl in symlinks:
#         if sl['timestamp'] > datetime(2019,1,1):
#             print(sl)

# In[11]:


[sl for sl in symlinks if '82574086fa7d851' in sl['old path']]


# # fsGIF hack
# 
# At the very begining output from MCMC runs was huge and so was saved in a “nosync” folder to prevent clogging up other computers. This was fixed, but I only changed the folder until much later, so now most of the MCMC data is in unsynced folders. As a workaround, when I need MCMC data, I copy it to a new folder (`mcmc_nosync` -> `mcmc`) and add the new paths here.

#     symlinks.add( [output_source_path], [real_link_path], [timestamp] )

# In[12]:


from datetime import datetime


# In[13]:


dirpaths.add('mcmc')
symlinks.add("/home/alex/Recherche/macke_lab/run/fsGIF/data/run_dump/20180831-065439_1/mcmc/53c81f2276f69ca2cbc0325710dc35e27d22a56a.dill",
             "/home/alex/Recherche/macke_lab/run/fsGIF/data/mcmc/6b19ced0fd59a8f39217d094ce2d5664bf7df93a.dill",
             datetime(2018, 11, 3))
symlinks.add("/home/alex/Recherche/macke_lab/run/fsGIF/data/run_dump/20180831-065439_2/mcmc/da73afce80573436da0fe280024410f41289465d.dill",
             "/home/alex/Recherche/macke_lab/run/fsGIF/data/mcmc/e8dda1ae1532286eac26fa59101dfe1559008f58.dill",
             datetime(2018, 11, 3))


# This spikes run was run with `tn=10`, but the figures notebook expects `tn=10.`. The link maps the results to the expected location.

# In[14]:


dirpaths.add('spikes')
for s in ['', '_activity', '_expected_activity']:
    symlinks.add("/home/alex/Recherche/macke_lab/run/fsGIF/data/run_dump/20190627-164646/spikes/eca7877f3818a8eef144998a2409976a1d045286{}.npr".format(s),
                 "/home/alex/Recherche/macke_lab/run/fsGIF/data/spikes/8ae16a4263f5fecc26103ae8c78d4a39e36785f1{}.npr".format(s),
                 datetime(2019, 6, 27))


# # Sanity check

# We used to have an issue where links would also be placed in 'run_dump', which was clearly incorrect.
# If that happens agains, the following will catch it and stop execution before creating the links.

# In[15]:


D = ({m['new path']:m['old path'] for m in symlinks if 'run_dump' in m['new path']})
assert(len(D) == 0)


# If there's junk in the interface directories (e.g. because we changed naming conventions),
# we can clean it up by clearing out old links (anything which is not a link is left in place)

#     for dirpath in dirpaths:
#         for fname in os.listdir(os.path.join(DATADIR, dirpath)):
#             path = os.path.join(DATADIR, dirpath, fname)
#             if os.path.islink(path):
#                 os.remove(path)

# # Create the links

# Finally, create all the links.<br>
# `create_links` makes relative links, so if the link and target share enough of their path, they should be portable between computers.

# In[16]:


create_links(symlinks)


# In[ ]:




