# A *mesoGIF* implementation

## Context

This repository archives the code used to obtain the results in [René et al., 2020](https://arxiv.org/abs/1910.01618). To reproduce the software environment used for that publication (warts and all), see the folders under [containers](./containers).

### Deprecation notice

This code was built on a previous experimental version of our [*sinn* inference library](https://github.com/mackelab/sinn). *Sinn* has since seen many, backwards-incompatible usability enhancements, and we highly recommend using the newest version. Unfortunately this means that our implementation of the *mesoGIF* model is currently locked to the old *sinn* version until we find the time to update it.

You can find a partially updated *mesoGIF* implementation in *sinn*'s [examples directory](https://github.com/mackelab/sinn/tree/master/examples).

## Installation

We recommend that the following commands be run in their own virtual environment, to avoid disturbing other packages.

    python3 -m venv fsgif     
    source fsgif/bin/activate

First upgrade `pip` and `setuptools` (depending on your system, some packages may fail to install if you skip this step)

    pip install --upgrade pip
    pip install --upgrade setuptools

Then install the package by navigating to this directory and running

    pip install .

(Add the `-e` option to install in develop mode, which allows modifying the code without requiring a reinstall every time.)

## Running

### Running in other scripts

Code is installed under the name `fsGIF` and can be imported into Python scripts as

    import fsGIF

## Varia

### Why is the package called “fsGIF” ?

For historical reasons (“fs” stood for “finite size”). Please use “*mesoGIF*” in your own code and publication, which we find is a more accurate description of the model.
