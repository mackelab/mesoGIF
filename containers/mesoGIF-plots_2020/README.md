# Purpose

This container archives the code used to produce the figures in the paper *Inference of a mesoscopic population model for population spike trains*, by René et al. Note that the *data* in those figures used slightly older versions of our code librairies, which are archived in the related container [mesoGIF-2020](../mesoGIF_2020)".


# Installation

Simply execute `install.sh` :

    chmod u+x install.sh
    ./install.sh

If this package contains private repositories accessed through `ssh`, the easiest is to execute

    ssh-add

before starting the installation. (See <https://www.ssh.com/ssh/agent>.)

If installing on a remote machine, see the additional instructions [below](#extra-indications-for-installing-on-a-remote-server).


# Running the code

This container archives code dependencies for a Jupyter notebook; you should not need to run code from the command line. If you do, you are likely looking for the "mesoGIF" container.

# Running in a Jupyter Notebook

Since this package installs in its own virtual environment, you need to tell Jupyter how to find it. This is done by [registering it as a kernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html). In brief,

    source venv/bin/activate
    python -m ipykernel install --user --name mesoGIF-figs --display-name "mesoGIF (figs)"

Then within a Jupyter notebook select "mesoGIF-figs" as the kernel.


## Extra indications for installing on a remote server

If there are no private repositories, the instructions above should work as-is.

If there are private repositories, make sure you have SSH agent forwarding configured. (<https://www.ssh.com/ssh/agent#sec-SSH-Agent-Forwarding>)
You can then run `ssh-add` on the local machine, before `ssh`-ing to the server and running the install script.

Note that agent forwarding may not work through additional clients such as [tmux](https://tmux.github.io/).
