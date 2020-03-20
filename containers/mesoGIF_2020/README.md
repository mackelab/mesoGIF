# Purpose

This container archives the code used to produce the results in the paper *Inference of a mesoscopic population model for population spike trains*, by René et al. [arXiv](https://arxiv.org/abs/1910.01618) The instructions below and accompanying [requirements.txt](./requirements.txt) will create a virtual environment with package versions as they were used to produce the published results. In particular, these versions *don't include subsequent usability fixes* .

# Installation

Simply execute `install.sh` :

    chmod u+x install.sh
    ./install.sh

If this package contains private repositories accessed through `ssh`, the easiest is to execute

    ssh-add

before starting the installation. (See <https://www.ssh.com/ssh/agent>.)

If installing on a remote machine, see the additional instructions [below](#extra-indications-for-installing-on-a-remote-server).


# Running the code

To run the code, change to the `run` directory. If it isn't already, activate the virtual environment

    source venv/bin/activate

Running a script is then done by executing a line of the form

    python ../code/[script file] params/[param file] ""

for example,

    python ../code/gradient_descent.py params/gradient_descent.py ""

Note the extra quotes at the end – this is an artifact of the way we've accomodated parallelized calls.

Executing through `Sumatra` is very similar

    smt run -m ../code/[script file] params/[param file] ""

To execute multiple runs at once, optionally providing the number of cores to use, use the provided `smttk` wrapper for `Sumatra`:

    smttk run -n[cores] -m ../code/[script file] params: params/[param file 1] params/[param file 2] ...

Note that when using `smttk` we don't need the trailing quotes.
If only one parameter file is provided, `params:` is not necessary:

    smttk run -n[cores] -m ../code/[script file] params/[param file]

Any parameter file can use the specialized [expansion syntax](#parameter-expansion-syntax) to define a range of parameters to iterate over.

# Running in a Jupyter Notebook

Since this package installs in its own virtual environment, you need to tell Jupyter how to find it. This is done by [registering it as a kernel](https://ipython.readthedocs.io/en/stable/install/kernel_install.html). In brief,

    source venv/bin/activate
    python -m ipykernel install --user --name mesoGIF --display-name "Python (mesoGIF)"

Then within a Jupyter notebook select "mesoGIF" as the kernel.

# Parameter expansion syntax

Parameter files follow the same format as NeuroEnsemble's [Parameters](https://parameters.readthedocs.io/en/latest/) package. One addition is made to the format to allow easily specifying ranges of parameters. For example, if a script requires a single parameter `mu` and$ that we want to run it with values 1, 5 and 20, we would write the following in the parameters file:

    {
      mu: *[1, 5, 20]
    }

This is only supported when calling with `smttk`.

## Extra indicationsn for installing on a remote server

If there are no private repositories, the instructions above should work as-is.

If there are private repositories, make sure you have SSH agent forwarding configured. (<https://www.ssh.com/ssh/agent#sec-SSH-Agent-Forwarding>)
You can then run `ssh-add` on the local machine, before `ssh`-ing to the server and running the install script.

Note that agent forwarding may not work through additional clients such as [tmux](https://tmux.github.io/).
