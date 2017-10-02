from sumatra.launch import *
import multiprocessing as mp
from collections import namedtuple

@component
class BatchLaunchMode(LaunchMode):
    """
    Enable running batch computations, using the multiprocessing module.
    Effectively this uses multiprocessing to spawn a number of processes,
    and executes a SerialLaunchMode in each.
    """
    name = "batch"

    _ExecArgs = namedtuple("_ExecArgs",
                          ['working_directory', 'options', 'executable',
                           'main_file', 'arguments', 'append_label'])

    def __init__(self, n=1, options=None, working_directory=None):
        """
        `n` - the number of processes to execute simultaneously.
        `options` -
        `working_directory` -
        """
        super().__init__(working_directory, options)
        self.n = n

    def __str__(self):
        return "batch"

    def run(self, executable, main_file, arguments_list, append_label,
            cores):
        """
        Run multiple computations, each in its own shell, with the given
        executable, script and arguments; one computation is launched for
        each item in `arguments_list`. In contrast to LaunchMode,
        `append_label` is required (but may be "") to avoid mixing the
        output from different runs; a different number is appended to the
        label for each run. The result is appended to the command line calls.
        As many simultaneous processes will be launched as specified by `cores`.
        (If `cores` is None, then as many processes are launched as there are
        CPUs on the system.)
        Return True if the computation finishes successfully, False otherwise.
        """
        nruns = len(arguments_list)
        exec_args_list = [ _ExecArgs(working_directory,
                                     options,
                                     executable,
                                     main_file,
                                     arguments,
                                     append_label + '_' + str(i))
                           for arguments, i in zip(arguments_list,
                                                   range(nruns)) ]
        with mp.Pool(cores) as pool:
            pool.map(self._exec_subprocess, exec_args_list)

    @staticmethod
    def _exec_subprocess(args):
        launcher = LaunchMode(a.working_directory, args.options)
        return launcher.run(args.executable, args.main_file,
                            args.arguments, args.append_label)
