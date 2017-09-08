import os
import parameters

import core

try:
    import click
except ImportError:
    pass
else:

    @click.group()
    def cli():
        pass

    @click.command()
    @click.argument('src', nargs=1)
    @click.argument('dst', nargs=1)
    @click.option('--datadir', default="")
    @click.option('--suffix', default="")
    @click.option('--ext', default=".sir")
    @click.option('--link/--no-link', default=True)
    def rename(src, dst, ext, datadir, suffix, link):
        """
        Rename a result file based on the old and new parameter files.
        TODO: Allow a list of suffixes
        TODO: Update Sumatra records database
        TODO: Match any extension; if multiple are present, ask something.
        Parameters
        ----------
        src: str
            Original parameter file
        dst: str
            New parameter file
        link: bool (default: True)
            If True, add a symbolic link from the old file name to the new.
            This avoids invalidating everything linking to the old filename.
        """
        if ext != "" and ext[0] != ".":
            ext = "." + ext
        old_params = parameters.ParameterSet(src)
        new_params = parameters.ParameterSet(dst)
        old_filename = core.RunMgr._get_filename(old_params, suffix) + ext
        new_filename = core.RunMgr._get_filename(new_params, suffix) + ext
        old_filename = os.path.join(datadir, old_filename)
        new_filename = os.path.join(datadir, new_filename)

        if not os.path.exists(old_filename):
            raise FileNotFoundError("The file '{}' is not in the current directory."
                                    .format(old_filename))
        if os.path.exists(new_filename):
            print("The target filename '{}' already exists. Skipping the"
                  "renaming of '{}'.".format(new_filename, old_filename))
        else:
            os.rename(old_filename, new_filename)
            print("Renamed {} to {}.".format(old_filename, new_filename))
        if link:
            # Allowing the link to be created even if rename failed allows to
            # rerun to add missing links
            # FIXME: Currently when rerunning the script dies on missing file
            #   before getting here
            os.symlink(os.path.basename(new_filename), old_filename)
            print("Added symbolic link from old file to new one")

    cli.add_command(rename)

    if __name__ == "__main__":
        cli()
