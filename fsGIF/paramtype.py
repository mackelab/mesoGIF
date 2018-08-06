"""
Defines the parameter format we use.
"""

from sumatra.core import component
import sumatra as smt

@component
class FsGIFParams(smt.parameters.NTParameterSet):
    """
    WIP: Still need to figure out how to convince Sumatra to load this
    parameter type. At present it's saved properly (can be checked by opening
    `.smt/records` in an SQL browser), but in the web interface the parameters
    section is blank, and when loaded programmatically the parameters are just
    a single string.
    """
    name = '.params'  # Rename to something more concise

# ==================================================
# Incomplete code for a free-form parameter, as would be required by
# generate-mle-activity.py

import os.path
import re
import parameters
from mackelab.utils import strip_comments

@component
class FreeformParams(parameters.ParameterSet, smt.parameters.ParameterSet):
    """
    Goal is to define a 'format' which does no formatting at all, and lets
    scripts deal with parsing parameters. This avoids Sumatra trying to guess
    the format, which can cause errors.

    WARNING: This class is incomplete. It doesn't deal well with nested
    parameter sets and overall may not be that useful. Although
    sumatra.parameters says that only `save` and `update` need to be supported,
    doing so means that parameters aren't saved. This seems to require `as_dict`, which then forces use to perform at least partial parsing of the
    parameters, which starts to defeat the purpose of a class that bypasses all
    parsing.
    """
    name = '.ffparams'

    do_not_quote = re.compile("url\(.+\)|\d+|\d*.\d+")
        # Regex for strings that should not be quoted
        # Any string whose beginning matches is not quoted

    def __init__(self, initialiser):
        if os.path.exists(initialiser):
            with open(initialiser) as f:
                self.paramstr = f.read()
        else:
            self.paramstr = initialiser

        # Creates a dictionary where illegal entries are quoted
        super().__init__(self._make_dict(self.paramstr))

    def __str__(self):
        return self.paramstr

    def pretty(self, expand_urls=False):
       paramstr = self.paramstr
       if expand_urls:
           paramstr = ml.parameters.expand_urls(paramstr)
       return paramstr

    def save(self, filename, add_extension=False):
        # We don't use _dict here because it added some quotes.

        if add_extension:
            filename += self.name
        with open(filename, "w") as f:
            f.write(self.paramstr)
        return filename

    def __eq__(self, other):
        raise NotImplementedError
    def __ne__(self, other):
        raise NotImplementedError
    def update(self, E, **F):
        raise NotImplementedError
    def pop(self, key, d=None):
        raise NotImplementedError


    def _make_dict(self, paramstr):
        regex = re.compile('".*?"|\'.*?\'|:|\{|\}|,|\s+|[^"\'\s,\{\}]+')
        whitespace = re.compile('\s+')

        paramstr = strip_comments(paramstr)

        keys = []
        values = []
        # state: unopened | key | value | closed
        state = 'unopened'
        match = regex.search(paramstr)
        #while match is not None or state != 'closed':
        for i in range(len(regex.findall(paramstr))):
            if match is None or state == 'closed':
                break

            s = paramstr[match.start():match.end()]

            if state == 'unopened':
                if s == '{':
                    state = 'key'
                    keys.append(slice(match.end(), match.end()))
                else:
                    continue

            else:
                if s == '}':
                    #assert(state == 'value')
                    state = 'closed'
                elif s == ':':
                    assert(state == 'key')
                    keys[-1] = slice(keys[-1].start, match.start())
                    values.append(slice(match.end(), match.end()))
                    state = 'value'
                elif s == ',':
                    assert(state == 'value')
                    values[-1] = slice(values[-1].start, match.start())
                    keys.append(slice(match.end(), match.end()))
                    state = 'key'
                else:
                    pass
                    #if state == 'key':
                    #    keys[-1] = slice(keys[-1].start, match.end())

            match = regex.search(paramstr, match.end())

        if len(keys) != len(values):
            assert(len(keys) == len(values) + 1)
            assert(keys[-1].start == keys[-1].stop)
            del keys[-1]

        keys = [ensure_quotes(paramstr[slc].strip(), self.do_not_quote)
                for slc in keys]
        values = [ensure_quotes(paramstr[slc].strip(), self.do_not_quote)
                  for slc in values]

        return {key: value for key, value in zip(keys, values)}

def ensure_quotes(s, do_not_quote=None):
    """
    Will not work if `s` contains both double and single quotes.
    """
    if s[0] in ['"', "'"] and s[-1] == s[0]:
        # Already in quotes
        return s

    if do_not_quote is not None:
        if do_not_quote.match(s) is not None:
            # Don't quote this string
            return s

    # We need to add quotes. Decide between " or '.
    if '"' not in s:
        return '"' + s + '"'
    elif "'" not in s:
        return "'" + s + "'"
    else:
        raise ValueError("`s` must not contain both single and double "
                         "quotes.")
