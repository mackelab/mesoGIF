# Set of functions accompanying the Figures.ipynb Jupyter notebook
# for our paper on inferring a complex mesoscale model (Schwalger et al., 2017)

# File created 15 Oct. 2018
# Author: Alexandre René

from .init import *

# =========================================
# Module-wide variable definitions

transparent = '#FFFFFF00'

default_dt = 0.01  # Default subsampling for plots at 10ms bins

varstrings = {'w':   ['$w_{EE}$', '$w_{EI}$', '$w_{IE}$', '$w_{II}$'],
              'Δu':  ['$\\Delta u_E$', '$\\Delta u_I$'],
              'c':   ['$c_{E}$', '$c_{I}$'],
              'J_θ': ['$J_{θ,E}$'],
              'τ_θ': ['$τ_{θ,E}$'],
              'τ_m': ['$τ_{m,E}$', '$τ_{m,I}$'],
              'τ_s': ['$τ_{s,E}$', '$τ_{s,I}$']
             }
    # Names corresponding to indices in fitcoll.result

# =========================================
# Generic data structures

class AxAttrs(AttrDict):
    """Only use strings as keys."""
    def __init__(self, *args, **kwargs):
        kwargs = {str(k): v for k,v in kwargs.items()}
        if len(args) == 1 and isinstance(args[0], dict):
            args = ({str(k): v for k,v in args[0].items()},)
        super().__init__(*args, **kwargs)
    def __getitem__(self, key):
        if isinstance(key, (list, tuple)):
            return type(key)(self[k] for k in key)
        else:
            return super().__getitem__(str(key))
    def __setitem__(self, key, value):
        super().__setitem__(str(key), value)

# def remove_duplicates(records):
#     """TODO: move to RecordSet"""
#     for model, recset in records.items():
#         digests = []
#         recs = deque()
#         for rec in recset:
#             d = ml.parameters.digest(rec.parameters)
#             if d not in digests:
#                 digests.append(d)
#                 recs.append(rec)
#         records[model] = RecordSet(recs)
#     return records

# Functions for keeping track of which records / data sets were used to produce
# figures. These are still WIP, but the idea is to provide an interface through
# which all load calls are routed. This interface keeps track of these calls
# in a class variable, such that at the end we can output a list of all used
# records / data. This list could be used to bundle all relevant data into an
# archive.
# Because keeping everyting in memory can be a bad idea, the interface should
# also allow to "unload" data, while keeping track of where it was so it can
# be loaded again.

# This interface functions are generic, and should eventually be put into the
# `mackelab` packages. Probably merged with `smttk.RecordList` and friends.

class RecordSet:
    """
    Ensemble of records for a figure.
    TODO: Allow adding records.
    """
    project = 'fsGIF'
    recordstore = recordstore

    def __init__(self, *records, min_data=1, nodill=True, nodups=True):
        """
        min_data: int
            Records with less than this number of outputpaths are ignored.
            Setting this to something >0 usually removes runs that failed
            before producing any output.
        nodill: bool
            If any of output file has a 'dill' extension, discard the record.
            Reasoning: 'dill' is the fallback format for the save function,
            and thus may indicate a failed or corrupted run.
            FIXME: Should be False by default.
        nodups: bool
            If more than one record had the same parameters, keep only one.
            TODO: make this the latest record.
        """
        if len(records) == 1 and isinstance(records[0], smttk.RecordList):
            records = records[0]
        else:
            recs = set()
            for rec in records:
                if isinstance(rec, Iterable) and not isinstance(rec, (str, bytes)):
                    recs.update(rec)
                else:
                    recs.add(rec)
            #if any(r not in labels):
            #    print(labels which are not in)
            #    print(warning: may take longer)
            records = smttk.get_records(self.recordstore, self.project, recs,
                                        min_data=min_data)
        # Remove records that were lost due to full disk
        if nodill:
            records = records.filter(
                lambda rec: all(p[-5:] != '.dill' for p in rec.outputpath)).list
        setattr(self, 'recordlist', records)
        setattr(self, 'records', {rec.label: Record(self.project, rec)
                                  for rec in records})
        if nodups:
            self.remove_duplicates()

    def __str__(self):
        return str(self.records)

    def __repr__(self):
        return repr(self.records)

    def __iter__(self):
        return iter(self.records.values())

    def __getattr__(self, attr):
        if attr in ('recordlist', 'records'):
            raise AttributeError
        return getattr(self.recordlist, attr)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, key):
        if isinstance(key, Iterable) and not isinstance(key, (str, bytes)):
            if len(key) == 1:
                return self[next(iter(key))]
            else:
                return set(ml.utils.flatten([self[k] for k in key]))
        else:
            try:
                return self.records[key]
            except (KeyError, TypeError):
                lbls = {rec.label for rec in self.recordlist.filter.label(key)}
                return set(ml.utils.flatten(self.records[lbl] for lbl in lbls))

    def remove_duplicates(self):
        digests = []
        lbls_to_remove = deque()
        for rec in self.recordlist:
            d = ml.parameters.digest(rec.parameters)
            if d in digests:
                lbls_to_remove.append(rec.label)
            else:
                digests.append(d)
        self.remove(lbls_to_remove)

    def remove(self, labels):
        recs = deque()
        for rec in self.recordlist:
            if rec.label not in labels:
                recs.append(rec)
        for lbl in labels:
            del self.records[lbl]
        self.recordlist = smttk.RecordList(recs)

class Record:
    """
    Loading records through this class keeps track of them,
    so a list of those used for the paper can be made at the end.
    """
    project = 'fsGIF'
    all_records = {}

    def __new__(cls, project, rec=None):
        if rec is None:
            rec = project
            project = cls.project
        if isinstance(rec, str): label = rec
        else: label = rec.label
        if label not in cls.all_records:
            cls.all_records[label] = super().__new__(cls)
        return cls.all_records[label]

    def __init__(self, project, rec=None):
        # TODO: Avoid duplication with __new__
        if rec is None:
            rec = project
            project = self.project
        if isinstance(rec, str):
            rec = smttk.get_record(recordstore, project, rec)
        if not hasattr(self, 'record'): setattr(self, 'record', rec)
        if not hasattr(self, 'data'): setattr(self, 'data', {})

    def __getattr__(self, attr):
        if attr in ('record', 'data'):
            raise AttributeError
        return getattr(self.record, attr)

    def load(self, path=None, exclude=None):
        """
        path==None : call `load()` on all loaded data
        path=='all': load all of records data
        """
        if isinstance(path, (str, bytes)): path = [path]
        if isinstance(exclude, (str, bytes)): exclude = [exclude]
        elif exclude is None: exclude = ()
        if len(self.outputpath) == 1 and path is ():
            # If only one output file, don't force passing 'all' to load it
            outpath = self.outputpath[0]
            self.data[outpath] = Data(outpath)
        else:
            for outpath in self.outputpath:
                if path is None and outpath in self.data:
                    self.data[outpath].load()
                elif path is None and exclude is None:
                    continue
                elif ( path == 'all'
                       or ( (path is None or any(p in outpath for p in path))
                            and not any(e in outpath for e in exclude) ) ):
                    if outpath in self.data:
                        self.data[outpath].load()
                    else:
                        self.data[outpath] = Data(outpath)

    def unload(self):
        for d in self.data.values():
            d.unload()

class Data:
    """
    Loading data through this class keeps track of them,
    so a list of those used for the paper can be made at the end.
    """
    all_data = {}

    def __new__(cls, datapath):
        if datapath not in cls.all_data:
            cls.all_data[datapath] = super().__new__(cls)
        return cls.all_data[datapath]

    def __init__(self, datapath):
        if not hasattr(self, 'path'): setattr(self, 'path', datapath)
        self.load()

    def __getattr__(self, attr):
        if attr in ('_data', 'path'):
            raise AttributeError
        self.load()
        return getattr(self._data, attr)

    def load(self):
        if not hasattr(self, '_data'):
            self._data = ml.iotools.load(self.path)

    def unload(self):
        """Unload data to save memory. Will be reloaded when needed"""
        if hasattr(self, '_data'): del self._data

class FitCollView(FitCollection):
    """
    Same interface as a FitCollection, but only keeps the last fit collection
    in memory. If a previous collection is queried, it is loaded and replaces
    the one currently in memory.
    Thus it is best to batch operations on a given fit collection.
    """
    current_key = None
    current_fitcoll = None

    def __init__(self, fit_list, **kwargs):
        self.fit_list = fit_list
        self.kwargs = kwargs
        self.key = self._serialize()
        self.__loading = False  # Prevents endless recursion in getattr

    def _serialize(self):
        # Hacky but probably good enough
        return hash(str(self.fit_list) + str(self.kwargs))

    def _ensure_loaded(self):
        if not self.__loading:
            self.__loading = True
            if self.current_key != self.key:
                self.current_fitcoll = FitCollection()
                self.current_fitcoll.load(self.fit_list)
                self.current_key = self.key
            self.__loading = False

    def __getattr__(self, attr):
        self._ensure_loaded()
        return getattr(self.current_fitcoll, attr)

    def __iter__(self):
        self._ensure_loaded()
        return iter(self.current_fitcoll)
    def __next__(self):
        self._ensure_loaded()
        return next(self.current_fitcoll)
    def __getitem__(self, key):
        self._ensure_loaded()
        return self.current_fitcoll[key]

    def __len__(self):
        return len(self.fit_list)

# ================
# Some loading functions

import re
from zipfile import BadZipfile
from mackelab.smttk import RecordList, RecordView

outpatterns = {'activity' : re.compile(".*/[0-9a-f]*\.npr"),
               'spikes'   : re.compile(".*/[0-9a-f]*_activity\.npr")
              }
def load_trace(record, dt=default_dt):
    script = record.main_file.rsplit('/',1)[1]
    if script == 'generate_activity.py':
        outpattern = outpatterns['activity']
    elif script =='generate_spikes.py':
        outpattern = outpatterns['spikes']
    else:
        raise ValueError("Records were generated with the script {}"
                         "It should be either 'generate_activity.py' "
                         "or 'generate_spikes.py'.".format(script))
    paths = [p for p in record.outputpath if outpattern.fullmatch(p) is not None]
    assert len(paths) == 1
    try:
        trace = ml.iotools.load(paths[0])
    except (EOFError, OSError, BadZipfile):
        trace = None
    if isinstance(trace, histories.History):
        if dt is not None:
            trace = anlz.subsample(trace, target_dt=dt)
    return trace
def trace_iterator(records, dt=default_dt):
    script = records.latest.main_file.rsplit('/',1)[1]
    records.list  # Allow tqdm to get length
    for rec in tqdm(records):
        assert rec.main_file.rsplit('/',1)[1] == script
        trace = load_trace(rec, dt=dt)
        if trace is not None:
            yield trace

# ======================================
# Project specific data structures

# ================
# Trace statistics

from functools import lru_cache

class BarStats:
    def __init__(self, reftraces, *, bootstrap_samples=100, **tracelists):
        """
        bootstrap_samples: int
            Set to `False` or `0` to avoid computing any bootstrap statistics.
        """
        super().__init__()
        # Trial-averaged statistics have to be computed on the same number
        # of trials
        counts = [len(t) for t in tracelists.values()]
        ntraces = min(counts)
        if ntraces != max(counts):
            for k, traces in tracelists.items():
                if isinstance(traces, (list, tuple)):
                    tracelists[k] = traces[:ntraces]
                elif isinstance(traces, dict):
                    tracelists[k] = \
                        {seed: trace for (seed, trace), _
                                     in zip(traces.items(), range(ntraces))}
                elif isinstance(traces, Iterable):
                    tracelists[k] = [t for t,_ in zip(traces, range(ntraces))]
                else:
                    raise ValueError("Unrecognized trace list type '{}'."
                                     .format(type(traces)))
            warn("Not all lists of traces have the same length. Only {} traces "
                 "from each set were kept, to keep statistics consistent."
                 .format(ntraces))
        refbar = sum(reftraces) / len(reftraces)
        self.ρ = AxAttrs()
        self.rmse = AxAttrs()
        for k, traces in tracelists.items():
            tracesbar = sum(traces) / len(traces)
            _ρ = corr(tracesbar, refbar)
            self.ρ[k] = [r[0] for r in _ρ]
            self.rmse[k] = rms(tracesbar, refbar)
        # Generate bootstrap statistics
        if not bootstrap_samples:
            self.bootstrap_stds = None
        else:
            # Create multiple BarStats instance with subsets of traces
            # Each subset has size `ntraces-1`
            boot_stats = deque()
            for _ in range(bootstrap_samples):
                iarr = np.random.choice(len(reftraces), size=len(reftraces))
                reftraces_boot = deque(reftraces[i] for i in iarr)
                tracelists_boot = {}
                for k, traces in tracelists.items():
                    iarr = np.random.choice(ntraces, size=ntraces)
                    tracelists_boot[k] = deque(traces[i] for i in iarr)
                boot_stats.append(
                    BarStats(reftraces_boot, bootstrap_samples=False,
                             **tracelists_boot))
            self.bootstrap_std = AxAttrs(ρ=AxAttrs(), rmse=AxAttrs())
                # ρ = {'true':np.std(
                #     [np.mean(stats.ρ.true) for stats in boot_stats]) },
                # rmse = {'true':np.std(
                #     [np.mean(stats.rmse.true) for stats in boot_stats])}
                # )
            for k in tracelists:
                self.bootstrap_std['ρ'][k] = np.std(
                    [np.mean(stats.ρ[k]) for stats in boot_stats], ddof=1)
                self.bootstrap_std['rmse'][k] = np.std(
                    [np.mean(stats.rmse[k]) for stats in boot_stats], ddof=1)

    def asdict(self):
        return {'ρ': self.ρ, 'rmse': self.rmse}
    def __str__(self):
        return str(self.asdict())
    def __repr__(self):
        return repr(self.asdict())

class TraceStats:
    def __init__(self, dist=stats.norm, dt=None):
        self.traces = None
        self.acc = None
        self.acc2 = None
        self.time = None
        self.N = 0
        self.dt = dt
        if isinstance(dist, stats.norm):
            self.dist = dist(self.μ, self.σ)
        else:
            raise NotImplementedError
    @property
    def μ(self):
        if self.acc is None: return None
        return self.acc/self.N
    @property
    def Σ(self):
        if self.acc2 is None: return None
        return self.acc2/self.N - self.μ[:,np.newaxis,...]**2
    @property
    def σ(self):
        if self.acc2 is None: return None
        return np.sqrt(self.Σ)
    def ppf(self, q):
        return self.dist.ppf(q)
    def add_trace(self, hist):
        if isinstance(trace, histories.History):
            if self.dt is not None:
                trace = anlz.subsample(trace, target_dt=self.dt)
            time = trace.time
            trace = trace.trace
        else:
            if time is None:
                time = np.arange(len(trace))
            if self.dt is not None:
                assert isinstance(self.dt, int) and self.dt >= 1
                time = anlz.subsample(time, amount=self.dt)
                trace = anlz.subsample(trace, amount=self.dt)
        if self.acc is None:
            assert self.acc2 is None
            self.time = hist.time
            self.acc = hist.trace
            self.acc2 = hist.trace**2
        else:
            assert np.all(self.time == hist.time)
            self.acc += hist.trace
            self.acc2 += hist.trace**2
        self.N += 1
        if not self.traces is None:
            self.traces.append(trace)

class TracePDF:
    def __init__(self, dt=None, timelim=None):
        """
        Uses a small averaging window to smooth out fluctuations due
        to limited data and finite populations.

        Parameters
        ----------
        timelim: (float, float)
            Only used if loading histories (instead of plain arrays).
            In this case the loaded history is truncated to the (min,max)
            values given by `timelim`.
        """
        self.dt = dt
        self.ordered_samples = None
        self.traces = deque()  # We need the traces for RMSE, corr stats
        self.time = None
        if timelim is not None:
            assert len(timelim) == 2
        self.timelim = timelim
        self.N = 0
        self._splines = None
        self._cdf = None
    def __add__(self, other):
        if not isinstance(other, TracePDF):
            raise ValueError
        assert other.dt == self.dt and other.timelim == self.timelim
        new = TracePDF(dt=None, timelim=None)  # Set None: already susbsampled
        for trace in self.traces:
            new.add_trace(trace, self.time)
        for trace in other.traces:
            new.add_trace(trace, other.time)
        return new

    @property
    def μ(self):
        if self.ordered_samples is None: return None
        return np.sum(self.ordered_samples, axis=1) / self.N
    @property
    def Σ(self):
        if self.ordered_samples is None: return None
        μ = self.μ[:,None]
        return np.sum((self.ordered_samples - μ)**2, axis=1) / self.N
    @property
    def σ(self):
        if self.ordered_samples is None: return None
        return np.sqrt(self.Σ)
    def sampleppf(self, q):
        """
        Note: ppf for samples is somewhat ambiguous:
        Should ppf(99) return last or second to last sample ?
        What about ppf(1) ?
        """
        if q < 0 or q > 1:
            return np.nan
        i = int(q // (1/self.N))
        return self.ordered_samples[:,i,...]
    @lru_cache(maxsize=4)
    def kernelppf(self, q):
        """
        This function attempts to address the coarseness of the `sampleppf`
        method by fitting a spline to the histogram at every time point,
        computing the cdf and then solving for cdf(A) = q.
        Assuming the ppf is continuous, with enough data this should converge
        to something smooth (in contrast to the sampleppf). In practice
        however, with 100 traces we still see larger jumps than with `sampleppf`.
        """
        return np.array([brentq(lambda A: cdf(A)-q, cdf.t.min(), cdf.t.max())
                         for cdf in tqdm(self._cdfsplines.flat)]) \
               .reshape(self._cdfsplines.shape)
    ppf = sampleppf
    def add_trace(self, trace, time=None, inplace=False):
        if isinstance(trace, histories.History):
            if self.timelim is not None:
                trace = trace.truncate(self.timelim[0], self.timelim[1], inplace=inplace)
            if self.dt is not None:
                trace = anlz.subsample(trace, target_dt=self.dt)
            time = trace.time
            trace = trace.trace
        else:
            if time is None:
                time = np.arange(len(trace))
            if self.dt is not None:
                assert isinstance(self.dt, int) and self.dt >= 1
                time = anlz.subsample(time, amount=self.dt)
                trace = anlz.subsample(trace, amount=self.dt)
        if trace.ndim == 1:
            trace = trace[:,np.newaxis]
        if self.ordered_samples is None:
            self.time = time
            self.ordered_samples = trace[:,np.newaxis,...]
        else:
            assert np.all(self.time == time)
            x = self.ordered_samples
            shape = (x.shape[0], x.shape[1]+1) + x.shape[2:]
            _x = np.empty(shape)
            slice_idcs = list(itertools.product(*(range(n) for n in x.shape[2:])))
            for i, (v,a) in enumerate(zip(x, trace)):
                for jj in slice_idcs:
                    vslice = (slice(None),) + jj
                    xslice = (i,) + vslice
                    k = np.searchsorted(v[vslice], a[jj])
                    _x[xslice] = np.insert(v[vslice], k, a[jj])
            self.ordered_samples = _x
        self.traces.append(trace)
        self.N += 1
        self._splines = None  # Invalidate splines
        self._cdf = None
    @property
    def _pdfsplines(self):
        if self._splines is None:
            self._splines = self._compute_splines()
        return self._splines
    @property
    def _cdfsplines(self):
        if self._cdf is None:
            self._cdf = self._compute_cdf()
        return self._cdf
    def _compute_splines(self):
        # Get possible count values
        x = self.ordered_samples
        slice_idcs = list(itertools.product(*(range(n) for n in x.shape[2:])))
        splines = np.empty((len(x),*x.shape[2:]), dtype=np.object)
        Avals = np.array([np.unique(x[(np.s_[:],np.s_[:],*jj)]) for jj in slice_idcs]) \
                .reshape(x.shape[2:])
        for ti in tqdm(range(len(x))):
            for jj in slice_idcs:
                A, counts = np.unique(x[(ti,slice(None),*jj)], return_counts=True)
                ## Add a point at end to ensure pdf goes to zero
                ## For small counts this might be useful?
                #Δ = np.diff(A).max()
                #A = np.concatenate((A, [A[-1]+Δ]))
                #counts = np.concatenate((counts, [0.]))
                k = 3  # Spline order
                if len(A) == 1:
                    # This should only happen on small datasets used for testing, so a kludgy solution is OK
                    assert A.shape == (1,)
                    assert counts.shape == (1,)
                    i = np.searchsorted(Avals[jj], A[0])
                    A = Avals[jj]
                    count = counts[0]
                    counts = np.zeros(A.shape)
                    counts[i] = count
                elif len(A) <= 3:
                    k = len(A) - 1
                t,c,k = splrep(A,counts,k=k)
                spline = BSpline(t,c,k)
                # Normalize so we have unit integral
                I = spline.integrate(A[0], A[-1])
                splines[(ti,)+jj] = BSpline(t,c/I,k)
        #splines = np.array([BSpline(*splrep(
        #                            np.unique(x[(ti,slice(None))+jj], return_counts=True)))
        #                    for i in range(len(x)) for jj in slice_idcs]).reshape(x.shape[2:])
        return splines
    def _compute_cdf(self):
        return np.array([spline.antiderivative() for spline in tqdm(self._pdfsplines.flat)]) \
                    .reshape(self._pdfsplines.shape)

def trace_stats(stat_fn):
    TraceColl = (TraceStats, TracePDF)
    def trace_stat_wrapper(data_traces, ref_traces):
        if isinstance(data_traces, TraceColl) and isinstance(ref_traces, TraceColl):
            assert data_traces.dt == ref_traces.dt
            L = min(len(data_traces.time), len(ref_traces.time))
            assert np.all(np.isclose(data_traces.time[:L], ref_traces.time[:L]))
                 # HACK: assumes aligned at 0
            if L < len(data_traces.time):
                data_traces = np.array(data_traces.traces)[:,:L]
            else:
                data_traces = np.array(data_traces.traces)
            if L < len(ref_traces.time):
                ref_traces = np.array(ref_traces.traces)[:,:L]
            else:
                ref_traces = np.array(ref_traces.traces)
        elif isinstance(data_traces, TraceColl):
            ref_traces = np.array(ref_traces)
            assert len(data_traces.time) == ref_traces.shape[1]
            data_traces = np.array(data_traces.traces)
        elif isinstance(ref_traces, TraceColl):
            data_traces = np.array(data_traces)
            assert len(ref_traces.time) == data_traces.shape[1]
            ref_traces = np.array(ref_traces.traces)
        else:
            data_traces = np.array(data_traces)
            ref_traces = np.array(ref_traces)

        return stat_fn(data_traces, ref_traces)

    return trace_stat_wrapper

@trace_stats
def trace_rms(data_traces, ref_traces):
    """Returns (a,b) such that all rms values are within a±b."""
    assert data_traces.shape[1:] == ref_traces.shape[1:]
    _rms = np.zeros((len(data_traces)*len(ref_traces), *data_traces.shape[2:]))
    for i, (dtrace, rtrace) in enumerate(itertools.product(data_traces, ref_traces)):
        _rms[i] = rms(dtrace, rtrace)
    #a, b = _rms.min(axis=0), _rms.max(axis=0)
    #return np.stack(((a+b)/2, (a-b)/2)).T
    μ = np.mean(_rms, axis=0); σ = np.std(_rms, axis=0)
    return np.stack((μ, σ)).T

@trace_stats
def trace_corr(data_traces, ref_traces, pthresh=0.001):
    """Returns (a,b) such that all PearsonR values are within a±b."""
    assert data_traces.shape[1:] == ref_traces.shape[1:]
    _corr = np.zeros((len(data_traces)*len(ref_traces), *data_traces.shape[2:], 2))
    for i, (dtrace, rtrace) in enumerate(itertools.product(data_traces, ref_traces)):
        _corr[i] = corr(dtrace, rtrace)
    # Check that no p-value exceeds threshold
    if np.any(_corr[...,1] > pthresh):
        warn("Correlation p-values are as high as {}.".format(np.max(_corr[...,1])))
    #a, b = _corr.min(axis=0), _corr.max(axis=0)
    #return np.stack(((a+b)/2, (a-b)/2)).T
    μ = np.mean(_corr[...,0], axis=0); σ = np.std(_corr[...,0], axis=0)
    return np.stack((μ, σ)).T

def load_traces(records, dt=default_dt, timelim=None, inplace=False):
    pdf = TracePDF(dt=dt, timelim=timelim)
    for trace in trace_iterator(records):
        pdf.add_trace(trace, inplace=inplace)
    return pdf

class SimStats:
    def __init__(self, data_traces, ref_traces, timelim=None, dt=default_dt):
        if isinstance(data_traces, RecordList):
            data_traces = load_traces(data_traces, timelim=timelim, dt=dt)
        if isinstance(ref_traces, RecordList):
            ref_traces = load_traces(ref_traces, timelim=timelim, dt=dt)
        self.corr = trace_corr(data_traces, ref_traces)
        self.rms  = trace_rms(data_traces, ref_traces)
        self.traces = data_traces
class SimStatsColl:
    def __init__(self, ref_sims, *, timelim=None, dt=default_dt, **test_sims):
        if isinstance(ref_sims, RecordList):
            ref_sims = load_traces(ref_sims, timelim=timelim, dt=dt)
        self.test_sims = {k: SimStats(v, ref_sims, timelim=timelim, dt=dt)
                          for k,v in test_sims.items()}
        self.ref_sims = ref_sims
        #time = None
        #for test_sim in self.test_sims.values():
        #    if time is None:
        #        time = getattr(test_sim.traces, 'time', None)
        #    else:
        #        assert np.all(np.isclose(time, test_sim.traces.time))
        #self.time = time
    def __getitem__(self, key):
        return self.test_sims[key]
    def __getattr__(self, key):
        return self.test_sims[key]
    @property
    def corr(self):
        return {k: sim.corr for k,sim in self.test_sims.items()}
    @property
    def rms(self):
        return {k: sim.rms for k,sim in self.test_sims.items()}

# ================
# Fit collections

ParamFit = namedtuple('ParamFit', ['indices', 'labels', 'target', 'yticks', 'ylim', 'yscale'])
ParamFit.__new__.__defaults__ = (None, None, None, None, None, 'linear')

# Default plot configurations
plotdescs_2pop = {
    'w'  : ParamFit(yticks=[-8, 0, 8], ylim=(-12.5, 12.5)),
    'τ_θ': ParamFit(indices=[0], yticks=[1e-10, 1e-4, 1e2], yscale='log'),
    'J_θ': ParamFit(indices=[0], yticks=[0, 1, 2, 3], ylim=(0, 3), yscale='linear'),
    'c'  : ParamFit(yticks=[0, 10, 20], ylim=(0, 29)),
    'Δu' : ParamFit(yticks=[0, 10, 20, 30], ylim=(0, 36)),
    'τ_m': ParamFit(yticks=[1e-5, 1e0, 1e5, 1e10], ylim=(1e-7, 1e8), yscale='log'),
    'τ_s': ParamFit(yticks=[1e-7, 1e-2, 1e3], ylim=(1e-10, 1e5), yscale='log')
}
plotdescs_4pop = {
    'w'  : ParamFit(yticks=[-8, 0, 8], ylim=(-12.5, 12.5)),
    'τ_θ': ParamFit(indices=[0, 1], yticks=[1e-10, 1e-4, 1e2], yscale='log'),
    'J_θ': ParamFit(indices=[0, 1], yticks=[0, 1, 2, 3], ylim=(0, 5)),
    'c'  : ParamFit(yticks=[0, 10, 20], ylim=(0,35)),
    'Δu' : ParamFit(yticks=[0, 10, 20, 30], ylim=(0, 45)),
    'τ_m': ParamFit(yticks=[1e-5, 1e0, 1e5, 1e10], ylim=(1e-7, 1e8), yscale='log'),
    'τ_s': ParamFit(yticks=[1e-7, 1e-2, 1e3], ylim=(1e-10, 1e5), yscale='log')
}

class FitSimCompare:
    def __init__(self, fits, truesim, truesimref, input_params,
                 varstrings=None, avg_in_log_space=False):
        self.fits       = fits
        self.truesim    = truesim
        self.truesimref = truesimref

        # Load fits
        if varstrings is None: varstrings = globals()['varstrings']
        self.fitcolls = {}
        for key, recs in self.fits.items():
            self.fitcolls[key] = sinn.optimize.gradient_descent.FitCollection()
            self.fitcolls[key].load(recs.extract("parameters", "outputpath"))
            self.fitcolls[key].set_varstrings(varstrings, check_names=False)

        # Load ground truth sims
        # TODO: we should not need to use _data
        self.truesim.load(('activity', 'expected_activity'))
        self.truesimref.load(('activity', 'expected_activity'))
        for path, data in self.truesim.data.items():
            if 'expected_activity' in path:
                self.true_a = data._data
            elif 'activity' in path:
                self.true_A = data._data
        for path, data in self.truesimref.data.items():
            if 'expected_activity' in path:
                self.true_a_ref = data._data
            elif 'activity' in path:
                self.true_A_ref = data._data

        # Get the parameter set used to simulate
        # TODO: Check that parameter set is the same as all sims
        fitcoll = next(iter(self.fitcolls.values()))
        self.pset = ml.parameters.params_to_arrays(
            fitcoll.reffit.parameters.posterior.data.params.model)
        self.avg_pset = core.average_hetero_params(
            self.pset, transform=avg_in_log_space)

        # Set the time interval for plots
        # Default values; change by assigning new value to the attribute
        hist = next(iter(self.truesim.data.values()))
        self.start = hist.t0
        self.end = hist.tn
        self.dt = hist.dt
        #self.start, self.end, self.dt = None, None, None

        # Get simulations from the fit results
        self.result_nbar = {}
        self.result_A = {}
        for key, fitcoll in self.fitcolls.items():
            nbar = core.get_result_sim(
                fitcoll, input_params=input_params, suffix='nbar',
                desc='map-activity_hetero-data_{}'.format(key))
            if nbar is None:
                self.result_nbar[key] = None
                self.result_A[key] = None
            else:
                self.result_nbar[key] = nbar
                self.result_A[key] = core.get_result_sim(
                    fitcoll, input_params=input_params, suffix=None,
                    desc='map-activity_hetero-data_'.format(key))

        # Get simulations using theoretical parameters
        _pset= core.get_pset(fitcoll, self.avg_pset, input_params=input_params)
        nbar = core.get_param_sim(
            _pset, desc="activity_avg-hetero-parameters", suffix='nbar')
        if nbar is None:
            self.avg_nbar = None
            self.avg_A = None
        else:
            self.avg_nbar = nbar
            self.avg_A = core.get_param_sim(
                _pset, desc="activity_avg-hetero-parameters", suffix=None)

    def __getitem__(self, key):
        return self.fitcolls[key]

    def fit_counts(self):
        return pd.DataFrame({key: {'Number of fits': len(recs)}
                             for key, recs in self.fits.items()})

    def set_hists(self):
        if getattr(self, 'result_hists_a', None) is not None:
            return
        start, end, dt = self.start, self.end, self.dt
        N = self.pset.N
        def subsample(hist):
            return hist
            # return core.subsample(hist, target_dt=dt, aggregation='mean')
        self.result_hists_a = {
            key: subsample(nbar.truncate(start, end) / N / nbar.dt)
            for key, nbar in self.result_nbar.items() }
        #result_hists_A = {
        #    key: subsample(A.truncate(start, end))
        #    for key, A in self.result_A.items()}
        #}
        self.avg_hist_a = subsample(
            self.avg_nbar.truncate(start, end) / N / self.avg_nbar.dt)
        #avg_hist_A = subsample(self.avg_A.truncate(start, end))
        self.true_a = subsample(self.true_a.truncate(start, end))
        self.true_a_ref = subsample(self.true_a_ref.truncate(start, end))

class FitResultSims:
    """
    Collection of fit collections. To each FitCollection we associate a
    simulation using the result of that fit.
    Each fit collection can use different fit parameters, but should be
    inferring the same model parameters.
    """

    def __init__(self, fitrecords, input_params, start, stop,
                 key_desc="", trace_records=None, trace_psets=None,
                 seeds=4, trueseeds=4, seed0=314, trueseed0=None):
        """
        Parameters
        ----------
        fitrecords: dict
            key: whatever was used to classify the fits
            value: instance of RecordSet. Each RecordSet creates one
                   FitCollection.
        input_params: ParameterSet
            Definition for the test input. Does not need to be the same as the
            input used for fitting.
        start, stop: int | float
            Set the time interval over which to compute error between simulated
            and true trace.
        key_desc: str
            Description of what the key represents (e.g. "Data length: {}s").
            Used when printing diagnostics for missing data. If it contains
            the characters '{}', they are replaced by the key. Otherwise
            `print(key_desc, key)` is printed.
        trace_records: dict
            NOT IMPLEMENTED
            Will be checked for entries matching the keys in `fitrecords`.
            If found, these are used as the simulated trace. Otherwise, we try
            to load a simulated trace based on the fit result parameters.
            Note that for keys for which we provide a trace, the fit collection
            is effectively ignored.
            key: same as `fitrecords`
            value: record label or record
        trace_psets: dict
            Hack to force a particular parameter set, rather than the one
            obtained by reading 'fitrecords' – corresponding fit collection
            is ignored. Bypasses provenance checks, so use with caution.
        seeds: int | iterable of ints
            Number of seeds to use for traces. If an iterable, should not share
            elements with `trueseeds`, to avoid noise correlations.
        seed0: int
            If seeds are assigned sequentially, they start from this value.
            Ignored if `seeds` is an iterable.
        trueseeds: int | iterable of ints
            Random seeds used to generate the test simulations with same model
            and parameters as the reference.
        trueseed0: int | None
            If trueseeds are assigned sequentially, they start from this value.
            Ignored if `trueseeds` is an iterable.
            If None, set to the next largest seed after `seeds`.
        """
        if isinstance(seeds, int):
            seeds = np.arange(seed0, seed0+seeds)
        assert isinstance(seeds, Iterable)
        if trueseed0 is None:
            trueseed0 = max(seeds) + 1
        if isinstance(trueseeds, int):
            trueseeds = np.arange(trueseed0, trueseed0+trueseeds+1)
        assert isinstance(trueseeds, Iterable)
        if set(seeds).intersection(trueseeds) != set():
            logger.warning("One of the values in `trueseeds` is `seed`. "
                           "You probably don't want that, as you this will "
                           "overestimate performance.")
        #if refseed is None: refseed = max(trueseeds)+100
        #assert not any(s == refseed for s in trueseeds)
        self.seeds = seeds
        self.trueseeds = trueseeds
        #self.refseed = refseed
        if trace_records is None:
            trace_records = {}
        if trace_psets is None:
            trace_psets = {}
        self.Atrue = {}       # Simulations with the true parameters
        # self.Atrue = None   # True simulation used as reference for corr and RMS
        # self.Atrue2 = None  # True simulation using same seed as inferred models
        # self.Atrue_actual = None  # Atrue is actually a; this stores actual A;
        #                           # we do it this way for backwards compat.
        self.time = None
        self.true_dt = None
        self.tslice = slice(start, stop)  # Store the slice used for calculating errors
        self.traces = {}                  # TODO: traces -> exp_traces
        self.traces_actual = {}           # TODO: traces_actual -> traces
        self.fitcolls = {}
        for key, fitrecs in fitrecords.items():
            if isinstance(fitrecs, FitCollection):
                self.fitcolls[key] = fitrecs
            else:
                self.fitcolls[key] = FitCollView(
                    list(fitrecs.filter.script('gradient_descent')
                                .extract("parameters", "outputpath")))
        # Error measures
        self._rms = None
        self._corr = None

        for key, rec in trace_records.items():
            raise NotImplementedError

        for key, fitcoll in self.fitcolls.items():
            if len(fitcoll.fits) == 0:
                # No fits; nothing to do
                continue
            posterior = fitcoll.reffit.parameters.posterior
            sgd = fitcoll.reffit.parameters.sgd
            modelparams = ml.parameters.params_to_arrays(
                posterior.model.params)
            Npops = len(posterior.model.params.N)

            if (posterior.input != posterior.data.params.input):
                raise ValueError("Garbage data: fit and data input do not match.")

            if posterior.data.dir == 'spikes':
                datamodel = 'spikes'
                data_sim_kwargs = {
                    'suffix'      : 'expected_activity',
                    'desc'        : 'λbar_true-params',
                    'missing_msgs': 'Could not find a reference λbar.\n'
                }
                mapdesc = ('{}-activity_data-spikes_{}pops'
                           .format(sgd.cost, Npops))
                dt = 0.0002 # HACK: hardcoded
                def data_transform(A):
                    return anlz.subsample(A, target_dt=0.001)
            else:
                assert posterior.data.dir == 'activity'
                datamodel = 'activity'
                data_sim_kwargs = {
                    'suffix'      : 'nbar',
                    'desc'        : 'activity_true-params',
                    'missing_msgs': 'Could not find a reference nbar.\n'
                }
                mapdesc = ('{}-activity_data-activity_{}pops'
                           .format(sgd.cost, Npops))
                dt = 0.001 # HACK: hardcoded
                def data_transform(nbar):
                    return nbar / modelparams.N / nbar.dt

            for _seed in list(seeds) + list(trueseeds):# + [refseed]:
                # FIXME: What happens if the seed is the same but p's differ ?
                if _seed in self.Atrue: continue
                p = core.get_pset(fitcoll.reffit, modelparams,
                                  input_params=input_params, seed=_seed)
                p.dt = dt
                datatrue = core.get_param_sim(p, model=datamodel,
                                              **data_sim_kwargs)
                if datatrue is not None:
                    datatrue = data_transform(datatrue)
                    slc = datatrue.get_tidx(self.tslice)
                    self.Atrue[_seed] = datatrue[slc]
                    if self.time is None:
                        self.time = datatrue._tarr[slc]
                    true_dt = datatrue.dt  # Would not be necessary if Atrue were a history
                else:
                    # Don't try to load nbartrue again
                    self.Atrue[_seed] = None

            # p.seed = refseed
            # data_sim_kwargs['suffix'] = None
            # datatrue_actual = core.get_param_sim(p, model=datamodel,
            #                               **data_sim_kwargs)
            # if isinstance(datatrue_actual, histories.Spiketrain):
            #     csc = datatrue_actual._data.tocsc()
            #     tslc = datatrue_actual.get_tidx(self.tslice)
            #     self.Atrue_actual = np.concatenate(
            #         [csc[tslc,slc].sum(axis=-1).A
            #          for slc in datatrue_actual.pop_slices], axis=1)
            # else:
            #     self.Atrue_actual = datatrue_actual
            # if self.Atrue_actual is not None:
            #     self.Atrue_actual = \
            #         anlz.subsample(self.Atrue_actual,
            #                        amount = int(round(0.001/dt)),
            #                        aggregation='sum'
            #         ) / modelparams.N / 0.001

            for _seed in seeds:
                if key in trace_psets and _seed in trace_psets[key]:
                    p = trace_psets[key][_seed]
                    nbar = core.get_param_sim(
                        p, suffix='nbar',
                        desc="{}-activity_{}pops".format(sgd.cost, Npops))
                    A = core.get_param_sim(
                        p, suffix=None,
                        desc="{}-activity_{}pops".format(sgd.cost, Npops))
                else:
                    nbar = core.get_result_sim(
                        fitcoll, input_params=input_params, suffix='nbar',
                        desc="{}-activity_{}pops".format(sgd.cost, Npops),
                        seed=_seed)
                    A = core.get_result_sim(
                        fitcoll, input_params=input_params, suffix=None,
                        desc="{}-activity_{}pops".format(sgd.cost, Npops),
                        seed=_seed)
                if nbar is None:
                    # An error message was already printed; add to the output the key
                    if '{}' in key_desc:
                        print(key_desc.format(key))
                    else:
                        print(key_desc, key)
                else:
                    if datatrue is not None: assert(nbar.dt == datatrue.dt)
                    if key not in self.traces:
                        self.traces[key] = {}
                    self.traces[key][_seed] = nbar[self.tslice] / modelparams.N / nbar.dt
                    slc = A.get_tidx(self.tslice)
                    self.traces_actual[key] = A[slc]

    def add_sim(self, key, params, model='activity',
                desc="", missing_msgs=None, key_desc=""):
        if isinstance(params, str):
            params = ParameterSet(params, basepath=core.home_dir)
        modelparams = ml.parameters.params_to_arrays(params.model)

        # if key in self.traces:
        #     # FIXME: I don't understand what this is supposed to accomplish
        #     rec = Record(traces[key])
        #     nbar = ml.iotools.load(rec.get_outputpath('nbar'))
        # else:
        if key in self.traces:
            logger.warning("A simulation corresponding to key {} is already "
                           "loaded.".format(key))
            return
        nbar = core.get_param_sim(
            params, suffix='nbar', datadir=None, model=model,
            desc=desc, missing_msgs=missing_msgs)
        A = core.get_param_sim(
            params, suffix=None, datadir=None, model=model,
            desc=desc, missing_msgs=missing_msgs)
        if nbar is None:
            # An error message was already printed; add to the output the key
            # TODO: Add to `missing_msgs`
            if '{}' in key_desc:
                print(key_desc.format(key))
            else:
                print(key_desc, key)
        else:
            slc = nbar.get_tidx(self.tslice)
            if hasattr(self, 'time') and self.time is not None:
                assert np.isclose(self.time, nbar._tarr[slc]).all()
            else:
                self.time = nbar._tarr[slc]
            self.traces[key] = nbar[slc] / modelparams.N / nbar.dt
            slc = A.get_tidx(self.tslice)
            self.traces_actual[key] = A[slc]

    def __str__(self):
        return "FitResultSims {{{}}}".format(
            ', '.join(str(k) for k in self.traces.keys()))

    def __repr__(self):
        # TODO: More informative repr ?
        return str(self)

    @property
    def rms(self):
        # FIXME: RMS dict is currently frozen after first call.
        #        Problem if e.g. we later add fits / fitcolls.
        if self._rms is None: self._rms = {}
        for key, traces in self.traces.items():
            if key not in self._rms:
                self._rms[key] = {(seed, trueseed):
                                    rms(traces[seed], self.Atrue[trueseed])
                                  for seed in self.seeds
                                  for trueseed in self.trueseeds}

        if 'true' not in self._rms:
            self._rms['true'] = {(seed, trueseed):
                                rms(self.Atrue[seed], self.Atrue[trueseed])
                              for seed in self.seeds
                              for trueseed in self.trueseeds}
        # if (len(self.Atrue) >= 2
        #     and all(A is not None for A in self.Atrue.values())):
        #     reftrace = self.Atrue[self.refseed]
        #     self._rms['true'] = {seed: rms(reftrace, self.Atrue[seed])
        #                          for seed in self.trueseeds}
        return self._rms

    @property
    def corr(self):
        # FIXME: corr dict is currently frozen after first call.
        #        Problem if e.g. we later add fits / fitcolls.
        if self._corr is None: self._corr = {}
        for key, traces in self.traces.items():
            if key not in self._corr:
                self._corr[key] = {(seed, trueseed):
                                     corr(traces[seed], self.Atrue[trueseed])
                                   for seed in self.seeds
                                   for trueseed in self.trueseeds}
        if 'true' not in self._corr:
            self._corr['true'] = {(seed, trueseed):
                                 corr(self.Atrue[seed], self.Atrue[trueseed])
                               for seed in self.seeds
                               for trueseed in self.trueseeds}
        # if (len(self.Atrue) >= 2
        #     and all(A is not None for A in self.Atrue.values())):
        #     reftrace = self.Atrue[self.refseed]
        #     self._corr['true'] = {seed: corr(reftrace, self.Atrue[seed])
        #                           for seed in self.trueseeds}
        ρ = {key: {seedtuple: np.array([r for r,p in rp])
                   for seedtuple, rp in corr.items()}
             for key, corr in self._corr.items()}
        return ρ

    @property
    def fit_counts(self):
        return pd.DataFrame({L:len(fc) for L, fc in self.fitcolls.items()},
                            index=["No. of fits"])

    def set_varstrings(self, varstrings, check_names=True):
        """
        Call `set_varstring()` on all collections.
        """
        for fitcoll in self.fitcolls.values():
            fitcoll.set_varstrings(varstrings, check_names)

    def get_varstring(self, name, idx=None):
        """
        Call `get_varstring()` on the first fit collection (all fit collections
        have the same parameters, and so their `get_varstring` functions should
        be equivalent).
        """
        return next(iter(self.fitcolls.values())).get_varstring(name, idx)

def rms(trace, reftrace):
    return np.sqrt( ( (trace-reftrace)**2 ).mean(axis=0) )

def corr(trace, reftrace):
    return [sp.stats.pearsonr(trace[:,i], reftrace[:,i])
            for i in range(trace.shape[1])]

def get_data_model_params(fitcoll):
    """
    Extract the model parameters that were used to generated the synthetic data for a FitCollection. For heterogeneous parameters, looks for a `loc`
    """
    pset = fitcoll.reffit.parameters.posterior.data.params.model
    populations = pset.get('populations', None)
    modelparams = ParameterSet({})
    for pname in fitcoll.reffit.data._trace.keys():
        param = pset[pname]
        pshape = fitcoll.reffit.data._trace[pname].shape[1:]
        if isinstance(param, ParameterSet):
            # Find loc
            if populations is not None:
                loc = []
                for α in populations:
                    if α in param and 'loc' in param[α]:
                        loc.append(param[α].loc)
                    else:
                        loc.append(param.loc)
            else:
                loc = [param.loc]*np.prod(pshape)
            loc = np.array(loc)
            # If the loc is for a transformed variable, invert the transform
            if 'transform' in param:
                backtransform = ml.parameters.Transform(param.transform.back)
                loc = backtransform(loc)
            # Add to params
            modelparams[pname] = loc
        else:
            modelparams[pname] = np.array(param)
        # If necessary, broadcast up the model parameter to the right shape
        #if modelparams[pname].size < np.prod(pshape):
        #    modelparams[pname] = modelparams[pname] * np.ones(pshape)
        #elif modelparams[pname].shape != pshape:
        #    modelparams[pname] = modelparams[pname].reshape(pshape)

    return modelparams

# ================================
# Project-specific plotting functions

def plot_fits(fitcoll, **kwargs):
    core.plot_fit([tup for tup in core.flat_params if tup[0] in fitcoll.MLE],
                   fitcoll, ncols=5, colwidth=3.25, rowheight=1.5,
                   format=core.plotformat,
                   only_finite=False,
                   xscale='linear',
                   **kwargs)

from operator import sub
def plot_series(hist, xlim=None, ylim=None, xscalelen=None, xscaleoffset=0,
                ybounds=None, ylabel="", ylabelshift=0, idx=None, ax=None, color=None, c=None,
                **kwargs):
    if idx is None: idx = slice(None)
    if ax is None: ax = plt.gca()
    traces = hist.trace[:, idx]
    if c is not None and color is not None:
        logger.warning("Both `c` and `color` were specified. Keeping `color`")
    elif c is not None:
        color = c
    if isinstance(color, Iterable) and not isinstance(color, str):
        for i, c in zip(range(traces.shape[1]), color):
            ax.plot(hist.time, traces[:,i], color=c, **kwargs)
    else:
        ax.plot(hist.time, hist.trace[:,idx], color=color, **kwargs)

    if xlim is not None: ax.set_xlim(xlim)
    else: xlim = ax.get_xlim()
    x0 = float(xlim[0]); xn = float(xlim[1])
    if ylim is None: ylim = (hist[x0:xn].min(), hist[x0:xn].max())
    ax.set_ylim(ylim)
    if ybounds is None: ybounds = ax.get_ylim()
    if xscalelen is None:
        xscalelen = -sub(*xlim) * 0.2
    ml.plot.draw_xscale(xscalelen, "{}s".format(ml.utils.int_if_close(xscalelen)),
                        offset=xscaleoffset)
    ax.set_yticks(ybounds)
    lspine = ax.spines['left']
    lspine.set_bounds(*ybounds)
    lspine.set_position(('outward', 4))
    # FIXME: Should do subtraction in display units
    ax.yaxis.set_label_coords(xlim[0] - 0.1*np.diff(xlim)[0] + ylabelshift, np.mean(ylim),
                              transform=ax.transData)
    ax.set_ylabel(ylabel, horizontalalignment='center')

def plot_series_stats(trace_stats, xlim=None, ylim=None, xscalelen=None, xscaleoffset=0,
                      ybounds=None, ylabel="", ylabelshift=0, idx=None, maxstops=1000, ax=None,
                      linecolor=None, shadecolor=None, linewidth=None,
                      zorder=0, alpha=1., label=None):
    if ax is None: ax = plt.gca()

    # Decimate histories so they don't have too many points to plot
    #assert np.all(trace_stats.μ.time == trace_stats.σ.time)
    step = max(int(len(trace_stats.μ) / maxstops), 1)
    #μ = anlz.decimate(trace_stats.μ, step)
    #σ = anlz.decimate(trace_stats.σ, step)
    #time = anlz.decimate(trace_stats.time, step)
    assert isinstance(trace_stats.μ, np.ndarray)
    assert isinstance(trace_stats.σ, np.ndarray)
    assert isinstance(trace_stats.time, np.ndarray)
    μ = trace_stats.μ[::step]
    σ = trace_stats.σ[::step]
    time = trace_stats.time[::step]

    assert μ.ndim == 2
    if idx is None: idx = list(range(μ.shape[1]))
    elif not isinstance(idx, Iterable):
        idx = [idx]
    # From plot_step_sim_μσ
    idcs = [(slice(None),) + (_idx,) for _idx in idx]
        # Each index in idcs yieds all time points for one component
    if not isinstance(linecolor, Iterable) or isinstance(linecolor, str):
        linecolor = [linecolor]*len(idcs)
    if not isinstance(shadecolor, Iterable) or isinstance(shadecolor, str):
        shadecolor = [shadecolor]*len(idcs)
    # Each index in idcs yieds all time points for one component
    for idx, c in zip(idcs, shadecolor):
        if str(c).lower() != 'none':
            ax.fill_between(time, μ[idx]-σ[idx], μ[idx]+σ[idx],
                            linewidth=0,
                            color=c, zorder=zorder-100, alpha=0.5*alpha)
    for i, idx in enumerate(idcs):
        ax.plot(time, μ[idx], color=linecolor[i], zorder=zorder, alpha=alpha, label=label, linewidth=linewidth)

    # From plot_series
    if xlim is not None: ax.set_xlim(xlim)
    else: xlim = ax.get_xlim()
    x0 = float(xlim[0]); xn = float(xlim[1])
    if xscalelen is None:
        xscalelen = -sub(*xlim) * 0.2
    ml.plot.draw_xscale(xscalelen, "{}s".format(ml.utils.int_if_close(xscalelen)),
                        offset=xscaleoffset)
    #if ylim is None: ylim = (np.min(μ[x0:xn]-σ[x0:xn]), np.max(μ[x0:xn]+σ[x0:xn]))
    if ylim is None: ylim = (np.min(μ-σ), np.max(μ+σ))
    ax.set_ylim(ylim)
    if ybounds is None: ybounds = ax.get_ylim()
    lspine = ax.spines['left']
    ax.set_yticks(ybounds)
    lspine.set_bounds(*ybounds)
    lspine.set_position(('outward', 4))
    # FIXME: Should do subtraction in display units
    ax.yaxis.set_label_coords(xlim[0] - 0.1*np.diff(xlim)[0] + ylabelshift, np.mean(ylim),
                              transform=ax.transData)
    ax.set_ylabel(ylabel, horizontalalignment='center')

def plot_raster(hist, xlim=None, ylim=None, xscalelen=None, xscaleoffset=0,
                ybounds=None, ylabel="", ylabelshift=0,
                lineheight=1, markersize=1, ax=None, color=None, c=None, **kwargs):

    if ax is None: ax = plt.gca()
    if c is not None and color is not None:
        logger.warning("Both `c` and `color` were specified. Keeping `color`")
    elif c is not None:
        color = c
    if not isinstance(color, Iterable) or isinstance(color, str):
        color = [color]*len(hist.pop_slices)
    # Set number of neurons to print per population
    if ylim is not None:
        n = int(round(abs(sub(*ylim)) / len(hist.pop_slices)))
    else:
        n = 20
        ylim = (0, n*len(hist.pop_slices))
    if xlim is None: xlim = (hist.t0, hist.tn)
    tarr = hist._tarr[hist._data.row]
    tidcs = np.where(np.logical_and(xlim[0] <= tarr, tarr <= xlim[1]))[0]
    for i, (slc, c) in enumerate(zip(hist.pop_slices, color)):
        popstart = slc.start
        popstop = min(popstart+n, slc.stop)
        popidcs = np.where(np.logical_and(popstart <= hist._data.col,
                                          hist._data.col < popstop))[0]
        idcs = np.intersect1d(tidcs, popidcs)

        ax.scatter(hist._tarr[hist._data.row[idcs]],
                   (hist._data.col[idcs]-popstart)*lineheight + i*n*lineheight,
                   s = markersize, c=c, **kwargs)

    #ax.plot(hist.time, hist.trace, alpha=0.8)
    if xlim is not None: ax.set_xlim(xlim)
    else: xlim = ax.get_xlim()
    if ylim is not None: ax.set_ylim(ylim)
    else: ylim = ax.get_ylim()
    if ybounds is None: ybounds = ax.get_ylim()
    if xscalelen is None:
        xscalelen = -sub(*xlim) * 0.2
    ml.plot.draw_xscale(xscalelen, "{}s".format(ml.utils.int_if_close(xscalelen)),
                        offset=xscaleoffset)

    ax.set_yticks(ybounds)
    lspine = ax.spines['left']
    lspine.set_bounds(*ybounds)
    lspine.set_position(('outward', 4))
    ax.yaxis.set_label_coords(xlim[0] + 0.1*sub(*xlim) + ylabelshift, np.mean(ylim),
                              transform=ax.transData)
    ax.set_ylabel(ylabel, horizontalalignment='center')

def plot_fitcoll_grid(fitcoll, plotdescs, nrows = None, ncols = 3,
                      modelparams='data', plot_vars=None, labels=None,
                      figsize=None, **kwargs):
    """
    Parameters
    ----------
    ficoll: FitCollection
    nrows: int
    ncols: int
    modelparams: 'data' | ParameterSet | None
        'data': Load from `fitcoll.reffit.parameters.posterior.data.params.model`
        ParameterSet: ParameterSet or dict of parameters
        None: Don't draw lines for target values
    plot_vars: list | None
        list: Plot these variables (not implemented)
        None: Plot all tracked variables
    **kwargs: Extra keyword arguments are given to FitCollection.plot().
    """

    # Extract / prepare data
    fit = fitcoll.reffit.data
    if modelparams == 'data':
        modelparams = get_data_model_params(fitcoll)
    if plot_vars is None:
        plot_vars = fit.tracked
    # TODO: Do something with plot_vars

    #if plotdescs is None: plotdescs = fits2pop
    if labels is not None:
        for var, lbls in labels.items():
            plotdescs[var] = plotdescs[var]._replace(labels=lbls)
    plotdescs = OrderedDict((v, plotdescs[v]) for v in plot_vars)
    for v, d in plotdescs.items():
        l, i, t = d.labels, d.indices, d.target
        if l is None: l = fit.get_varstrings(v)
        if i is None: i = range(len(l))
        if t is None: t = modelparams
        plotdescs[v] = d._replace(**{'indices': i, 'labels': l, 'target': t})

    nplots = sum(len(d.indices) for d in plotdescs.values())
    if nrows is None:
        nrows = int(np.ceil(nplots/ncols))

    # If there are too many lines, pgf can't export a plot unless it's rasterized
    rasterize = len(fitcoll.fits) * nplots > 1000  # Heuristic

    # Do plot
    plt.figure(figsize=figsize)

    k = 0

    axes = []
    for v, d in plotdescs.items():
        for i, lbl in zip(d.indices, d.labels):
            k += 1
            ax = plt.subplot(nrows,ncols,k, rasterized=rasterize)
            fitcoll.plot(v, idx=i, ax=ax, target=modelparams,
                         ylabel=lbl, yticks=d.yticks, ylim=d.ylim, yscale=d.yscale, keep_range=1,
                         **kwargs)
            if k <= nplots - ncols: # Display iterations axis on bottom row
                ax.xaxis.set_visible(False)
            if k == (nrows-1)*ncols + 1:
                ax.set_xlabel("iterations")
            axes.append(ax)

    plt.subplots_adjust(wspace=0.5, hspace=0.2)

    return axes

# def plot_2pop_fitcoll_grid(fitcoll, nrows = None, ncols = 3, modelparams='data',
#                            plot_vars=None, **kwargs):
#     return plot_fitcoll_grid(fitcoll, plotdescs, nrows=nrows, ncols=ncols,
#                              modelparams=modelparams, plot_vars=plot_vars, **kwargs)
#
# def plot_4pop_fitcoll_grid(fitcoll, nrows = None, ncols = 4, modelparams='data',
#                            plot_vars=None, **kwargs):
#     return plot_fitcoll_grid(fitcoll, plotdescs, nrows=nrows, ncols=ncols,
#                              modelparams=modelparams, plot_vars=plot_vars, **kwargs)

def sim_correlation_plot(ax, ρ, xticks=None, yticks=None, ylim=None, yaxisshift=4, **kwargs):
    zorder = kwargs.pop('zorder', 2)
    keys = sorted(float(k) for k in ρ.keys() if k != 'true') # Remove 'true' key
    μvalues = np.array([ρ[key]['μ'] for key in keys])
    σvalues = np.array([ρ[key]['σ'] for key in keys])
    #marks = ax.scatter(keys, values, zorder=zorder, **kwargs)
    linestyle = kwargs.pop('linestyle', 'None')
    marker = kwargs.pop('marker', 'o')
    marks = ax.errorbar(keys, μvalues, yerr=σvalues,
                        linestyle=linestyle, marker=marker, **kwargs);
    if 'true' in ρ:
        μ = ρ['true']['μ']
        σ = ρ['true']['σ']
        ax.axhspan(μ-σ, μ+σ,
                    zorder=-2, facecolor='#EEEEEE');
        ax.axhline(μ, c='#999999', zorder=-2, linestyle='--');
        #ax.axhline(ρ['true'], color='gray', linestyle='dashed', zorder=-2);
    #ax.set_title("Correlation between inferred model and data");
    ax.set_xlabel("$L$ (s)")
    ax.set_ylabel("$\\rho$")

    if xticks is None:
        xticks = list(keys)
    if yticks is None:
        yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
    if ylim is None:
        ylim = (0, 1.1)
    ax.set_xlim(0, max(keys)*1.05)
    ax.set_ylim(ylim)
    yticks = [y for y in yticks if ylim[0] <= y <= ylim[-1]]
    ax.set_xticks(xticks);
    ax.set_yticks(yticks);
    ax.spines['left'].set_bounds(max(0, yticks[0]), min(1, yticks[-1]))
    ax.spines['left'].set_position(('outward', yaxisshift))

    return marks

def sim_rms_plot(ax, rms, xticks=None, yticks=None, ylim=None, yaxisshift=4, **kwargs):
    zorder = kwargs.pop('zorder', 2)
    keys = sorted(float(k) for k in rms.keys() if k != 'true') # Remove 'true' key
    # values = [rms[key] for key in keys]
    μvalues = np.array([rms[key]['μ'] for key in keys])
    σvalues = np.array([rms[key]['σ'] for key in keys])
    #marks = ax.scatter(keys, values, zorder=zorder, **kwargs)
    linestyle = kwargs.pop('linestyle', 'None')
    marker = kwargs.pop('marker', 'o')
    marks = ax.errorbar(keys, μvalues, yerr=σvalues,
                        linestyle=linestyle, marker=marker, **kwargs);
    if 'true' in rms:
        μ = rms['true']['μ']
        σ = rms['true']['σ']
        ax.axhspan(μ-σ, μ+σ,
                    zorder=-2, facecolor='#EEEEEE');
        ax.axhline(μ, c='#999999', zorder=-2, linestyle='--');
        # ax.axhline(rms['true'], color='gray', linestyle='dashed', zorder=-2);
    #ax.set_title("RMS error between inferred model and data");
    ax.set_xlabel("$L$ (s)")
    ax.set_ylabel("RMSE")

    ax.set_xlim(0, max(keys)*1.05)
    if ylim is None:
        ylim = (0, max(μvalues+σvalues)*1.05)
        ax.set_ylim(ylim)
    else:
        ax.set_ylim(ylim)
    if xticks is None:
        xticks = list(keys)
    #if yticks is None:
    #    yticks = [0, 1, 2, 3]
    ax.set_xticks(xticks);
    if yticks is not None:
        yticks = [y for y in yticks if ylim[0] <= y <= ylim[-1]]
        ax.set_yticks(yticks);
    else:
        yticks = ax.get_yticks()
    ax.spines['left'].set_bounds(max(0, yticks[0]), yticks[-1])
    ax.spines['left'].set_position(('outward', yaxisshift))

    return marks

def sim_compare_plot(ax, test_sims, datalens, seed=None, yoffset=45, xlim=(10, 11), ylim=(0, 40), linewidth=1.2):
    """
    seed: int
        If None, take min(test_sims.seeds)
    """

    nrows = len(datalens)
    truelabel = "True – meso"
    if seed is None:
        seed = min(test_sims.seeds)
    if not all(seed in traces for traces in test_sims.traces.values()):
        raise ValueError("Choose a simulation seed used for all models, "
                         "including the reference.")

    lines = deque()
    labels = deque()
    i = nrows - 1
    for L in (L for L in sorted(test_sims.traces.keys()) if L in datalens):
        trace = test_sims.traces[L][seed]
        # test_sims.seed: same seed as was used for traces in test_sims.traces
        trueline = ax.plot(test_sims.time,
                           test_sims.Atrue[seed][:,0] + i*yoffset,
                           label=truelabel,
                           linewidth=linewidth, alpha=0.2, color='#111111')
        lines.extend(ax.plot(test_sims.time, trace[:,0] + i*yoffset, label="$L$ = {}s".format(L),
                     linewidth=linewidth, alpha=0.8))
        labels.append("$L$ = {}s".format(L))
        i -= 1
        #truelabel = '_'  # Stop adding true trace to legend after first iteration

    lines.extend(trueline)
    labels.append(truelabel)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim[0], (nrows-1)*yoffset + ylim[1])

    ml.plot.draw_xscale(0.2, "0.2s", offset=0.4)
    ml.plot.draw_yscale(yoffset, "{} Hz$/$neuron".format(yoffset))

    ax.legend(lines, labels, loc='upper right', bbox_to_anchor=(1,0.9),
              framealpha=.9, ncol=2, edgecolor=transparent)

# Fig 4 plotting functions

def scatter_fitresults(ax, fitcolls, xvar, yvar, modelparams=None,
                       xtarget=None, ytarget=None,
                       reltols = [0.05, .1], legend_kwargs=None,
                       relcolor='grey', relalpha=None, relcontrast=0.6):
    """
    At least one of `modelparams` or `xtarget` & `ytarget` should be provided.
    relalpha overrides alpha in relcolor if provided; default alpha: 0.8
    relcontrast: ∈ [0, 1]. Opacity difference between edge and fill colors of
        relative error box. 1: fill is transparent. 0: no difference (=>no edge)
    """
    relcolor = mpl.colors.to_rgba(relcolor)
    if relalpha is not None:
        pass
    elif relcolor[3] == 0:
        relalpha = 0.8
    else:
        relalpha = relcolor[3]
    α = relalpha/len(reltols)
    relcolor = (*relcolor[:3], α)
    relec = relcolor;
    relfc = (*relcolor[:3], (1-relcontrast)*α)

    if isinstance(xvar, (tuple, list)):
        xidx = xvar[1]
        xvar = xvar[0]
    else:
        xidx = 0
    if isinstance(yvar, (tuple, list)):
        yidx = yvar[1]
        yvar = yvar[0]
    else:
        yidx = 0
    if modelparams is None:
        modelparams = ml.parameters.params_to_arrays(
            next(iter(fitcolls.values())).reffit.parameters.posterior.data.params.model)
    if xtarget is None: xtarget = modelparams[xvar].flat[xidx]
    if ytarget is None: ytarget = modelparams[yvar].flat[yidx]
    for i, (L, fitcoll) in enumerate(fitcolls.items()):
        if len(fitcoll) > 0:
            xarr = [fit.result[xvar][xidx] for fit in fitcoll.fits]
            yarr = [fit.result[yvar][yidx] for fit in fitcoll.fits]
            ax.scatter(xarr, yarr, s=8, label='_', color=colours_light[i], zorder=0, alpha=0.35)
            x = fitcoll.result[xvar][xidx]
            y = fitcoll.result[yvar][yidx]
            ax.scatter(x, y, s=8, label="$L$ = {}s".format(L), color=colours[i], zorder=1)
            ax.set_xlabel("${}$".format(xvar))
            ax.set_ylabel("${}$".format(yvar))

    # Draw target
    ax.scatter(xtarget, ytarget, marker='*', s=100, color='#F04800', label="true")
    for reltol in reltols:
        w, h = reltol*xtarget, reltol*ytarget
        rect = mpl.patches.Rectangle((xtarget-w/2, ytarget-h/2), w, h,
                                     ec=relec, fc=relfc,
                                     linestyle='dashed',
                                     zorder=-2)
        ax.add_patch(rect)
    # Formatting
    ml.plot.detach_spines(ax)
    if legend_kwargs is None: legend_kwargs = {}
    if 'loc' not in legend_kwargs: legend_kwargs['loc'] = 'best'
    if 'borderaxespad' not in legend_kwargs: legend_kwargs['borderaxespad'] = 1
    ax.legend(**legend_kwargs)

def fitstats(η, datalens, test_sims, param_subsets):
    if isinstance(η, Iterable) and not isinstance(η, str):
        return pd.concat((fitstats(_η, datalens, test_sims, param_subsets) for _η in η),
                         keys=η, names=['subset'])

    fitcolls = test_sims[η].fitcolls
    varindex = pd.MultiIndex.from_tuples(
            [(varname, idx) for varname, idxlist in param_subsets[η].items() for idx in idxlist],
            names = ['name', 'idx'])
    modelparams = ml.parameters.params_to_arrays(
                next(iter(fitcolls.values())).reffit.parameters.posterior.data.params.model)

    stats = deque()
    statindex = pd.Index(['res', 'μ', 'σ', 'CV', 'Δrel'], name='stat')
    Lindex = pd.Index([L for L in datalens if len(fitcolls[L]) > 0], name='L')
    target = pd.Series([modelparams[varname].flat[idx] for varname, idx in varindex],
                       index=varindex)
    for L in Lindex:
        fitcoll = fitcolls[L]
        results = pd.DataFrame([[fit.result[varname][idx] for varname, idx in varindex] for fit in fitcoll.fits],
                                 columns=varindex, index=range(len(fitcoll.fits)))
        fitresult = pd.Series([fitcoll.result[varname][idx] for varname, idx in varindex],
                               index=varindex)
        μ = results.mean()
        σ = results.std()
        CV = abs(σ / μ * 100)
        Δrel = abs((fitresult - target) / target)
        collstats = pd.DataFrame([fitresult, μ, σ, CV, Δrel], index=statindex)
        stats.append(collstats.stack().stack().reorder_levels(['stat', 'name', 'idx']))

    return pd.concat(stats, axis='columns', keys=Lindex)

def get_fitstat_plotdata(η, datalens, test_sims, param_subsets, rows, statlbls):
    stats = fitstats(η, datalens, test_sims, param_subsets)
    plotdata = None
    for η in rows.keys():
        df = stats.loc[[(η, stat) + idx for idx in rows[η] for stat in statlbls]]
        if plotdata is None:
            plotdata = df
        else:
            plotdata = plotdata.append(df)

    return plotdata

def plot_fitstats(η, datalens, test_sims, param_subsets, rows, statlbls,
                  ax=None, legend_kwargs=None, legend2_kwargs=None):
    if ax is None: ax = plt.gca()
    if legend_kwargs is None: legend_kwargs={}
    if legend2_kwargs is None: legend2_kwargs={}
    plotstats = get_fitstat_plotdata(η, datalens, test_sims, param_subsets, rows, statlbls)
    stats = {}
    if 'CV' in statlbls:
        stats['CV'] = plotstats.loc[(slice(None),'CV'), :]
    if 'Δrel' in statlbls:
        stats['Δ'] = plotstats.loc[(slice(None),'Δrel'), :]
    varidcs = zip( *(next(iter(stats.values()))
                     .index.droplevel('stat').get_level_values(i)
                     for i in range(3)) )
    legend_labels = ['{} (${}_{}$)'
                     .format(test_sims[_η].get_varstring(name, idx), *_η)
                     for _η, name, idx in varidcs]

    def get_df_title(lbl):
        if lbl == 'CV':
            df = stats['CV'].T
            title = "$|CV|$"
            label = "$|CV|$ (%)"
        else:
            df = stats['Δ'].T
            title = "$\\Delta_{rel}$"
            label = "$\\Delta_{rel}$"
        return df, title, label

    default_legend_kwargs = {'loc':'lower left', 'bbox_to_anchor':(0,1),
                             'ncol': 2}
    legend_kwargs = {**default_legend_kwargs, **legend_kwargs}
    with plt.style.context({'legend.borderaxespad': 0.5,
                            }):

        df, title, label = get_df_title(statlbls[0])
        if len(statlbls) == 1: title = None
        df.columns = df.columns.droplevel('stat')
        df.plot(ax=ax)
        ax.legend(legend_labels, title=title, **legend_kwargs)
        #ax.set_yticks([0, 3, 6, 9, 12])
        ax.set_ylabel(label)
        ax.set_xlabel("$L$ (s)")

        ml.plot.detach_spines(ax)

        if len(statlbls) > 1:
            ax2 = ax.twinx()
            ax2.spines['left'].set_visible(False)
            ax2.spines['right'].set_visible(True)
            df, title, label = get_df_title(statlbls[1])
            df.columns = df.columns.droplevel('stat')
            df.plot(ax=ax2, linestyle='dashed')
            default_legend2_kwargs = {'loc': 'lower right',
                                      'bbox_to_anchor': (1,1)}
            legend2_kwargs = {**default_legend_kwargs,
                              **default_legend2_kwargs,
                              **legend2_kwargs}
            ax2.legend(legend_labels, title=title, **legend2_kwargs)
            ax2.set_ylabel('$\\Delta_{rel}$')

            ml.plot.detach_spines(ax2)

        else:
            ax2 = None

        #ax.get_figure().subplots_adjust(top=0.8)

    return ax, ax2

def get_μσ(hist, window=0.1, μ=None, σ=None, μ_fn=None, σ_fn=None):
    if μ is None: μ = {}
    if σ is None: σ = {}
    if μ_fn is None:
        μ_fn = 'mean'
        #def μ_fn(W): return np.mean(W, axis=0)
    if hist.name not in μ:
        _μ = anlz.filter(hist, μ_fn, window=window).align_to(hist)
        #_μ = anlz.subsample(hist, target_dt=window, aggregation=μ_fn)
        μ[hist.name] = _μ
    else:
        _μ = μ[hist.name]
    _hist = hist.align_to(_μ)  # Only for truncation
    if σ_fn is None:
        def σ_fn(W): return np.std(W, ddof=1, axis=0)
    if hist.name not in σ:
        _σ = anlz.filter(_hist-_μ, σ_fn, window=window)
        #_σ = anlz.subsample(_hist-_μ, target_dt=window, aggregation=σ_fn,
        #                    warn=False)
        σ[hist.name] = _σ
    else:
        _σ = σ[hist.name]
    #assert np.all(sinn.isclose(_σ.time, _μ.time))
    #_σ will have a shorter time array than _μ because of second filter
    _μ = _μ.align_to(_σ)

    return _μ, _σ

def plot_step_sim_μσ(hist, refhist=None, window=0.1, ax=None, ylim=(0,40),
                     μ=None, σ=None, μ_fn=None, σ_fn=None,
                     maxstops=200):
    """
    `refhist` is used  calculate RMSE and correlation.
    `μ`, `σ` can be used to pass cache dictionaries. Useful if the statistics
    are costly to compute, so that we don't compute them every time we redraw
    the plot.
    `μ_fn`, `σ_fn` are the functions used to compute the drawn line and the
    half-width of the filled area around that line respectively. They are
    passed to `sinn.analyze.subsample` as the `aggregation`, and can take any
    callable or string that parameter accepts.
    `maxstops` is the maximum number of points to be plotted. Limiting this
    number is especially important when exporting pgf plots, to avoid exceeding
    TeX memory. Step size is computed as `len(hist)/maxstops`, so for histories
    that exceed `maxstops` by less than a factor of two, no decimation actually
    happens.
    """
    if ax is None: ax = plt.gca()
    _μ, _σ = get_μσ(hist, window, μ, σ, μ_fn, σ_fn)

    # Decimate histories so they don't have too many points to plot
    step = max(int(len(_σ) / maxstops), 1)
    _σ = anlz.decimate(_σ, step)
    _μ = anlz.decimate(_μ, step)

    idcs = [(slice(None),) + idx for idx in np.ndindex(_μ.shape)]
        # Each index in idcs yieds all time points for one component
    for i, idx in enumerate(idcs):
        ax.fill_between(_μ.time, _μ.trace[idx]-_σ.trace[idx], _μ.trace[idx]+_σ.trace[idx],
                        color=colours_light[i], zorder=0, alpha=0.5)
    for i, idx in enumerate(idcs):
        ax.plot(_μ.time, _μ.trace[idx], color=colours[i], zorder=1)
    ax.set_ylim(ylim)

    ml.plot.draw_xscale(2, "2 s", xshift=4)
    ml.plot.draw_yscale(20, "20 Hz$/$neuron", yshift=-4)

    # Compute RMS and correlation
    if refhist is not None:
        # HACK: align both ways to make sure that we used time steps from
        # refhist but also truncate refhist to the range of hist
        _hist = hist.align_to(refhist)
        reftrace = refhist.align_to(_hist).trace
        histtrace = _hist.trace
        _rms = rms(histtrace, reftrace).mean()
        ρ = np.mean([c[0] for c in corr(histtrace, reftrace)])
        text_Δ = 1 * ml.plot.inches_to_y(mpl.rcParams['font.size'] / 72, ax=ax)
            # Font size is in points (1/72 inch)
        with mpl.style.context({'font.weight': 'normal'}):
            ax.text(17, ylim[1]-0.5*text_Δ, "RMSE: {:.2f}".format(_rms), ha='right', va='top')
            ax.text(17, ylim[1]-1.5*text_Δ, "ρ: {:.2f}".format(ρ), ha='right', va='top')

def subset_string(subset):
    return "${}_{}$".format(subset[0], subset[1])

def fitstat_table(stat, subsets, datalens, test_sims, param_subsets,
                  latex=True):
    df = pd.concat([fitstats(η, datalens, test_sims, param_subsets)
                     .loc[['Δrel', 'CV'], slice(None)]
                    for η in subsets],
                   keys=subsets, names = ['subset'])
    subdf = df.loc[(slice(None), stat), slice(None)]
    subdf.index = subdf.index.droplevel(1)
    index = subdf.index
    i = subdf.index.names.index('subset'); ηs = subdf.index.levels[i][subdf.index.labels[i]]
    i = subdf.index.names.index('name'); varnames = subdf.index.levels[i][subdf.index.labels[i]]
    i = subdf.index.names.index('idx'); varidcs = subdf.index.levels[i][subdf.index.labels[i]]

    subdf.index = pd.MultiIndex.from_tuples(
        [(subset_string(η), test_sims[η].get_varstring(varname, varidx))
         for η, varname, varidx in zip(ηs, varnames, varidcs)],
        names=['Subset', 'Parameter'])

    subdf = subdf.sort_index()

    if latex:
        float_format = {'CV': '{:.2f}', 'Δrel': '{:.3f}'}.get(stat).format
        latex = subdf.to_latex(float_format=float_format, escape=False)
        #latex = subdf.to_latex()
        latex = latex.replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule', '\\hline')
        return latex
    else:
        return subdf

class HeteroParams:
    # TODO: Include plotting functions in ParameterSetSampler
    def __init__(self, pset):
        self.hetero_idcs = {}
        self.popsizes = None
        for name, val in pset.items():
            if isinstance(val, ParameterSet):
                comps = []
                for s in val.shape:
                    if s == 1:
                        comps.append(0)
                    elif isinstance(s, int):
                        comps.append(slice(None))
                    else:
                        assert(isinstance(s, str))
                        popsizes = tuple(int(c) for c in s.split('+'))
                        if self.popsizes is not None:
                            assert(self.popsizes == popsizes)
                        else:
                            self.popsizes = popsizes
                            self.npops = len(self.popsizes)
                            self.popslices = []
                            k = 0
                            for n in popsizes:
                                self.popslices.append(slice(k, k+n))
                                k += n
                        comps.append(self.popslices)
                comps = [c if isinstance(c, list) else [c]*self.npops
                         for c in comps]
                self.hetero_idcs[name] = [idx for idx in zip(*comps)]
        self.sampler = ml.parameters.ParameterSetSampler(pset)
        self.params = self.sampler.sample()
    def hist(self, varidx, bins=20, ax=None, transform=None, **kwargs):
        """
        `varidx` : (name, idx) pair
        """
        if ax is None:
            ax = plt.gca()
        name, i = varidx
        idx = self.hetero_idcs[name][i]
        data = self.params[name][idx]
        weights = np.ones(len(data))/len(data)
        #plt.hist(np.log10(data), bins=bins, weights=weights, alpha=1 - 0.5*(i+1)/npops);
        histkwargs = {'alpha': 0.5}
        histkwargs.update(kwargs)
        if transform is not None:
            data = transform(data)
        ax.hist(data, bins=bins, weights=weights, **histkwargs);

# ========================
# Posterior plots

def plot_marginals(marginals, cols, upper_rows=None, lower_rows=None,
                   gridspec=None, stddevs=(1.,)):
    ncols = nrows = len(cols)
    if upper_rows is None: upper_rows = []
    if lower_rows is None: lower_rows = []
    assert(max(len(upper_rows), len(lower_rows)) == nrows - 1)
    Δ = nrows - 1 - len(upper_rows)
    if Δ > 0:
        upper_rows.extend([None]*Δ)
    Δ = nrows - 1 - len(lower_rows)
    if Δ > 0:
        lower_rows.extend([None]*Δ)
    assert(len(upper_rows) == len(lower_rows) == nrows - 1)
    keys = np.zeros((nrows, ncols), dtype=np.object)
    # Following assumes diag going from top left to bottom right
    for i, nm in enumerate(cols):
        keys[:,i] = [(nm2, nm) for nm2 in upper_rows[:i]] + [(nm,)] + [(nm2, nm) for nm2 in lower_rows[i:]]

    if gridspec is None:
        gs = GridSpec(nrows, ncols)
    else:
        gs = mpl.gridspec.GridSpecFromSubplotSpec(nrows, ncols,
                                                  subplot_spec=gridspec)

    # Create an object to store the axes. This is returned on function exit
    axes = np.zeros(keys.shape, dtype=np.object)
    # Loop over the axes actually plot them
    for (i,j), key in np.ndenumerate(keys):
        if None in key:
            # No parameter provided for this position
            continue
        ax = plt.subplot(gs[i,j])
        axes[i,j] = ax
        if len(key) == 1:
            key = key[0]
            marginals.plot_marginal1D(key, ax=ax)
            tick_color = ml.stylelib.colorschemes.cmaps[marginals.marginals1D[key].cmap].white
            ax.xaxis.set_tick_params(direction='in', color=tick_color)
        elif len(key) == 2:
            marginals.plot_marginal2D(
                *key, ax=ax, colorbar=False,
                stddevs=stddevs)

        if i == 0:
            ax.set_xlabel(cols[j])
            ax.xaxis.set_label_position('top')
        elif i == nrows-1:
            ax.set_xlabel(cols[j])
            ax.xaxis.set_label_position('bottom')
        else:
            ax.set_xlabel("")
        if j == 0:
            ylabel = cols[i] if i == j else lower_rows[i-1]  # Depends on diag direction
            ax.set_ylabel(ylabel)
            ax.yaxis.set_label_position('left')
        elif j == nrows-1:
            ylabel = cols[i] if i == j else upper_rows[i]
            ax.set_ylabel(ylabel)
            ax.yaxis.set_label_position('right')
        else:
            ax.set_ylabel("")

        # First hide all spines, than show those we want
        for spine in ax.spines.values():
            spine.set_visible(False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        if i == j == 0:
            ax.xaxis.set_visible(True)
            ax.xaxis.set_tick_params(bottom=False, labelbottom=False)
        elif i == 0 and j > 0:
            ax.spines['top'].set_visible(True)
            ax.xaxis.set_visible(True)
            ax.xaxis.set_tick_params(bottom=False, top=True, labelbottom=False, labeltop=False)
        elif i == nrows-1:
            ax.spines['bottom'].set_visible(True)
            ax.xaxis.set_visible(True)
            ax.xaxis.set_tick_params(bottom=True, top=False)
        if j == 0 and i > 0:
            ax.spines['left'].set_visible(True)
            ax.yaxis.set_visible(True)
            ax.yaxis.set_tick_params(left=True, right=False)
        elif j == nrows-1 and i != j:
            ax.spines['right'].set_visible(True)
            ax.yaxis.set_visible(True)
            ax.yaxis.set_tick_params(left=False, right=True, labelleft=False, labelright=True)

    ml.plot.subplots_adjust_margins(spacing=0.07,
                                    top=0.25, bottom=0.35,
                                    left=0.5, right=0.5)

    return axes

# =============================
# Aligning ylabels
# Backported from newer matplotlib: https://github.com/matplotlib/matplotlib/blob/ea4274ea0dc4070cdfa268e54b802afd546c5f90/lib/matplotlib/figure.py
# Use as `lib.align_labels(fig)` rather than `fig.align_labels()`

def get_rows_columns(self):
    """
    Copied from newer GridSpec: https://github.com/matplotlib/matplotlib/blob/0536a669742e88e56a9e080580dbba727c9f33cd/lib/matplotlib/gridspec.py
    Return the subplot row and column numbers as a tuple
    ``(n_rows, n_cols, row_start, row_stop, col_start, col_stop)``.
    """
    assert isinstance(self, mpl.gridspec.SubplotSpec)
    gridspec = self.get_gridspec()
    nrows, ncols = gridspec.get_geometry()
    row_start, col_start = divmod(self.num1, ncols)
    row_stop, col_stop = divmod(self.num2, ncols)
    return nrows, ncols, row_start, row_stop, col_start, col_stop

def align_xlabels(self, axs=None):
    """
    Align the ylabels of subplots in the same subplot column if label
    alignment is being done automatically (i.e. the label position is
    not manually set).

    Alignment persists for draw events after this is called.

    If a label is on the bottom, it is aligned with labels on axes that
    also have their label on the bottom and that have the same
    bottom-most subplot row.  If the label is on the top,
    it is aligned with labels on axes with the same top-most row.

    Parameters
    ----------
    axs : list of `~matplotlib.axes.Axes`
        Optional list of (or ndarray) `~matplotlib.axes.Axes`
        to align the xlabels.
        Default is to align all axes on the figure.

    See Also
    --------
    matplotlib.figure.Figure.align_ylabels

    matplotlib.figure.Figure.align_labels

    Notes
    -----
    This assumes that ``axs`` are from the same `.GridSpec`, so that
    their `.SubplotSpec` positions correspond to figure positions.

    Examples
    --------
    Example with rotated xtick labels::

        fig, axs = plt.subplots(1, 2)
        for tick in axs[0].get_xticklabels():
            tick.set_rotation(55)
        axs[0].set_xlabel('XLabel 0')
        axs[1].set_xlabel('XLabel 1')
        fig.align_xlabels()

    """
    from matplotlib.cbook import Grouper

    if axs is None:
        axs = self.axes
    axs = np.asarray(axs).ravel()
    for ax in axs:
        logger.debug(' Working on: %s', ax.get_xlabel())
        ss = ax.get_subplotspec()
        nrows, ncols, row0, row1, col0, col1 = get_rows_columns(ss)
        labpo = ax.xaxis.get_label_position()  # top or bottom

        # loop through other axes, and search for label positions
        # that are same as this one, and that share the appropriate
        # row number.
        #  Add to a grouper associated with each axes of sibblings.
        # This list is inspected in `axis.draw` by
        # `axis._update_label_position`.
        for axc in axs:
            if axc.xaxis.get_label_position() == labpo:
                ss = axc.get_subplotspec()
                nrows, ncols, rowc0, rowc1, colc, col1 = \
                        get_rows_columns(ss)
                if (labpo == 'bottom' and rowc1 == row1 or
                    labpo == 'top' and rowc0 == row0):
                    # grouper for groups of xlabels to align
                    if not hasattr(self, '_align_xlabel_grp'):
                        self._align_xlabel_grp = Grouper()
                    self._align_xlabel_grp.join(ax, axc)

def align_ylabels(self, axs=None):
    """
    Align the ylabels of subplots in the same subplot column if label
    alignment is being done automatically (i.e. the label position is
    not manually set).

    Alignment persists for draw events after this is called.

    If a label is on the left, it is aligned with labels on axes that
    also have their label on the left and that have the same
    left-most subplot column.  If the label is on the right,
    it is aligned with labels on axes with the same right-most column.

    Parameters
    ----------
    axs : list of `~matplotlib.axes.Axes`
        Optional list (or ndarray) of `~matplotlib.axes.Axes`
        to align the ylabels.
        Default is to align all axes on the figure.

    See Also
    --------
    matplotlib.figure.Figure.align_xlabels

    matplotlib.figure.Figure.align_labels

    Notes
    -----
    This assumes that ``axs`` are from the same `.GridSpec`, so that
    their `.SubplotSpec` positions correspond to figure positions.

    Examples
    --------
    Example with large yticks labels::

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(np.arange(0, 1000, 50))
        axs[0].set_ylabel('YLabel 0')
        axs[1].set_ylabel('YLabel 1')
        fig.align_ylabels()

    """
    from matplotlib.cbook import Grouper

    if axs is None:
        axs = self.axes
    axs = np.asarray(axs).ravel()
    for ax in axs:
        logger.debug(' Working on: %s', ax.get_ylabel())
        ss = ax.get_subplotspec()
        nrows, ncols, row0, row1, col0, col1 = get_rows_columns(ss)
        labpo = ax.yaxis.get_label_position()  # left or right
        # loop through other axes, and search for label positions
        # that are same as this one, and that share the appropriate
        # column number.
        # Add to a list associated with each axes of sibblings.
        # This list is inspected in `axis.draw` by
        # `axis._update_label_position`.
        for axc in axs:
            if axc != ax:
                if axc.yaxis.get_label_position() == labpo:
                    ss = axc.get_subplotspec()
                    nrows, ncols, row0, row1, colc0, colc1 = \
                            get_rows_columns(ss)
                    if (labpo == 'left' and colc0 == col0 or
                        labpo == 'right' and colc1 == col1):
                        # grouper for groups of ylabels to align
                        if not hasattr(self, '_align_ylabel_grp'):
                            self._align_ylabel_grp = Grouper()
                        self._align_ylabel_grp.join(ax, axc)

def align_labels(self, axs=None):
    """
    Align the xlabels and ylabels of subplots with the same subplots
    row or column (respectively) if label alignment is being
    done automatically (i.e. the label position is not manually set).

    Alignment persists for draw events after this is called.

    Parameters
    ----------
    axs : list of `~matplotlib.axes.Axes`
        Optional list (or ndarray) of `~matplotlib.axes.Axes`
        to align the labels.
        Default is to align all axes on the figure.

    See Also
    --------
    matplotlib.figure.Figure.align_xlabels

    matplotlib.figure.Figure.align_ylabels
    """
    align_xlabels(self, axs=axs)
    align_ylabels(self, axs=axs)
    if axs is None:
        axs = self.axes
    for ax in axs:
        # import pdb; pdb.set_trace()
        _update_xlabel_position(ax.xaxis, self.canvas.renderer)
        _update_ylabel_position(ax.yaxis, self.canvas.renderer)


def _get_xtick_boxes_siblings(self, renderer):
    """
    Get the bounding boxes for this `.axis` and its siblings
    as set by `.Figure.align_xlabels` or  `.Figure.align_ylablels`.
    By default it just gets bboxes for self.
    """
    bboxes = []
    bboxes2 = []
    # get the Grouper that keeps track of x-label groups for this figure
    grp = self.figure._align_xlabel_grp
    # if we want to align labels from other axes:
    for nn, axx in enumerate(grp.get_siblings(self.axes)):
        ticks_to_draw = axx.xaxis._update_ticks(renderer)
        tlb, tlb2 = axx.xaxis._get_tick_bboxes(ticks_to_draw, renderer)
        bboxes.extend(tlb)
        bboxes2.extend(tlb2)
    return bboxes, bboxes2

def _update_xlabel_position(self, renderer):
    """
    Update the label position based on the bounding box enclosing
    all the ticklabels and axis spine
    """
    import matplotlib.transforms as mtransforms

    if not self._autolabelpos:
        return

    # get bounding boxes for this axis and any siblings
    # that have been set by `fig.align_xlabels()`
    bboxes, bboxes2 = _get_xtick_boxes_siblings(self, renderer=renderer)

    x, y = self.label.get_position()
    if self.label_position == 'bottom':
        try:
            spine = self.axes.spines['bottom']
            spinebbox = spine.get_transform().transform_path(
                spine.get_path()).get_extents()
        except KeyError:
            # use axes if spine doesn't exist
            spinebbox = self.axes.bbox
        bbox = mtransforms.Bbox.union(bboxes + [spinebbox])
        bottom = bbox.y0

        self.label.set_position(
            (x, bottom - self.labelpad * self.figure.dpi / 72)
        )

    else:
        try:
            spine = self.axes.spines['top']
            spinebbox = spine.get_transform().transform_path(
                spine.get_path()).get_extents()
        except KeyError:
            # use axes if spine doesn't exist
            spinebbox = self.axes.bbox
        bbox = mtransforms.Bbox.union(bboxes2 + [spinebbox])
        top = bbox.y1

        self.label.set_position(
            (x, top + self.labelpad * self.figure.dpi / 72)
)

def _get_ytick_boxes_siblings(self, renderer):
    """
    Get the bounding boxes for this `.axis` and its siblings
    as set by `.Figure.align_xlabels` or  `.Figure.align_ylablels`.
    By default it just gets bboxes for self.
    """
    bboxes = []
    bboxes2 = []
    # get the Grouper that keeps track of y-label groups for this figure
    grp = self.figure._align_ylabel_grp
    # if we want to align labels from other axes:
    for axx in grp.get_siblings(self.axes):
        ticks_to_draw = axx.yaxis._update_ticks(renderer)
        tlb, tlb2 = axx.yaxis._get_tick_bboxes(ticks_to_draw, renderer)
        bboxes.extend(tlb)
        bboxes2.extend(tlb2)
    return bboxes, bboxes2

def _update_ylabel_position(self, renderer):
    """
    Update the label position based on the bounding box enclosing
    all the ticklabels and axis spine
    """
    import matplotlib.transforms as mtransforms

    if not self._autolabelpos:
        return

    # get bounding boxes for this axis and any siblings
    # that have been set by `fig.align_ylabels()`
    bboxes, bboxes2 = _get_ytick_boxes_siblings(self, renderer=renderer)

    x, y = self.label.get_position()
    if self.label_position == 'left':
        try:
            spine = self.axes.spines['left']
            spinebbox = spine.get_transform().transform_path(
                spine.get_path()).get_extents()
        except KeyError:
            # use axes if spine doesn't exist
            spinebbox = self.axes.bbox
        bbox = mtransforms.Bbox.union(bboxes + [spinebbox])
        left = bbox.x0
        self.label.set_position(
            (left - self.labelpad * self.figure.dpi / 72, y)
        )

    else:
        try:
            spine = self.axes.spines['right']
            spinebbox = spine.get_transform().transform_path(
                spine.get_path()).get_extents()
        except KeyError:
            # use axes if spine doesn't exist
            spinebbox = self.axes.bbox
        bbox = mtransforms.Bbox.union(bboxes2 + [spinebbox])
        right = bbox.x1

        self.label.set_position(
            (right + self.labelpad * self.figure.dpi / 72, y)
        )

# ========================================================
# Backport improvements to `ml.plot` after version freeze

def draw_xscale(length, label, ax=None, offset=0.05, scalelinewidth=2, color=None, xshift=0, yshift=0, **kwargs):
    from mackelab.plot import get_display_bbox
    """
    offset in inches
    **kwargs passed on to ax.xaxis.set_label_text
    """
    if ax is None:
        ax = plt.gca()
    if color is None:
        color = mpl.rcParams['axes.edgecolor']
    fig = ax.get_figure()
    fontsize = plt.rcParams['font.size']
    dpi = fig.dpi
    ax.set_xticks([])  # Remove ticks

    x0, xn = ax.get_xlim()
    y0, yn = ax.get_ylim()
    xmargin, ymargin = ax.margins()
    bbox = get_display_bbox(ax)
    dwidth = bbox.width
    dheight = bbox.height
    xwidth = xn - x0
    yheight = yn - y0
    # Convert xshift, yshift into data coords
    data_xshift = xshift * xwidth/dwidth
    data_yshift = yshift * yheight/dheight
    data_offset = offset * yheight/dheight
    data_linewidth = scalelinewidth * yheight/dheight


    spine = ax.spines['bottom']
    spine.set_visible(True)
    x = x0 + data_xshift
    y = y0 - offset - data_yshift - data_linewidth
    spine.set_bounds(x, x+length)
    spine.set_linewidth(scalelinewidth)
    spine.set_position(('data', y))
    spine.set_color(color)

    #y -= fontsize/dpi * yheight/dheight
    data_fontheight = fontsize * yheight/dheight  # FIXME: too small
    y -= data_linewidth + 0.3*data_fontheight
    ax.xaxis.set_label_coords(x, y, transform=ax.transData)
    ax.xaxis.set_label_text(label, color=color, horizontalalignment='left', verticalalignment='top', **kwargs)

ml.plot.draw_xscale = draw_xscale

def full_extent(ax, pad=0.0):
    """Get the full extent of an axes, including axes labels, tick labels, and
    titles."""
    # https://stackoverflow.com/a/14720600
    # For text objects, we need to draw the figure first, otherwise the extents
    # are undefined.
    fig = ax.get_figure()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    items = ax.get_xticklabels() + ax.get_yticklabels()
    items += [ax]
    for text in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        if text.get_text() != "":
            items += [text]
    bbox = mpl.transforms.Bbox.union([item.get_window_extent(renderer) for item in items])
    return bbox.expanded(1.0 + pad, 1.0 + pad)

def set_full_axes_background(ax, color, zorder=-10):
    # https://stackoverflow.com/a/14720600

    fig = ax.get_figure()

    axes = [ax] if not isinstance(ax, Iterable) else ax
    #extent = mpl.transforms.Bbox.union([full_extent(ax) for ax in axes])
    extent = mpl.transforms.Bbox.union([full_extent(ax) for ax in axes])

    # It's best to transform this back into figure coordinates. Otherwise, it won't
    # behave correctly when the size of the plot is changed.
    extent = extent.transformed(fig.transFigure.inverted())

    # We can now make the rectangle in figure coords using the "transform" kwarg.
    rect = mpl.patches.Rectangle([extent.xmin, extent.ymin], extent.width, extent.height,
                                  facecolor=color, edgecolor='none', zorder=zorder,
                                  transform=fig.transFigure)
    fig.patches.append(rect)

def _apply_op(self, op, b=None):
    if b is None:
        new_series = histories.Series(self)
        new_series.set_update_function(
            lambda t: op(self[t]),
            inputs = self)
        new_series.set_range_update_function(
            lambda tarr: op(self[self.time_array_to_slice(tarr)]))
        # new_series.add_input(self)
    elif isinstance(b, histories.HistoryBase):
        # HACK Should write function that doesn't create empty arrays
        shape = np.broadcast(np.empty(self.shape), np.empty(b.shape)).shape
        tnidx = min(self.tnidx, b.get_tidx_for(b.tnidx, self))
        new_series = histories.Series(self, shape=shape,
                            time_array=self._tarr[:tnidx+1])
        new_series.set_update_function(
            lambda t: op(self[t], b[t]),
            inputs = [self, b])
        new_series.set_range_update_function(
            lambda tarr: op(self[self.time_array_to_slice(tarr)],
                            b[b.time_array_to_slice(tarr)]))
        #new_series.add_input(self)
        computable_tidx = min(
            self.get_tidx_for(min(self.cur_tidx, self.tnidx), new_series),
            b.get_tidx_for(min(b.cur_tidx, b.tnidx), new_series))
    else:
        if hasattr(b, 'shape'):
            shape = np.broadcast(np.empty(self.shape),
                                    np.empty(b.shape)).shape
        else:
            shape = self.shape
        new_series = histories.Series(self, shape=shape)
        new_series.set_update_function(lambda t: op(self[t], b))
        new_series.set_range_update_function(
            lambda tarr: op(self[self.time_array_to_slice(tarr)], b))
        if shim.is_theano_variable(b):
            new_series.add_input([self, b])
        else:
            new_series.add_input(self)
        computable_tidx = self.get_tidx_for(min(self.cur_tidx, self.tnidx),
                                            new_series)

    # Since we assume the op is cheap, calculate as much as possible
    # without triggering updates in the inputs
    new_series.compute_up_to(computable_tidx)

    # if ( self._original_tidx.get_value() >= self.tnidx
    #      and ( b is None
    #            or not isinstance(b, HistoryBase)
    #            or b._original_tidx >= b.tnidx ) ):
    #      # All op members are computed, so filling the result series is 1) possible and 2) cheap
    #      new_series.set()
    return new_series
histories.Series._apply_op = _apply_op

from matplotlib.patches import Rectangle
def plot_value_profile(clist, style=None):
    """
    Plot the value profile of a color map. Ensuring each colour
    has a different values helps make the colors distinguishable
    even in grey scale.
    It also helps that the value progression be monotonous, that way
    legends go from e.g. light to dark.

    Parameters
    ----------
    clist: colour list
    style: str | None
        Either 'line' or 'scatter'. If None, style is chosen automatically.
    """
    if style is None:
        style = 'line' if len(clist) >= 8 else 'scatter'
    ax = plt.subplot(111)
    if style == 'line':
        ax.plot(get_value(clist))
    else:
        ax.scatter(range(len(clist)), get_value(clist))
ml.colors.plot_value_profile = plot_value_profile

def display_palette(colors, ax=None):
    #if (isinstance(colors, (str, bytes))
    #    or not isinstance(colors, Iterable)):
    #    colors = [colors]
    rowheight = 1.5  # Height of a color row in inches
    colors = np.atleast_1d(colors)
    if ax is None:
        ax = plt.gca()
    fig = ax.get_figure()
    if colors.ndim == 2:
        # Multiple color lists
        ax.remove()
        for i, row in enumerate(colors):
            ax = plt.subplot(len(colors),1,i+1)
            show_colors(row, ax=ax)
        fig.set_figheight(len(colors)*rowheight)
    else:
        fig.set_figheight(rowheight)
        ax.set_axis_off()
        w = 1/len(colors); h = 1
        for i, c in enumerate(colors):
            rect = Rectangle((i*w,0), w, h, color=c)
            ax.add_patch(rect);
        return ax
ml.colors.display_palette = display_palette
