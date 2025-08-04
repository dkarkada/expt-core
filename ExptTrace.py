import numpy as np


class ExptTrace():

    @classmethod
    def multi_init(cls, num_init, var_names):
        return [cls(var_names) for _ in range(num_init)]

    def __init__(self, var_names):
        if not isinstance(var_names, list):
            raise ValueError("var_names must be a list")
        if "outcome" in var_names:
            raise ValueError("variable name 'outcome' disallowed")
        self.var_names = var_names.copy()
        self._config2outcome = {}
        self.outcome_shape = None

    def __setitem__(self, key, val):
        config, outcome = key, val
        # if this is the first measurement, figure out shape of measurement outcome
        if self.outcome_shape is None:
            out_array = np.asarray(outcome)
            if not np.issubdtype(out_array.dtype, np.number):
                raise ValueError("measurement outcome must be numeric")
            self.outcome_shape = out_array.shape
        # otherwise, ensure new measurement has compatible shape
        elif np.shape(outcome) != self.outcome_shape:
            raise ValueError(f"outcome shape {np.shape(outcome)} != expected {self.outcome_shape}")
        # ensure config is a tuple of the correct length
        config = (config,) if not isinstance(config, tuple) else config
        if len(config) != len(self.var_names):
            raise ValueError(f"len config {len(config)} != num vars {len(self.var_names)}")
        # ensure config settings are of valid types
        allowed_types = (int, float, str, tuple)
        if not all(isinstance(c, allowed_types) for c in config):
            raise ValueError(f"config {config} elements must be one of {allowed_types}")
        # ensure config doesn't already exist, then write measurement outcome
        if config in self._config2outcome:
            raise ValueError(f"config {config} already exists. overwriting not supported")
        self._config2outcome[config] = outcome

    def __getitem__(self, key):
        # we need to know shape of measurement outcome
        if self.outcome_shape is None:
            raise RuntimeError("must add items before getting")
        # key = tuple of indexers (ints or slices). Selects configs.
        # ensure key is a tuple of the correct length
        var_indexers = (key,) if not isinstance(key, tuple) else key
        if len(var_indexers) != len(self.var_names):
            raise ValueError(f"num config vars {len(var_indexers)} != expected {len(self.var_names)}")

        # for each indep var, get the var value selected by the key.
        # if the indexer is a (full) slice, get the full axis for that var
        config_axes = []
        for idx, var_name in enumerate(self.var_names):
            var_setting = var_indexers[idx]
            config_axis = [var_setting]
            if isinstance(var_setting, slice):
                slc = (var_setting.start, var_setting.stop, var_setting.step)
                if not all([x is None for x in slc]):
                    raise ValueError(f"slice start/stop/step not supported ({var_name})")
                config_axis = self.get_axis(var_name)
            config_axes.append(config_axis)

        # create a meshgrid of all selected configs, populate with outcomes.
        # use masked array to handle missing/unwritten outcomes.
        config_shape = [len(ax) for ax in config_axes]
        result_mesh = np.ma.masked_all(config_shape + list(self.outcome_shape))
        for mesh_idxs in np.ndindex(*config_shape):
            config = tuple(config_axes[dim][idx] for dim, idx in enumerate(mesh_idxs))
            if config in self._config2outcome.keys():
                result_mesh[mesh_idxs] = self._config2outcome[config]

        # if all results are missing, raise KeyError.
        # if the key selects a single measurement, return a squeezed array.
        # if there are no missing results, return a regular ndarray.
        if np.all(result_mesh.mask):
            raise KeyError(f"config(s) {var_indexers} is/are missing")
        if np.prod(config_shape) == 1:
            return np.array(result_mesh).squeeze()
        if not np.ma.is_masked(result_mesh):
            return np.array(result_mesh)
        return result_mesh

    def get_axis(self, var_name):
        if var_name not in self.var_names:
            raise ValueError(f"var {var_name} not found")
        var_idx = self.var_names.index(var_name)
        # iterate through written configs and collect all var settings
        axis = set()
        for config in self._config2outcome.keys():
            axis.add(config[var_idx])
        return sorted(list(axis))

    def get(self, **kwargs):
        key = self._get_config_key(_mode='get', **kwargs)
        return self[key]

    def set(self, **kwargs):
        if "outcome" not in kwargs:
            raise ValueError(f"no outcome given")
        outcome = kwargs["outcome"]
        config = self._get_config_key(_mode='set', **kwargs)
        self[config] = outcome

    def is_written(self, **kwargs):
        config = self._get_config_key(_mode='set', **kwargs)
        return config in self._config2outcome.keys()

    def _get_config_key(self, _mode='set', **kwargs):
        key = []
        for var_name in self.var_names:
            var_indexer = kwargs.get(var_name, None)
            if var_indexer is None:
                if _mode == 'set':
                    raise ValueError(f"must specify var {var_name}")
                var_indexer = slice(None)  # full slice indexer in “get” mode
            key.append(var_indexer)
        return tuple(key)

    def serialize(self):
        return {
            "var_names": self.var_names,
            "config2outcome": self._config2outcome,
            "outcome_shape": self.outcome_shape
        }

    @classmethod
    def deserialize(cls, data):
        try:
            obj = cls(data["var_names"])
            obj._config2outcome = data["config2outcome"]
            obj.outcome_shape = data["outcome_shape"]
        except KeyError as e:
            raise ValueError(f"Missing key in serialized data: {e}")
        return obj
