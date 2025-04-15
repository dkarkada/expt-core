import numpy as np
import itertools

class ExptTrace():

    @classmethod
    def multi_init(cls, num_init, var_names):
        return [cls(var_names) for _ in range(num_init)]

    def __init__(self, var_names):
        assert "val" not in var_names, f"variable name 'val' disallowed"
        self.var_names = var_names
        self.vals = {}
        self.valshape = None

    def __setitem__(self, key, val):
        if self.valshape is None:
            self.valshape = np.shape(val)
        assert np.shape(val) == self.valshape, f"value shape {np.shape(val)} != expected {self.valshape}"
        key = tuple((key,)) if not isinstance(key, tuple) else key
        assert len(key) == len(self.var_names), f"num keys {len(key)} != num vars {len(self.var_names)}"
        assert key not in self.vals, f"key {key} already exists. overwriting not supported"
        self.vals[key] = val

    def __getitem__(self, key):
        assert self.valshape is not None, "must add items before getting"
        key = tuple((key,)) if not isinstance(key, tuple) else key
        assert len(key) == len(self.var_names), f"num keys {len(key)} != num vars {len(self.var_names)}"
        key_axes = []
        for idx, var_name in enumerate(self.var_names):
            key_i = key[idx]
            key_idx_extent = [key_i]
            if isinstance(key_i, slice):
                slice_is_full = all([x==None for x in [key_i.start, key_i.stop, key_i.step]])
                assert slice_is_full, f"slice start/stop/step not supported ({var_name})"
                key_idx_extent = self.get_axis(var_name)
            key_axes.append(key_idx_extent)
        shape = [len(key_idx_extent) for key_idx_extent in key_axes]
        if np.prod(shape) == 1:
            assert key in self.vals, f"key {key} not found"
            return self.vals[key]
        vals = np.zeros(shape + list(self.valshape))

        idx_maps = []
        for axis in key_axes:
            idx_maps.append({val: i for i, val in enumerate(axis)})
        for key in itertools.product(*key_axes):
            shape_idxs = tuple(idx_maps[dim][val] for dim, val in enumerate(key))
            assert key in self.vals, f"key {key} not found"
            vals[shape_idxs] = self.vals[key]

        return vals

    def get_axis(self, var_name):
        assert var_name in self.var_names, f"var {var_name} not found"
        idx = self.var_names.index(var_name)
        key_idx_extent = set()
        for keys in self.vals.keys():
            key_idx_extent.add(keys[idx])
        return sorted(list(key_idx_extent))

    def get(self, **kwargs):
        key = self._get_key(_mode='get', **kwargs)
        return self[key]

    def set(self, **kwargs):
        assert "val" in kwargs, f"no val given"
        val = kwargs["val"]
        key = self._get_key(_mode='set', **kwargs)
        self[key] = val

    def is_written(self, **kwargs):
        key = self._get_key(_mode='set', **kwargs)
        return key in self.vals

    def _get_key(self, _mode='set', **kwargs):
        for var_name in self.var_names:
            if _mode == 'set':
                assert var_name in kwargs, f"must specify var {var_name}"
            elif _mode == 'get':
                if var_name not in kwargs:
                    kwargs[var_name] = slice(None, None, None)
            assert kwargs[var_name] is not None, f"var {var_name} cannot be None"
        key = tuple([kwargs[var_name] for var_name in self.var_names])
        return key

    def serialize(self):
        return {
            "var_names": self.var_names,
            "vals": self.vals,
            "valshape": self.valshape
        }

    @classmethod
    def deserialize(cls, data):
        obj = cls(data["var_names"])
        obj.vals = data["vals"]
        obj.valshape = data["valshape"]
        return obj
