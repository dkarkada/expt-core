import numpy as np
import itertools
from typing import Comparable

class ExptTrace():

    @classmethod
    def multi_init(cls, num_init, var_names):
        return [cls(var_names) for _ in range(num_init)]

    def __init__(self, var_names):
        if not isinstance(var_names, list):
            raise ValueError("var_names must be a list")
        if "val" in var_names:
            raise ValueError("variable name 'val' disallowed")
        self.var_names = var_names.copy()
        self.vals = {}
        self.valshape = None

    def __setitem__(self, key, val):
        if self.valshape is None:
            val_array = np.asarray(val)
            if not np.issubdtype(val_array.dtype, np.number):
                raise ValueError("value must be numeric")
            self.valshape = val_array.shape
        if np.shape(val) != self.valshape:
            raise ValueError(f"value shape {np.shape(val)} != expected {self.valshape}")
        key = (key,) if not isinstance(key, tuple) else key
        if len(key) != len(self.var_names):
            raise ValueError(f"num keys {len(key)} != num vars {len(self.var_names)}")
        for k in key:
            if not isinstance(k, Comparable):
                raise ValueError(f"key element {k} must support ordering operations")

        if key in self.vals:
            raise ValueError(f"key {key} already exists. overwriting not supported")
        self.vals[key] = val

    def __getitem__(self, key):
        if self.valshape is None:
            raise RuntimeError("must add items before getting")
        key = (key,) if not isinstance(key, tuple) else key
        if len(key) != len(self.var_names):
            raise ValueError(f"num keys {len(key)} != num vars {len(self.var_names)}")

        key_axes = []
        for idx, var_name in enumerate(self.var_names):
            key_i = key[idx]
            key_idx_extent = [key_i]
            if isinstance(key_i, slice):
                slice_is_full = all([x==None for x in [key_i.start, key_i.stop, key_i.step]])
                if not slice_is_full:
                    raise ValueError(f"slice start/stop/step not supported ({var_name})")
                key_idx_extent = self.get_axis(var_name)
            key_axes.append(key_idx_extent)
        shape = [len(key_idx_extent) for key_idx_extent in key_axes]

        if np.prod(shape) == 1 and len(self.vals) > 1:
            if key not in self.vals:
                raise KeyError(f"key {key} not found")
            return self.vals[key]
        vals = np.ma.masked_all(shape + list(self.valshape))

        idx_maps = []
        for axis in key_axes:
            idx_maps.append({val: i for i, val in enumerate(axis)})
        for key in itertools.product(*key_axes):
            shape_idxs = tuple(idx_maps[dim][val] for dim, val in enumerate(key))
            if key in self.vals:
                vals[shape_idxs] = self.vals[key]

        if not np.ma.is_masked(vals):
            return np.array(vals)
        return vals

    def get_axis(self, var_name):
        if var_name not in self.var_names:
            raise ValueError(f"var {var_name} not found")
        idx = self.var_names.index(var_name)
        key_idx_extent = set()
        for keys in self.vals.keys():
            key_idx_extent.add(keys[idx])
        return sorted(list(key_idx_extent))

    def get(self, **kwargs):
        key = self._get_key(_mode='get', **kwargs)
        return self[key]

    def set(self, **kwargs):
        if "val" not in kwargs:
            raise ValueError(f"no val given")
        val = kwargs["val"]
        key = self._get_key(_mode='set', **kwargs)
        self[key] = val

    def is_written(self, **kwargs):
        key = self._get_key(_mode='set', **kwargs)
        return key in self.vals

    def _get_key(self, _mode='set', **kwargs):
        for var_name in self.var_names:
            if _mode == 'set':
                if var_name not in kwargs:
                    raise ValueError(f"must specify var {var_name}")
            elif _mode == 'get':
                if var_name not in kwargs:
                    kwargs[var_name] = slice(None, None, None)
            if kwargs[var_name] is None:
                raise ValueError(f"var {var_name} cannot be None")
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
        try:
            obj = cls(data["var_names"])
            obj.vals = data["vals"]
            obj.valshape = data["valshape"]
        except KeyError as e:
            raise ValueError(f"Missing key in serialized data: {e}")
        return obj
