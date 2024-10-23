import numpy as np


class ExperimentResults():
    """
    A class to manage *scalar* measurables organized along different named axes. One axis for
    each independent variable; if multiple dependent variables are measured, they should be organized
    along a final axis, e.g., axis=('result', ['measurable1', 'measurable2', ...]).

    The measured results are written to an ndarray of shape [len(axis_vals) for each axis]. Different
    subsets of the full results can be accessed by slicing the results tensor along certain axes.
    The methods here accept **kwargs of the form axis_name=axis_value, which indicates that a slice
    of the results tensor should be taken across axis_name at axis_value. (e.g., 'trial'=20
    refers to all the results for trial 20). If multiple slices are taken, the intersection is returned.
    If no slices are specified, all results are returned.
    """

    def __init__(self, axes, metadata=None):
        """
        axes (list of tuples): [(axis_name:str, axis_values:iterable) for all axes]
        metadata (dict): any additional metadata. Default: None
        """
        self.metadata = metadata
        self.axes = axes.copy()
        shape = [len(axis) for (_, axis) in axes]
        self.results = np.zeros(shape=shape)
        self.written = np.zeros(shape=shape)

        converter = lambda vals: {val: idx for idx, val in enumerate(vals)}
        self.idx_converter = {ax_name: converter(vals) for (ax_name, vals) in self.axes}

    def _to_idxs(self, **kwargs):
        idxs = []
        for (ax_name, _) in self.axes:
            if ax_name in kwargs:
                ax_val = kwargs[ax_name]
                idx = self.idx_converter[ax_name].get(ax_val)
                assert idx is not None, f"{ax_name}={ax_val}"
                idxs.append(slice(idx, idx+1))
            else:
                idxs.append(slice(None))
        return tuple(idxs)

    def is_written(self, **kwargs):
        """
        Check if specified results slice has been written. Useful for e.g., avoiding recomputing
        results when re-running an experiment that had been interrupted.

        **kwargs: kwargs of the form axis_name=axis_value
        Returns: True if all values have been written, False otherwise.
        """
        idxs = self._to_idxs(**kwargs)
        return np.all(self.written[idxs])

    def get(self, stats_axes=None, **kwargs):
        """
        Get slice of experiment results, at the specified axis-value pairs. If stats_axes are specified,
        calculate statistics along these axes. Results along unnamed axes are returned in full.

        stats_axes (list or None): List of names of the axes along which statistics are calculated.
        **kwargs: Specify slice of full results tensor with kwargs of the form axis_name=axis_value.

        Returns: If stats_axes is None, returns ndarray of results slice. Any sliced axes are squeezed out.
                 If stats_axes is provided, returns tuple (ndarray, ndarray) containing mean and std.
        """
        idxs = self._to_idxs(**kwargs)
        if not np.all(self.written[idxs]):
            print("warning: not all values have been written to")
        result = self.results[idxs]
        if stats_axes:
            axes_idxs = tuple([self.get_axis(axis, get_idx=True) for axis in stats_axes])
            mean = result.mean(axis=axes_idxs).squeeze()
            std = result.std(axis=axes_idxs).squeeze()
            return mean, std
        return result.squeeze()

    def get_axis(self, axis_name, get_idx=False):
        """
        Get either the values or the index of a specified axis.

        axis_name (str): The name of the axis.
        get_idx (bool): If True, return the index of the axis (re: the shape of the results tensor)
                        instead of the axis values. Default: False

        Returns: list or int: Axis values or index.
        """
        for i, (ax_name, vals) in enumerate(self.axes):
            if ax_name == axis_name:
                return i if get_idx else vals
        print(f"Axis '{axis_name}' not found.")
        return False

    def write(self, write_vals, **kwargs):
        """
        Write values to a slice of the results tensor.

        write_vals (ndarray): Values to be written.
        **kwargs: Specify slice of results tensor with kwargs of the form axis_name=axis_value.
        """
        idxs = self._to_idxs(**kwargs)
        idxs = tuple([slc if slc.start is None else slc.start for slc in idxs])
        hole_shape, fill_shape = np.shape(self.results[idxs]), np.shape(write_vals)
        if hole_shape != fill_shape:
            print(f"Bad shape: writing into shape = {hole_shape}, # writes = {fill_shape}.")
            return
        self.results[idxs] = write_vals
        self.written[idxs] = True

    def print_axes(self):
        for (ax_name, axis) in self.axes:
            print(f"{ax_name}: {axis}")
