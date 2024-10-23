import numpy as np
import pickle
import os


class FileManager():

    def __init__(self, root):
        """
        root (str): The root directory from which this FileManager works.
        """
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.filepath = self.root

    def set_filepath(self, *paths):
        """
        Set the current filepath relative to the root directory. Helpful for temporarily
        going into a subdirectory.

        *paths (str): Variable number of path components to join.
        """
        self.filepath = os.path.join(self.root, *paths)
        os.makedirs(self.filepath, exist_ok=True)

    def get_filename(self, fn):
        """
        Get the absolute file path given a filename relative to the current filepath.
        fn (str): The filename relative to the current filepath.
        """
        return os.path.join(self.filepath, fn)

    def save(self, obj, fn):
        """
        Store an object to disk.

        obj (object): The object to be saved.
        fn (str): The filename relative to the current filepath. Should end in .npy if obj is ndarray.
        """
        fn = self.get_filename(fn)
        if fn.endswith('.npy'):
            assert isinstance(obj, np.ndarray)
            np.save(fn, obj)
            return
        with open(fn, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, fn):
        """
        Load an object from disk.

        fn (str): The filename relative to the current filepath.
        Returns: The loaded object, or None if the file does not exist.
        """
        fn = self.get_filename(fn)
        if not os.path.isfile(fn):
            return None
        if fn.endswith('.npy'):
            obj = np.load(fn)
            return obj
        with open(fn, 'rb') as handle:
            obj = pickle.load(handle)
        return obj
