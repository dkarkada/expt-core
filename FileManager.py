import numpy as np
import pickle
import os

class FileManager():
    
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
    
    def set_filepath(self, filepath):
        self.filepath = os.path.join(self.root, filepath)
        os.makedirs(self.filepath, exist_ok=True)
    
    def get_filename(self, fn, filepath=''):
        self.set_filepath(filepath)
        fn = os.path.join(self.filepath, fn)
        return fn

    def save(self, obj, fn, filepath=''):
        fn = self.get_filename(fn, filepath)
        if fn.endswith('.npy'):
            assert isinstance(obj, np.ndarray)
            np.save(fn, obj)
            return
        with open(fn, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, fn, filepath=''):
        fn = self.get_filename(fn, filepath)
        if not os.path.isfile(fn):
            return None
        if fn.endswith('.npy'):
            obj = np.load(fn)
            return obj
        with open(fn, 'rb') as handle:
            obj = pickle.load(handle)
        return obj