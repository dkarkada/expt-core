import torch
import torch.nn.functional as F
import torchvision

import numpy as np
from einops import rearrange, reduce

class ImageData():
    """
    Get image datasets as numpy arrays.
    """

    dataset_dict = {
        'mnist': torchvision.datasets.MNIST,
        'fmnist': torchvision.datasets.FashionMNIST,
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
        'svhn': torchvision.datasets.SVHN,
        'imagenet32': None,
        'imagenet64': None,
    }

    def __init__(self, dataset_name, data_dir, classes=None, onehot=True):
        """
        dataset_name (str): one of  'mnist', 'fmnist', 'cifar10', 'cifar100', 'imagenet32', 'imagenet64'
        dataset_dir (str): the directory where the raw dataset is saved
        classes (iterable): a list of groupings of old class labels that each constitute a new class.
            e.g. [[0,1], [8]] on MNIST would be a binary classification problem where the first class
            consists of samples of 0's and 1's and the second class has samples of 8's
        onehot (boolean): whether to use one-hot label encodings (typical for MSE loss). Default: True
        format (str): specify order of (sample, channel, height, width) dims. 'NCHW' default, or 'NHWC.'
            torchvision.dataset('cifar10') uses latter, needs ToTensor transform to reshape; former is ready-to-use.

        returns: numpy ndarray with shape (b, c, h, w)
        """

        assert dataset_name in self.dataset_dict
        self.name = dataset_name

        def format_data(dataset):
            if self.name in ['cifar10','cifar100']:
                X, y = dataset.data, dataset.targets
                X = rearrange(X, 'b h w c -> b c h w')
                y = np.array(y)
            if self.name in ['mnist', 'fmnist']:
                X, y = dataset.data.numpy(), dataset.targets.numpy()
                X = rearrange(X, 'b h w -> b 1 h w')
            if self.name in ['svhn']:
                X, y = dataset.data, dataset.labels
            if self.name in ['imagenet32', 'imagenet64']:
                X, y = dataset['data'], dataset['labels']
                X = rearrange(X, 'b d -> b c h w', c=3, h=32, w=32)
                y -= 1

            if classes is not None:
                # convert old class labels to new
                converter = -1 * np.ones(int(max(y)) + 1)
                for new_class, group in enumerate(classes):
                    group = [group] if type(group) == int else group
                    for old_class in group:
                        converter[old_class] = new_class
                # remove datapoints not in new classes
                mask = (converter[y] >= 0)
                X = X[mask]
                y = converter[y][mask]

            # make elements of input O(1)
            X = X/255.0
            # shape labels (N, nclasses)
            y = F.one_hot(torch.Tensor(y).long()).numpy() if onehot else y[:, None]

            return X.astype(np.float32), y.astype(np.float32)

        if self.name in ['cifar10','cifar100', 'mnist', 'fmnist']:
            raw_train = self.dataset_dict[self.name](root=data_dir, train=True, download=True)
            raw_test = self.dataset_dict[self.name](root=data_dir, train=False, download=True)
        if self.name == 'svhn':
            raw_train = self.dataset_dict[self.name](root=data_dir, split='train', download=True)
            raw_test = self.dataset_dict[self.name](root=data_dir, split='test', download=True)
        if self.name in ['imagenet32', 'imagenet64']:
            raw_train = np.load(f"{data_dir}/{self.name}-val.npz")
            raw_test = np.load(f"{data_dir}/{self.name}-val.npz")

        # process raw datasets
        self.train_X, self.train_y = format_data(raw_train)
        self.test_X, self.test_y = format_data(raw_test)

    def get_dataset(self, n, get="train", rng=None):
        """Generate an image dataset.

        n (int): the dataset size
        rng (numpy RNG): numpy RNG state for random sampling. Default: None
        get (str): either "train" or "test." Default: "train"

        Returns: tuple (X, y) such that X.shape = (n, *in_shape), y.shape = (n, *out_shape)
        """

        assert int(n) == n
        n = int(n)
        assert n > 0
        assert get in ["train", "test"]
        full_X, full_y = (self.train_X, self.train_y) if get == "train" else (self.test_X, self.test_y)

        # get subset
        idxs = slice(n) if rng is None else rng.choice(len(full_X), size=n, replace=False)
        X, y = full_X[idxs].copy(), full_y[idxs].copy()
        assert len(X) == n
        return X, y


def preprocess(X, **kwargs):
    """
    Process image dataset. Returns vectorized (flattened) images.
    
    X (ndarray): image dataset, shape (N, c, h, w)
    kwargs:
        "grayscale" (bool, False): If true, average over channels. Eliminates channel dim.
        "center" (bool, False): If true, center image vector distribution.
        "normalize" (bool, False): If true, make all image vectors unit norm.
        "zca_strength" (float, 0): Flatten covariance spectrum according to S_new = S / sqrt(zca_strength * S^2 + 1)

    returns: ndarray with shape (N, d)
    """

    if kwargs.get('grayscale', False):
        X = reduce(X, 'N 3 h w -> N (h w)', 'mean')
    else:
        X = rearrange(X, 'N c h w -> N (c h w)')

    if kwargs.get('center', False):
        X_mean = reduce(X, 'N d -> d', 'mean')
        X -= X_mean

    if kwargs.get('normalize', False):
        X /= np.linalg.norm(X, axis=1, keepdims=True)

    zca_strength = kwargs.get('zca_strength', 0)
    if zca_strength:
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        zca_strength /= np.mean(S**2)
        Sp = S / np.sqrt(zca_strength * S**2 + 1)
        Sp /= np.linalg.norm(Sp)
        X = U @ np.diag(Sp) @ Vt

    if kwargs.get('center', False):
        X_mean = reduce(X, 'N d -> d', 'mean')
        X -= X_mean

    return X
