import torch
import torch.nn.functional as F
import torchvision

import numpy as np
from numpy.fft import rfft2, rfftfreq, fftfreq, irfft2

class ImageData():

    dataset_dict = {
        'mnist': torchvision.datasets.MNIST,
        'fmnist': torchvision.datasets.FashionMNIST,
        'cifar10': torchvision.datasets.CIFAR10,
        'cifar100': torchvision.datasets.CIFAR100,
        'svhn': torchvision.datasets.SVHN,
        'imagenet32': None,
        'imagenet64': None,
    }

    def __init__(self, dataset_name, work_dir=".", classes=None, binarize=False):
        """
        dataset_name (str): one of  'mnist', 'fmnist', 'cifar10', 'cifar100', 'imagenet32', 'imagenet64'
        classes (iterable): a list of groupings of old class labels that each constitute a new class.
            e.g. [[0,1], [8]] on MNIST would be a binary classification problem where the first class
            consists of samples of 0's and 1's and the second class has samples of 8's
        binarize (boolean): whether to use +1/-1 label encoding. Ignored if num_classes!=2
        """

        assert dataset_name in self.dataset_dict
        self.name = dataset_name

        def format_data(dataset):
            if self.name in ['cifar10','cifar100']:
                X, y = dataset.data, dataset.targets
            if self.name in ['mnist', 'fmnist']:
                X, y = dataset.data.numpy(), dataset.targets.numpy()
                X = X[:,:,:,None]
            if self.name in ['svhn']:
                X, y = dataset.data, dataset.labels
                X = X.transpose(0, 2, 3, 1)
            if self.name in ['imagenet32', 'imagenet64']:
                X, y = dataset['data'], dataset['labels']
                y -= 1
                X = X.reshape(-1, 32, 32, 3)
                
            n_classes = int(max(y)) + 1

            if classes is not None:
                # convert old class labels to new
                converter = -1 * np.ones(n_classes)
                for new_class, group in enumerate(classes):
                    group = [group] if type(group) == int else group
                    for old_class in group:
                        converter[old_class] = new_class
                # remove datapoints not in new classes
                mask = (converter[y] >= 0)
                X = X[mask]
                y = converter[y][mask]
                # update n_classes
                n_classes = int(max(y)) + 1

            # onehot encoding, unless binary classification (+1,-1)
            if n_classes == 2 and binarize:
                y = 2*y - 1
                y = y[:, None] #reshape
            else:
                y = F.one_hot(torch.Tensor(y).long()).numpy()

            return X.astype(np.float32), y.astype(np.float32)
        
        if self.name in ['cifar10','cifar100', 'mnist', 'fmnist']:
            raw_train = self.dataset_dict[self.name](root=f'{work_dir}/data', train=True, download=True)
            raw_test = self.dataset_dict[self.name](root=f'{work_dir}/data', train=False, download=True)
        if self.name == 'svhn':
            raw_train = self.dataset_dict[self.name](root=f'{work_dir}/data', split='train', download=True)
            raw_test = self.dataset_dict[self.name](root=f'{work_dir}/data', split='test', download=True)            
        if self.name in ['imagenet32', 'imagenet64']:
            raw_train = np.load(f"{work_dir}/data/{self.name}-val.npz")
            raw_test = np.load(f"{work_dir}/data/{self.name}-val.npz")

        # process raw datasets
        self.train_X, self.train_y = format_data(raw_train)
        self.test_X, self.test_y = format_data(raw_test)
        
        self.train_mean = self.train_X.mean(axis=0, keepdims=True)
        self.train_std = self.train_X.std(axis=0, keepdims=True)
        
        self.FT = FourierImage(self.train_X[0].shape)

    def get_dataset(self, n, rng=None, get="train", **kwargs):
        """Generate an image dataset.

        n (int): the dataset size
        rng (numpy RNG): numpy RNG state for random sampling. Default: None
        get (str): either "train" or "test." Default: "train"

        Returns: tuple (X, y) such that X.shape = (n, d_in), y.shape = (n, d_out)
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
        
        return self.preprocess(X, y, **kwargs)

    def preprocess(self, X, y, **kwargs):
        if 'fourier' in kwargs:
            exponent, phase_rand = kwargs.get('fourier')
            X = self.FT.change_ft_pl(X, exponent, phase_rand)
        elif kwargs.get('center', False):
            X -= self.train_mean
        if kwargs.get('flatten', False):
            X = X.reshape((len(X), -1))
        return X, y


class FourierImage:

    def __init__(self, shape):
        assert len(shape) == 3
        self.shape = shape
        kkx = rfftfreq(self.shape[0], d=1/self.shape[0])
        kky = fftfreq(self.shape[1], d=1/self.shape[1])
        self.kk = np.sqrt(kky[:, None]**2 + kkx[None, :]**2)

    def ft(self, img):
        assert img.shape == self.shape
        return self.ftchan(img, rfft2)

    def ftchan(self, img, transf):
        nchan = self.shape[-1]
        ftimg = [transf(img[:,:,c]) for c in range(nchan)]
        return np.moveaxis(ftimg, 0, 2)
    
    def change_ft_pl(self, dataset, exponent, phase_rand=1):
        new_pl = (self.kk + 1e-10) ** float(exponent/2)
        new_pl[0, 0] = 1
        
        def change_pl(img):
            new_ftimg = self.ft(img) * new_pl[:, :, None]

            n = phase_rand
            phase = np.random.uniform(-np.pi*n, np.pi*n, size=self.kk.shape)
            phase[0, 0] = 0
            new_ftimg *= np.exp(1j*phase)[:, :, None]

            newimg = self.ftchan(new_ftimg, irfft2)
            assert (newimg.imag == 0).all()
            return (newimg - newimg.min()) / (newimg.max() - newimg.min())
        
        new_dataset = [change_pl(img) for img in dataset]
        return np.array(new_dataset)