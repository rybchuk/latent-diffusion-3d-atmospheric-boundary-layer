import os
import numpy as np
import torch
from torch.utils.data import Dataset

ARR_EXTENSIONS = ['.npy']

def is_array_file(filename):
    return any(filename.endswith(extension) for extension in ARR_EXTENSIONS)

def make_dataset(dir, filetype='array'):
    if os.path.isfile(dir):
        samples = [i for i in np.genfromtxt(dir, dtype=np.str, encoding='utf-8')]
    else:
        samples = []
        assert os.path.isdir(dir), '%s is not a valid directory' % dir
        for root, _, fnames in sorted(os.walk(dir)):
            for fname in sorted(fnames):
                if filetype == 'image':
                    raise NotImplementedError
                elif filetype == 'array':
                    if is_array_file(fname):
                        path = os.path.join(root, fname)
                        samples.append(path)
                else:
                    raise ValueError(f'Filetype of "{filetype}" not recognized')           

    return samples

def numpy_loader(path):
    return np.load(path)

def numpy_transforms(arr, data_bounds):
    '''
    Inputs: 
      - arr: Numpy array, where data comes from natural values
      - data_bounds: Tuple containing values for rescaling (umin, umax, vmin, vmax[, wmin, wmax])
    Output: Torch tensor rescaled to [-1,1]
    '''

    # Rescale to [-1,1]
    if len(arr.shape) == 3:  # For planar data
        # First drop q but retain u&v for the QG data
        arr = arr[1:,:,:]  # Drop q, retain u+v

        # Then rescale
        umin, umax, vmin, vmax = data_bounds
        arr[0,:,:] = 2*(arr[0,:,:] - umin)/(umax - umin) - 1
        arr[1,:,:] = 2*(arr[1,:,:] - vmin)/(vmax - vmin) - 1

    elif len(arr.shape) == 4:  # For volume data
        umin, umax, vmin, vmax, wmin, wmax = data_bounds
        arr[0,:,:,:] = 2*(arr[0,:,:,:] - umin)/(umax - umin) - 1
        arr[1,:,:,:] = 2*(arr[1,:,:,:] - vmin)/(vmax - vmin) - 1
        arr[2,:,:,:] = 2*(arr[2,:,:,:] - wmin)/(wmax - wmin) - 1

    else:
        raise ValueError("Expected tuple containing (umin, umax, vmin, vmax[, wmin, wmax])!")

    # numpy array -> torch tensor with correct dtype
    arr = torch.from_numpy(arr).float()

    return arr

class LESBase(Dataset):
    def __init__(self, data_root=None, data_bounds=None, image_size=None):
        self.data_root = data_root
        self.data_bounds = data_bounds
        self.image_size = image_size
        self.loader = numpy_loader
        self.tfs = numpy_transforms

        self.data = make_dataset(self.data_root, filetype='array')  # Refer to samples as "imgs" for simplicity's sake

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        ret = self.tfs(self.loader(self.data[index]), self.data_bounds)
        return ret


class LEStrain(LESBase):
    NAME = "LES_train"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class LESvalidation(LESBase):
    NAME = "LES_validation"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class LEStest(LESBase):
    NAME = "LES_test"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)