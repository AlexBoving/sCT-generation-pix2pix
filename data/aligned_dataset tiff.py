import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
import numpy as np
import torch
from PIL import Image
import tifffile as tiff


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        # If my image is in 
        #AB = Image.open(AB_path).convert('I')
        stack = tiff.imread(AB_path)

        ct1 = stack[:, :, 0]
        ct2 = stack[:, :, 1]
        ct3 = stack[:, :, 2]

        # Convert to PIL image of 16-bit
        ct1 = Image.fromarray(ct1).convert('I;16')
        ct2 = Image.fromarray(ct2).convert('I;16')
        ct3 = Image.fromarray(ct3).convert('I;16')

        # split AB image into A and B
        w, h = ct1.size
        w2 = int(w / 2)

        MR1 = ct1.crop((0, 0, w2, h))
        MR2 = ct2.crop((0, 0, w2, h))
        MR3 = ct3.crop((0, 0, w2, h))
        CT2 = ct2.crop((w2, 0, w, h)) # C'est celui ci qu'on veut

        width, height = MR1.size
        
        CT2 = np.array(CT2, dtype=np.float32)
        MR1 = np.array(MR1, dtype=np.float32)
        MR2 = np.array(MR2, dtype=np.float32)
        MR3 = np.array(MR3, dtype=np.float32)

        if (CT2.max() - CT2.min()) != 0:
            CT2 = (CT2 - CT2.min()) / (CT2.max() - CT2.min())
        else:
            CT2 = CT2 * 0
        if (MR1.max() - MR1.min()) != 0:
            MR1 = (MR1 - MR1.min()) / (MR1.max() - MR1.min())
        else:
            MR1 = MR1 * 0
        if (MR2.max() - MR2.min()) != 0:
            MR2 = (MR2 - MR2.min()) / (MR2.max() - MR2.min())
        else:
            MR2 = MR2 * 0
        if (MR3.max() - MR3.min()) != 0:
            MR3 = (MR3 - MR3.min()) / (MR3.max() - MR3.min())
        else:
            MR3 = MR3 * 0

        # Convert the arrays back to PIL images
        CT2 = Image.fromarray(CT2, mode='F') # F: 32-bit floating point pixel
        MR1 = Image.fromarray(MR1, mode='F') # F: 32-bit floating point pixel
        MR2 = Image.fromarray(MR2, mode='F') # F: 32-bit floating point pixel
        MR3 = Image.fromarray(MR3, mode='F') # F: 32-bit floating point pixel

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, width, height)
        A_transform = get_transform(self.opt, transform_params, grayscale=True)
        B_transform = get_transform(self.opt, transform_params, grayscale=True)

        MR1_tensor = A_transform(MR1)
        MR2_tensor = A_transform(MR2)
        MR3_tensor = A_transform(MR3)
        CT = B_transform(CT2)

        # I want to have the three MR images in the first, second, and third channel of the tensor
        MR = torch.cat((MR1_tensor, MR2_tensor, MR3_tensor), dim=0)

        return {'A': MR, 'B': CT, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
