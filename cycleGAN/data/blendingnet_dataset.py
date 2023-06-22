import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import numpy as np

class BlendingnetDataset(BaseDataset):
    """
    Dataset for mouth blendingnet
    pass args:
        --dataset_mode blendingnet
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(
            preprocess='scale_width', input_nc=7, output_nc=3, direction="AtoB"
        )  # specify dataset-specific default values
        return parser
    
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A1 = os.path.join(opt.dataroot, opt.phase + 'A1')  # create a path '/path/to/data/trainA1'
        self.dir_A2 = os.path.join(opt.dataroot, opt.phase + 'A2')  # create a path '/path/to/data/trainA2'
        self.dir_A3 = os.path.join(opt.dataroot, opt.phase + 'A3')  # create a path '/path/to/data/trainA3'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A1_paths = sorted(make_dataset(self.dir_A1, opt.max_dataset_size))   # load images from '/path/to/data/trainA1'
        self.A2_paths = sorted(make_dataset(self.dir_A2, opt.max_dataset_size))   # load images from '/path/to/data/trainA2'
        self.A3_paths = sorted(make_dataset(self.dir_A3, opt.max_dataset_size))   # load images from '/path/to/data/trainA3'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        
        # self.A_paths = np.concatenate([self.A1_paths,self.A2_paths], axis=0)
        
        self.A1_size = len(self.A1_paths)  # get the size of dataset A
        self.A2_size = len(self.A2_paths)  # get the size of dataset A
        self.A3_size = len(self.A3_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        self.input_nc = self.opt.output_nc if btoA else 7     # get the number of channels of input image
        self.output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image
        

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A1_path = self.A1_paths[index % self.A1_size]  # make sure index is within then range
        A2_path = self.A2_paths[index % self.A2_size]  # make sure index is within then range
        A3_path = self.A3_paths[index % self.A3_size]  # make sure index is within then range

        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A1_img = Image.open(A1_path).convert('RGB')
        A1_img = A1_img.resize((512,512))
        A2_img = Image.open(A2_path).convert('RGB')
        A2_img = A2_img.resize((512,512))
        A3_img = Image.open(A3_path).convert('L')
        A3_img = A3_img.resize((512, 512))
        
        # A_img = Image.fromarray(A_img)
        B_img = Image.open(B_path).convert('RGB')
        
        # transform_params = get_params(self.opt, A_img.size)
        self.transform_A = get_transform(self.opt)
        self.transform_B = get_transform(self.opt)
        A1_img_trans = self.transform_A(A1_img) # 512*512*3
        A2_img_trans = self.transform_A(A2_img) # 512*512*3
        A3_img_trans = self.transform_A(A3_img) # 512*512*1
        A_img = np.concatenate([A1_img_trans, A2_img_trans, A3_img_trans],axis=2) # 512*512*7
        # apply image transformation
        # A = self.transform_A(A_img)
        
        A = A_img
        B = self.transform_B(B_img)

        return {'A': A, 'B': B, 'A_paths': A1_path, 'B_paths': B_path}
    # 'A2_paths': A2_path,

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A1_size, self.B_size)

