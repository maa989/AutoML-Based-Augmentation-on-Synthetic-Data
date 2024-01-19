import glob
from torch.utils.data import Dataset as torchDataset
from GTA5_labels import GTA5Labels
import os
from PIL import Image
import numpy as np

class GTA5(torchDataset):
    def __init__(self, root):
        self.root = root
        self.IMG_DIR_NAME = "images"
        self.LBL_DIR_NAME = "labels"
        self.SUFFIX = ".png"
        self.img_paths = self.create_imgpath_list()
        self.lbl_paths = self.create_lblpath_list()
        GTA5labels = GTA5Labels()
        self.label_map = GTA5labels.support_id_list()

    def create_imgpath_list(self):
        img_dir = os.path.join(self.root,self.IMG_DIR_NAME)
        img_path = [path for path in glob.glob(f"{img_dir}/*{self.SUFFIX}")]
        return img_path

    def create_lblpath_list(self):
        lbl_dir = os.path.join(self.root,self.LBL_DIR_NAME)
        lbl_path = [path for path in glob.glob(f"{lbl_dir}/*{self.SUFFIX}")]
        return lbl_path
    
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, idx, isPath=False):
        img_path, lbl_path = self.img_paths[idx], self.lbl_paths[idx]
        if isPath:
            return img_path, lbl_path

        img = self.read_img(img_path)
        lbl = self.read_img(lbl_path)
        return img, lbl
    def read_img(self, path):
        img = Image.open(str(path))
        img = np.array(img)
        return img
    def decode(self, cls, lbl):
        return cls._decode(lbl, label_map=cls.label_map.list_)
    def _decode(self, lbl, label_map):
        color_lbl = np.zeros((*lbl.shape, 3))
        for label in label_map:
            color_lbl[lbl == label.ID] = label.color
        return color_lbl