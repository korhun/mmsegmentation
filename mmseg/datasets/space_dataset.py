# koray
import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class PascalVOCDataset(CustomDataset):
    CLASSES = ('background', 'building')

    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, split, **kwargs):
        super(PascalVOCDataset, self).__init__(
            img_suffix='_rgb.jpg', seg_map_suffix='_map.jpg', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
