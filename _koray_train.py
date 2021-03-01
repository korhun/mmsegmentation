from typing import AnyStr

import io

import shutil

import mmcv
import os.path as osp
import numpy as np
from PIL import Image

# conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
# pip uninstall mmcv --yes
# pip uninstall mmcv-full --yes
# git clone https://github.com/open-mmlab/mmcv.git
# cd C:\_koray\git\mmcv
# https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16
# set MMCV_WITH_OPS=1
# pip install -e .
# cd C:\_koray\korhun\mmsegmentation
# python _koray.py
# python C:\_koray\korhun\mmsegmentation\_koray.py


# # Check Pytorch installation
import torch, torchvision

#
# # # Check MMSegmentation installation
# import mmseg
# print(mmseg.__version__)

# exit(0)


####################
from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset


@DATASETS.register_module()
class SpaceDataset(CustomDataset):
    CLASSES = ('background', 'building')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, split, **kwargs):
        super().__init__(img_suffix='_rgb.jpg', seg_map_suffix='_map.jpg',
                         split=split, **kwargs)
        # assert osp.exists(self.img_dir) and self.split is not None


####################


if __name__ == '__main__':
    print(torch.__version__, torch.cuda.is_available())

    data_root = "C:/_koray/korhun/mmsegmentation/data/space"
    img_dir = "C:/_koray/korhun/mmsegmentation/data/space/img"
    ann_dir = "C:/_koray/korhun/mmsegmentation/data/space/ann"
    # img_size = (320, 240)
    img_size = (900, 900)

    from mmcv import Config

    # cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py')
    cfg = Config.fromfile('C:/_koray/korhun/mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py')
    from mmseg.apis import set_random_seed

    # Since we use ony one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = 1
    cfg.model.auxiliary_head.num_classes = 1

    # Modify dataset type and path
    cfg.dataset_type = 'SpaceDataset'
    cfg.data_root = data_root

    cfg.data.samples_per_gpu = 8
    cfg.data.workers_per_gpu = 8
    # cfg.data.samples_per_gpu = 1
    # cfg.data.workers_per_gpu = 1

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.crop_size = (256, 256)
    # cfg.crop_size = (900, 900)
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=img_size, ratio_range=(0.5, 2.0)),
        dict(type='RandomCrop', crop_size=cfg.crop_size, cat_max_ratio=0.75),
        dict(type='RandomFlip', flip_ratio=0.5),
        dict(type='PhotoMetricDistortion'),
        dict(type='Normalize', **cfg.img_norm_cfg),
        dict(type='Pad', size=cfg.crop_size, pad_val=0, seg_pad_val=255),
        dict(type='DefaultFormatBundle'),
        dict(type='Collect', keys=['img', 'gt_semantic_seg']),
    ]

    cfg.test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(
            type='MultiScaleFlipAug',
            img_scale=img_size,
            # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
            flip=False,
            transforms=[
                dict(type='Resize', keep_ratio=True),
                dict(type='RandomFlip'),
                dict(type='Normalize', **cfg.img_norm_cfg),
                dict(type='ImageToTensor', keys=['img']),
                dict(type='Collect', keys=['img']),
            ])
    ]

    cfg.data.train.type = cfg.dataset_type
    cfg.data.train.data_root = cfg.data_root
    cfg.data.train.img_dir = img_dir
    cfg.data.train.ann_dir = ann_dir
    cfg.data.train.pipeline = cfg.train_pipeline
    # cfg.data.train.split = 'splits/train.txt'
    cfg.data.train.split = None

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.pipeline = cfg.test_pipeline
    # cfg.data.val.split = 'splits/val.txt'
    cfg.data.val.split = None

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.pipeline = cfg.test_pipeline
    # cfg.data.test.split = 'splits/val.txt'
    cfg.data.test.split = None

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    # cfg.load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
    cfg.load_from = 'C:/_koray/korhun/mmsegmentation/checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

    # Set up working dir to save files and logs.
    # cfg.work_dir = './work_dirs/tutorial'
    cfg.work_dir = 'C:/_koray/korhun/mmsegmentation/data/space/work_dir'

    cfg.total_iters = 200
    cfg.log_config.interval = 10
    cfg.evaluation.interval = 200
    cfg.checkpoint_config.interval = 200

    # Set seed to facitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # # Let's have a look at the final config used for training
    # print(f'Config:\n{cfg.pretty_text}')

    from mmseg.datasets import build_dataset
    from mmseg.models import build_segmentor
    from mmseg.apis import train_segmentor

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    # model = build_segmentor(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    train_cfg = cfg.get('train_cfg')
    test_cfg = cfg.get('test_cfg')
    model = build_segmentor(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # Create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    train_segmentor(model, datasets, cfg, distributed=False, validate=True, meta=dict())
