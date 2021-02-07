import mmcv
import matplotlib.pyplot as plt
import os.path as osp
import numpy as np
from PIL import Image


# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMSegmentation installation
import mmseg
print(mmseg.__version__)


# img = mmcv.imread('data/iccv09Data/images/6000124.jpg')
# plt.figure(figsize=(8, 6))
# plt.imshow(mmcv.bgr2rgb(img))
# plt.show()

if __name__ == '__main__':




    # # convert dataset annotation to semantic segmentation map
    data_root = 'data/iccv09Data'
    img_dir = 'images'
    ann_dir = 'labels'
    # # define class and plaette for better visualization
    classes = ('sky', 'tree', 'road', 'grass', 'water', 'bldg', 'mntn', 'fg obj')
    palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34],
               [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]
    # for file in mmcv.scandir(osp.join(data_root, ann_dir), suffix='.regions.txt'):
    #   seg_map = np.loadtxt(osp.join(data_root, ann_dir, file)).astype(np.uint8)
    #   seg_img = Image.fromarray(seg_map).convert('P')
    #   seg_img.putpalette(np.array(palette, dtype=np.uint8))
    #   seg_img.save(osp.join(data_root, ann_dir, file.replace('.regions.txt',
    #                                                          '.png')))







    # # Let's take a look at the segmentation map we got
    # import matplotlib.patches as mpatches
    # img = Image.open('data/iccv09Data/labels/6000124.png')
    # plt.figure(figsize=(8, 6))
    # im = plt.imshow(np.array(img.convert('RGB')))
    #
    # # create a patch (proxy artist) for every color
    # patches = [mpatches.Patch(color=np.array(palette[i])/255.,
    #                           label=classes[i]) for i in range(8)]
    # # put those patched as legend-handles into the legend
    # plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,
    #            fontsize='large')
    #
    # plt.show()







    # # split train/val set randomly
    # split_dir = 'splits'
    # mmcv.mkdir_or_exist(osp.join(data_root, split_dir))
    # filename_list = [osp.splitext(filename)[0] for filename in mmcv.scandir(
    #     osp.join(data_root, ann_dir), suffix='.png')]
    # with open(osp.join(data_root, split_dir, 'train.txt'), 'w') as f:
    #   # select first 4/5 as train set
    #   train_length = int(len(filename_list)*4/5)
    #   f.writelines(line + '\n' for line in filename_list[:train_length])
    # with open(osp.join(data_root, split_dir, 'val.txt'), 'w') as f:
    #   # select last 1/5 as train set
    #   f.writelines(line + '\n' for line in filename_list[train_length:])






    from mmseg.datasets.builder import DATASETS
    from mmseg.datasets.custom import CustomDataset

    @DATASETS.register_module()
    class StandfordBackgroundDataset(CustomDataset):
      CLASSES = classes
      PALETTE = palette
      def __init__(self, split, **kwargs):
        super().__init__(img_suffix='.jpg', seg_map_suffix='.png',
                         split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None

    from mmcv import Config
    cfg = Config.fromfile('configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py')


    from mmseg.apis import set_random_seed

    # Since we use ony one GPU, BN is used instead of SyncBN
    cfg.norm_cfg = dict(type='BN', requires_grad=True)
    cfg.model.backbone.norm_cfg = cfg.norm_cfg
    cfg.model.decode_head.norm_cfg = cfg.norm_cfg
    cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
    # modify num classes of the model in decode/auxiliary head
    cfg.model.decode_head.num_classes = 8
    cfg.model.auxiliary_head.num_classes = 8

    # Modify dataset type and path
    cfg.dataset_type = 'StandfordBackgroundDataset'
    cfg.data_root = data_root

    cfg.data.samples_per_gpu = 8
    cfg.data.workers_per_gpu=8

    cfg.img_norm_cfg = dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
    cfg.crop_size = (256, 256)
    cfg.train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations'),
        dict(type='Resize', img_scale=(320, 240), ratio_range=(0.5, 2.0)),
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
            img_scale=(320, 240),
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
    cfg.data.train.split = 'splits/train.txt'

    cfg.data.val.type = cfg.dataset_type
    cfg.data.val.data_root = cfg.data_root
    cfg.data.val.img_dir = img_dir
    cfg.data.val.ann_dir = ann_dir
    cfg.data.val.pipeline = cfg.test_pipeline
    cfg.data.val.split = 'splits/val.txt'

    cfg.data.test.type = cfg.dataset_type
    cfg.data.test.data_root = cfg.data_root
    cfg.data.test.img_dir = img_dir
    cfg.data.test.ann_dir = ann_dir
    cfg.data.test.pipeline = cfg.test_pipeline
    cfg.data.test.split = 'splits/val.txt'

    # We can still use the pre-trained Mask RCNN model though we do not need to
    # use the mask branch
    cfg.load_from = 'checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'

    # Set up working dir to save files and logs.
    cfg.work_dir = './work_dirs/tutorial'

    cfg.total_iters = 200
    cfg.log_config.interval = 10
    cfg.evaluation.interval = 200
    cfg.checkpoint_config.interval = 200

    # Set seed to facitate reproducing the result
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Let's have a look at the final config used for training
    print(f'Config:\n{cfg.pretty_text}')










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
    train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                    meta=dict())









    #
    # img = mmcv.imread('data/iccv09Data/images/6000124.jpg')
    #
    # model.cfg = cfg
    # result = inference_segmentor(model, img)
    # plt.figure(figsize=(8, 6))
    # show_result_pyplot(model, img, result, palette)
