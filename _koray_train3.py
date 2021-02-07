import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
from mmcv.runner import init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger


# ####################
# from mmseg.datasets.builder import DATASETS
# from mmseg.datasets.custom import CustomDataset
#
#
# @DATASETS.register_module()
# class SpaceDataset(CustomDataset):
#     CLASSES = ('background', 'building')
#     PALETTE = [[0, 0, 0], [255, 255, 255]]
#
#     def __init__(self, split, **kwargs):
#         super().__init__(img_suffix='_rgb.jpg', seg_map_suffix='_map.jpg',
#                          split=split, **kwargs)
#         # assert osp.exists(self.img_dir) and self.split is not None
#
#
# ####################

def get_cfg():
    # args_config = "C:/_koray/korhun/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_koray.py"
    args_config = "C:/_koray/korhun/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_koray_3.py"
    # args_config = "C:/_koray/korhun/mmsegmentation/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py"
    # args_config = "C:/_koray/korhun/mmsegmentation/configs/pspnet/pspnet_koray.py"
    if not os.path.isfile(args_config):
        print("File does not exists: " + args_config)
        exit(1)
    cfg = Config.fromfile(args_config)
    return cfg


def create_dir(dir_name, parents=True, exist_ok=True):
    from pathlib import Path
    Path(dir_name).mkdir(parents=parents, exist_ok=exist_ok)


def main():
    # args = parse_args()

    args_work_dir = "C:/_koray/korhun/mmsegmentation/data/space/work_dir"
    args_launcher = "none"
    args_seed = None
    args_deterministic = False
    args_no_validate = False
    config_name = "koray_train3"

    if not os.path.isdir(args_work_dir):
        create_dir(args_work_dir)

    # cfg = Config.fromfile(args_config)
    cfg = get_cfg()
    # if args.options is not None:
    #     cfg.merge_from_dict(args.options)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.work_dir = args_work_dir

    # if args.load_from is not None:
    #     cfg.load_from = args.load_from
    # if args.resume_from is not None:
    #     cfg.resume_from = args.resume_from
    # if args.gpu_ids is not None:
    #     cfg.gpu_ids = args.gpu_ids
    # else:
    #     cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    cfg.gpu_ids = range(0, 1)

    # init distributed env first, since logger depends on the dist info.
    if args_launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args_launcher, **cfg.dist_params)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, config_name))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args_seed is not None:
        logger.info(f'Set random seed to {args_seed}, deterministic: '
                    f'{args_deterministic}')
        set_random_seed(args_seed, deterministic=args_deterministic)
    cfg.seed = args_seed
    meta['seed'] = args_seed
    meta['exp_name'] = config_name

    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))

    logger.info(model)

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args_no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()

# python tools/train.py ${CONFIG_FILE} [optional arguments]
# python tools/train.py mmseg/datasets/custom.py
