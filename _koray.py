# import file_helper
from fnmatch import fnmatch

import cv2
import numpy as np
from typing import AnyStr

# # Check Pytorch installation
import torch, torchvision

from screen_capturer import ScreenCapturer
from youtube import YoutubeVideoSource

print(torch.__version__, torch.cuda.is_available())
# conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
# python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available())"
# should print -> 1.7.1 True


from mmseg.apis import inference_segmentor, init_segmentor
import mmcv
import os


def wildcard(txt, pattern, case_insensitive=True):
    if txt == pattern:
        return True
    else:
        return fnmatch(txt.lower(), pattern.lower()) if case_insensitive else fnmatch(txt, pattern)


def path_join(a: AnyStr, *paths: AnyStr) -> AnyStr:
    return os.path.join(a, *paths).replace("/", os.path.sep)


def enumerate_files(dir_path, recursive=False, wildcard_pattern=None, case_insensitive=True):
    if wildcard_pattern is None:
        for root, sub_dirs, files in os.walk(dir_path):
            for name in files:
                yield path_join(root, name)
            if not recursive:
                break
    else:
        for root, sub_dirs, files in os.walk(dir_path):
            for name in files:
                # name = os.path.basename(fn)
                if wildcard(name, wildcard_pattern, case_insensitive=case_insensitive):
                    yield path_join(root, name)
            if not recursive:
                break


# https://github.com/korhun/mmsegmentation/tree/master/configs/

# config_file = 'configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
# checkpoint_file = 'checkpoints/pspnet_r50b-d8_512x1024_80k_cityscapes_20201225_094315-6344287a.pth'


# config_file = 'configs/hrnet/fcn_hr48_512x1024_160k_cityscapes.py'
# checkpoint_file = 'checkpoints/fcn_hr48_512x1024_160k_cityscapes_20200602_190946-59b7973e.pth'

# config_file = 'configs/hrnet/fcn_hr48_480x480_80k_pascal_context.py'
# checkpoint_file = 'checkpoints/fcn_hr48_480x480_80k_pascal_context_20200911_155322-847a6711.pth'


# config_file = 'configs/ocrnet/ocrnet_hr48_512x512_40k_voc12aug.py'
# checkpoint_file = 'checkpoints/ocrnet_hr48_512x512_40k_voc12aug_20200614_015958-255bc5ce.pth'

# config_file = 'configs/ocrnet/ocrnet_hr48_512x512_160k_ade20k.py'
# checkpoint_file = 'checkpoints/ocrnet_hr48_512x512_160k_ade20k_20200615_184705-a073726d.pth'

#
# # config_file = 'configs/ocrnet/ocrnet_hr48_512x1024_40k_cityscapes.py'
# # checkpoint_file = 'checkpoints/ocrnet_hr48_512x1024_40k_cityscapes_20200601_033336-55b32491.pth'
#
#
#


# config_file = 'configs/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes.py'
# checkpoint_file = 'checkpoints/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'


# config_file = "C:/_koray/korhun/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_koray_deneme.py"
# config_file = "C:/_koray/korhun/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_koray.py"
# checkpoint_file = "C:/_koray/korhun/mmsegmentation/data/space/work_dir/latest.pth"
# checkpoint_file = "C:/_koray/korhun/mmsegmentation/data/space/work_dir_1/latest.pth"
# checkpoint_file = "C:/_koray/korhun/mmsegmentation/data/space/work_dir_2/latest.pth"


# config_file = "C:/_koray/korhun/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_koray_3.py"
# checkpoint_file = "C:/_koray/korhun/mmsegmentation/data/space/work_dir_3/latest.pth"

###ok
config_file = "C:/_koray/korhun/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_koray.py"
checkpoint_file = "C:/_koray/korhun/mmsegmentation/data/space/work_dir_2/latest.pth"


# #
#
# config_file = 'configs/ocrnet/ocrnet_r101-d8_512x1024_80k_b16_cityscapes.py'
# checkpoint_file = 'checkpoints/ocrnet_r101-d8_512x1024_80k_b16_cityscapes-78688424.pth'


# config_file = 'C:/_koray/korhun/mmsegmentation/configs/unet/pspnet_unet_s5-d16_256x256_40k_hrf.py'
# checkpoint_file = 'C:/_koray/test_data/space/aws/spacenet-6/2-MaksimovKA/sp_6__2_MaksimovKA.pth'


# config_file = 'configs/deeplabv3plus/deeplabv3plus_r101b-d8_769x769_80k_cityscapes.py'
# checkpoint_file = 'checkpoints/deeplabv3plus_r101b-d8_769x769_80k_cityscapes_20201226_205041-227cdf7c.pth'


# config_file = 'configs/emanet/emanet_r101-d8_512x1024_80k_cityscapes.py'
# checkpoint_file = 'checkpoints/emanet_r101-d8_512x1024_80k_cityscapes_20200901_100301-2d970745.pth'


def display(is_img, img_or_fn, results, wait=1, window_name="Netcad-NDU Segmentation Test"):
    result = results[0]

    if is_img:
        img0 = img_or_fn
    else:
        img0 = cv2.imread(img_or_fn)

    img = img0.astype(np.int32)
    palette = [None, [0, 255, 255]]
    for label, color in enumerate(palette):
        # color_seg[result == label, :] = color
        if color is not None:
            img[result == label, :] = color

    # # convert to BGR
    # color_seg = color_seg[..., ::-1]

    # img = img0 * 0.5 + color_seg * 0.5
    # img = img0 + color_seg
    img = img0 * 0.5 + img * 0.5
    img = img.astype(np.uint8)

    cv2.imshow(window_name, img)
    cv2.waitKey(wait)



# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')


for frame in ScreenCapturer(bbox=(200, 200, 200+1200, 200+1000)).get_frames():
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = inference_segmentor(model, frame)
    # model.show_result(frame, result, show=True, wait_time=1)
    display(True, frame, result)



# url = "https://youtu.be/ZORzsubQA_M"
# for frame in YoutubeVideoSource(url).get_frames():
#     result = inference_segmentor(model, frame)
#     model.show_result(frame, result, show=True, wait_time=1)
#     # display(True, frame, result)


# for img in enumerate_files("C:/_koray/test_data/space/test"):
for img_fn in enumerate_files("C:/_koray/test_datasssssssssssssssssssss/space/test/val"):
    result = inference_segmentor(model, img_fn)
    # model.show_result(img_fn, result, show=True, wait_time=1000)
    display(False, img_fn, result, wait=1000)

# video = mmcv.VideoReader('C:/_koray/test_data/driving.mp4')
# video = mmcv.VideoReader('C:/_koray/test_data/highway/highway_1600.mp4')
video = mmcv.VideoReader('C:/_koray/test_data/aerial/mexico.mp4')
# video = mmcv.VideoReader('C:/_koray/test_data/aerial/japan.mp4')
# video = mmcv.VideoReader('C:/_koray/test_data/aerial/china.mp4')
# video = mmcv.VideoReader('C:/_koray/test_data/aerial/barcelona.mp4')
# video = mmcv.VideoReader('C:/_koray/test_data/çanakkale0/meydan2.mp4')

for frame in video:
    result = inference_segmentor(model, frame)
    model.show_result(frame, result, show=True, wait_time=1)
    # display(True, frame, result)

# python tools/train.py ${CONFIG_FILE} [optional arguments]
# python tools/train.py
