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

###ok
# config_file = "C:/_koray/korhun/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_koray.py"
# checkpoint_file = "C:/_koray/korhun/mmsegmentation/data/space/work_dir_2/latest.pth"

# ###ok
# config_file = "C:/_koray/korhun/mmsegmentation/data/space/work_dir_pspnet_koray/pspnet_koray.py"
# checkpoint_file = "C:/_koray/korhun/mmsegmentation/data/space/work_dir_pspnet_koray/latest.pth"

config_file = "C:/_koray/korhun/mmsegmentation/data/space/work_dir_ocrnet_hr18_512x1024_160k_cityscapes_koray/ocrnet_hr18_512x1024_160k_cityscapes_koray.py"
checkpoint_file = "C:/_koray/korhun/mmsegmentation/data/space/work_dir_ocrnet_hr18_512x1024_160k_cityscapes_koray/latest.pth"



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



#########screen capture
for frame in ScreenCapturer(bbox=(200, 200, 200+1200, 200+1000)).get_frames():
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = inference_segmentor(model, frame)
    # model.show_result(frame, result, show=True, wait_time=1)
    display(True, frame, result)



##########youtube
# url = "https://youtu.be/ZORzsubQA_M"
# for frame in YoutubeVideoSource(url).get_frames():
#     result = inference_segmentor(model, frame)
#     model.show_result(frame, result, show=True, wait_time=1)
#     # display(True, frame, result)


# #########images
# # for img in enumerate_files("C:/_koray/test_data/space/test"):
# for img_fn in enumerate_files("C:/_koray/test_data/space/test/val"):
#     result = inference_segmentor(model, img_fn)
#     # model.show_result(img_fn, result, show=True, wait_time=1000)
#     display(False, img_fn, result, wait=1000)



##########video
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
