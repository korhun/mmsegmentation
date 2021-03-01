# import file_helper
from fnmatch import fnmatch

import cv2
import numpy as np
from typing import AnyStr

# # Check Pytorch installation
import torch, torchvision

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


def put_text(img, text_, center, color=None, font_scale=0.5, thickness=1, back_color=None):
    if back_color is None:
        back_color = [0, 0, 0]
    if color is None:
        color = [255, 255, 255]
    y = center[1]
    # font = cv2.FONT_HERSHEY_COMPLEX
    font = cv2.FONT_HERSHEY_DUPLEX
    coor = (int(center[0] + 5), int(y))
    cv2.putText(img=img, text=text_, org=coor,
                fontFace=font, fontScale=font_scale, color=back_color, lineType=cv2.LINE_AA,
                thickness=thickness + 2)
    cv2.putText(img=img, text=text_, org=coor,
                fontFace=font, fontScale=font_scale, color=color,
                lineType=cv2.LINE_AA, thickness=thickness)


# https://github.com/korhun/mmsegmentation/tree/master/configs/

###ok
# config_file = "D:/_koray/korhun/mmsegmentation/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_80k_cityscapes_koray.py"
# checkpoint_file = "D:/_koray/korhun/mmsegmentation/data/space/work_dir_2/latest.pth"

# ###ok rotterdam
# config_file = "D:/_koray/korhun/mmsegmentation/data/space/work_dir_pspnet_koray/pspnet_koray.py"
# checkpoint_file = "D:/_koray/korhun/mmsegmentation/data/space/work_dir_pspnet_koray/latest.pth"

# # ###ok ++  Bakü
# config_file = "D:/_koray/korhun/mmsegmentation/data/space/work_dir_ocrnet_hr18_512x1024_160k_cityscapes_koray/ocrnet_hr18_512x1024_160k_cityscapes_koray.py"
# checkpoint_file = "D:/_koray/korhun/mmsegmentation/data/space/work_dir_ocrnet_hr18_512x1024_160k_cityscapes_koray/latest.pth"


# # # yollar
# config_file = "D:/_koray/korhun/mmsegmentation/data/space/work_dir_ocrnet_hr48_512x1024_160k_custom_koray_SV3_roads/ocrnet_hr48_512x1024_160k_custom_koray.py"
# checkpoint_file = "D:/_koray/korhun/mmsegmentation/data/space/work_dir_ocrnet_hr48_512x1024_160k_custom_koray_SV3_roads/latest.pth"


# # SN7_buildings 18
# config_file = "D:/_koray/korhun/mmsegmentation/data/space/SN7_buildings_ocrnet_hr18_512x1024_160k_cityscapes_koray/ocrnet_hr18_512x1024_160k_cityscapes_koray.py"
# checkpoint_file = "D:/_koray/korhun/mmsegmentation/data/space/SN7_buildings_ocrnet_hr18_512x1024_160k_cityscapes_koray/latest.pth"

# # SN7_buildings 48
# config_file = "D:/_koray/korhun/mmsegmentation/data/space/SN7_buildings_ocrnet_hr48_512x1024_160k_custom_koray/ocrnet_hr48_512x1024_160k_custom_koray.py"
# checkpoint_file = "D:/_koray/korhun/mmsegmentation/data/space/SN7_buildings_ocrnet_hr48_512x1024_160k_custom_koray/latest.pth"
# SN7_buildings 48
config_file = "D:/_koray/train_datasets/space/weights/SN7_buildings/ocrnet_hr48_512x1024_160k/ocrnet_hr48_512x1024_160k_custom_koray.py"
checkpoint_file = "D:/_koray/train_datasets/space/weights/SN7_buildings/ocrnet_hr48_512x1024_160k/iter_148000.pth"



_display_inited = False


def display(is_img, img_or_fn, results, wait=1, window_name="Netcad-NDU Segmentation Test"):
    res = results[0]

    if is_img:
        img0 = img_or_fn
    else:
        img0 = cv2.imread(img_or_fn)

    img = img0.astype(np.int32)

    # palette = [None, [0, 255, 255]]
    # for label, color in enumerate(palette):
    #     if color is not None:
    #         img[res == label, :] = color
    # img = img0 * 0.5 + img * 0.5

    palette = [[0,0,0], [0, 255, 255]]
    for label, color in enumerate(palette):
        if color is not None:
            img[res == label, :] = color
    img = img0 * 0.5 + img * 1.5


    img = img.astype(np.uint8)

    non_zero = np.count_nonzero(res)
    non_zero_per = non_zero * 100 / float(res.shape[0] * res.shape[1])
    put_text(img, "BI: %{:0.2f}".format(non_zero_per), [img.shape[1] - 220, 50], color=[0, 0, 255], font_scale=1, thickness=1, back_color=[0, 0, 0])

    cv2.imshow(window_name, img)
    global _display_inited
    if not _display_inited:
        _display_inited = True
        # cv2.moveWindow(window_name, -1800, 50) #home
        # cv2.moveWindow(window_name, -1800, 220) #office
    cv2.waitKey(wait)


# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')



#########tiff
# for img_fn in enumerate_files("D:/_koray/test_data/space/test"):
for img_fn in enumerate_files("D:/_koray/test_data/space/SN7_buildings_test_public/test_public/L15-0369E-1244N_1479_3214_13/images_masked"):


    # tiff = tiff_helper.open_tiff(img_fn)
    # img = cv2.imread(img_fn)
    # cv2.imshow("aaa", img)
    # cv2.waitKey(0)


    result = inference_segmentor(model, img_fn)
    # model.show_result(img_fn, result, show=True, wait_time=1000)
    display(False, img_fn, result, wait=1000)


# #########screen capture
# for frame in ScreenCapturer(bbox=(400, 200, 400 + 1200, 200 + 1000)).get_frames():
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = inference_segmentor(model, frame)
#     # model.show_result(frame, result, show=True, wait_time=1)
#     display(True, frame, result)

##########youtube
# url = "https://youtu.be/ZORzsubQA_M"
# for frame in YoutubeVideoSource(url).get_frames():
#     result = inference_segmentor(model, frame)
#     model.show_result(frame, result, show=True, wait_time=1)
#     # display(True, frame, result)

#
# #########images
# for img_fn in enumerate_files("D:/_koray/test_data/space/test"):
# # for img_fn in enumerate_files("D:/_koray/test_data/space/test/val"):
# # for img_fn in enumerate_files("D:/_koray/train_datasets/spacenet/mm/building/global/rgb"):
#     result = inference_segmentor(model, img_fn)
#     # model.show_result(img_fn, result, show=True, wait_time=1000)
#     display(False, img_fn, result, wait=1000)

##########video
# video = mmcv.VideoReader('D:/_koray/test_data/driving.mp4')
# video = mmcv.VideoReader('D:/_koray/test_data/highway/highway_1600.mp4')
video = mmcv.VideoReader('D:/_koray/test_data/aerial/mexico.mp4')
# video = mmcv.VideoReader('D:/_koray/test_data/aerial/japan.mp4')
# video = mmcv.VideoReader('D:/_koray/test_data/aerial/china.mp4')
# video = mmcv.VideoReader('D:/_koray/test_data/aerial/barcelona.mp4')
# video = mmcv.VideoReader('D:/_koray/test_data/çanakkale0/meydan2.mp4')
for frame in video:
    result = inference_segmentor(model, frame)
    model.show_result(frame, result, show=True, wait_time=1)
    # display(True, frame, result)

# python tools/train.py ${CONFIG_FILE} [optional arguments]
# python tools/train.py
