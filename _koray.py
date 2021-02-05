import file_helper
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

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
config_file = 'configs/ocrnet/ocrnet_hr48_512x1024_160k_cityscapes.py'
checkpoint_file = 'checkpoints/ocrnet_hr48_512x1024_160k_cityscapes_20200602_191037-dfbf1b0c.pth'
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





# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# # test a single image and show the results
# img = 'test/demo.png'  # or img = mmcv.imread(img), which will only load it once
# result = inference_segmentor(model, img)
# # visualize the results in a new window
# model.show_result(img, result, show=True)
# # or save the visualization results to image files
# model.show_result(img, result, out_file='result.jpg')


for img in file_helper.enumerate_files("C:/_koray/test_data/space/test"):
   result = inference_segmentor(model, img)
   model.show_result(img, result, show=True, wait_time=2000)


video = mmcv.VideoReader('C:/_koray/test_data/driving.mp4')
# video = mmcv.VideoReader('C:/_koray/test_data/highway/highway_1600.mp4')
# video = mmcv.VideoReader('C:/_koray/test_data/aerial/mexico.mp4')
# video = mmcv.VideoReader('C:/_koray/test_data/çanakkale0/meydan2.mp4')

for frame in video:
   result = inference_segmentor(model, frame)
   model.show_result(frame, result, show=True, wait_time=1)



# python tools/train.py ${CONFIG_FILE} [optional arguments]
# python tools/train.py