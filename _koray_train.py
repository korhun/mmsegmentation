import mmcv
import os.path as osp
import numpy as np
from PIL import Image

# help: https://colab.research.google.com/github/open-mmlab/mmsegmentation/blob/master/demo/MMSegmentation_Tutorial.ipynb#scrollTo=WnGZfribFHCx
#
#
# # convert dataset annotation to semantic segmentation map
# data_root = 'iccv09Data'
# img_dir = 'images'
# ann_dir = 'labels'
# # define class and plaette for better visualization
# classes = ('background', 'building')
# palette = [[0, 0, 0], [192, 128, 128]]
# for file in mmcv.scandir(osp.join(data_root, ann_dir), suffix='.regions.txt'):
#   seg_map = np.loadtxt(osp.join(data_root, ann_dir, file)).astype(np.uint8)
#   seg_img = Image.fromarray(seg_map).convert('P')
#   seg_img.putpalette(np.array(palette, dtype=np.uint8))
#   seg_img.save(osp.join(data_root, ann_dir, file.replace('.regions.txt',
#                                                          '.png')))


# region convert
import file_helper
import cv2
import os
import shapely.wkt

end_txt = "                   \r"


def convert_spacenet_dataset_to_mm():
    dir_images = "C:/_koray/train_datasets/spacenet/SN6_buildings/train/AOI_11_Rotterdam/PS-RGB"
    image_prefix = "SN6_Train_AOI_11_Rotterdam_PS-RGB_"
    image_suffix = ".tif"
    fn_summary = "C:/_koray/train_datasets/spacenet/SN6_buildings/train/AOI_11_Rotterdam/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv"

    dir_out_rgb = "C:/_koray/train_datasets/spacenet/SN6_buildings/train/AOI_11_Rotterdam/mm/rgb"
    dir_out_map = "C:/_koray/train_datasets/spacenet/SN6_buildings/train/AOI_11_Rotterdam/mm/map"
    out_suffix_rgb = '_rgb.jpg'
    out_suffix_map = '_map.jpg'
    map_color = (255, 255, 255)
    first_delete_out = True
    # first_delete_out = False

    if first_delete_out:
        for d in [dir_out_rgb, dir_out_map]:
            if os.path.isdir(d):
                file_helper.delete_dir(d)
    for d in [dir_out_rgb, dir_out_map]:
        if not os.path.isdir(d):
            file_helper.create_dir(d)

    i = 0
    image_id=0
    for line in file_helper.read_lines(fn_summary):
        try:
            items = line.split(",")
            image_id = items[0]
            image_fn = file_helper.path_join(dir_images, image_prefix + image_id + image_suffix)
            i += 1
            print("{} - {}".format(i, image_id), end=end_txt)
            if os.path.isfile(image_fn):
                fn_img_rgb = file_helper.path_join(dir_out_rgb, image_id + out_suffix_rgb)
                fn_img_map = file_helper.path_join(dir_out_map, image_id + out_suffix_map)
                if not os.path.isfile(fn_img_rgb):
                    img_rgb = cv2.imread(image_fn)
                    cv2.imwrite(fn_img_rgb, img_rgb)
                    img_map = img_rgb
                    img_map[:] = [0, 0, 0]
                else:
                    img_map = cv2.imread(fn_img_map)

                if "POLYGON EMPTY" not in line:
                    pl_txt = line[line.index('"') + 1:line.rindex('"')]

                    polygon = shapely.wkt.loads(pl_txt)
                    int_coords = lambda x: np.array(x).round().astype(np.int32)
                    exterior = [int_coords(polygon.exterior.coords)]

                    alpha = 1
                    image = img_map
                    overlay = image.copy()
                    cv2.fillPoly(overlay, exterior, color=map_color)
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                    # cv2.imshow("Polygon", image)
                    # cv2.waitKey(1)

                cv2.imwrite(fn_img_map, img_map)
            else:
                print("Cannot find file: " + image_id)
        except Exception as e:
            print("Error - i:{}  image_id:{}   err:{} ".format(str(i), image_id, str(e)))


convert_spacenet_dataset_to_mm()
# endregion
