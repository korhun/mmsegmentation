# import file_helper
from fnmatch import fnmatch

from typing import AnyStr

import io

import shutil
import cv2
import os
import shapely.wkt
import numpy as np


# import file_helper


def delete_dir(path_to_dir):
    shutil.rmtree(path_to_dir)


def create_dir(dir_name, parents=True, exist_ok=True):
    from pathlib import Path
    Path(dir_name).mkdir(parents=parents, exist_ok=exist_ok)


def read_lines(filename, encoding="utf-8"):
    with io.open(filename, mode="r", encoding=encoding) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
        return content


def path_join(a: AnyStr, *paths: AnyStr) -> AnyStr:
    return os.path.join(a, *paths).replace("/", os.path.sep)


def wildcard(txt, pattern, case_insensitive=True):
    if txt == pattern:
        return True
    else:
        return fnmatch(txt.lower(), pattern.lower()) if case_insensitive else fnmatch(txt, pattern)


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


end_txt = "                                 \r"


def _read_image_uint8(image_fn):
    return cv2.imread(image_fn)


def _read_image_uint16(image_fn):
    img = cv2.imread(image_fn, -1)
    # img = cv2.imread( image_fn, cv2.IMREAD_ANYDEPTH)
    # img = cv2.imread( image_fn, cv2.IMREAD_ANYCOLOR)
    # res = cv2.convertScaleAbs(img)
    # res = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # res = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    res = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    # res[:, :, 0] = cv2.equalizeHist(res[:, :, 0])
    res[:, :, 0] = cv2.equalizeHist(res[:, :, 0])
    res[:, :, 1] = cv2.equalizeHist(res[:, :, 1])
    res[:, :, 2] = cv2.equalizeHist(res[:, :, 2])
    # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

    # img_yuv = cv2.cvtColor(res, cv2.COLOR_BGR2YUV)
    # img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # res = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)

    # res[:, :, 0] = cv2.equalizeHist(res[:, :, 0])

    return res


def convert_spacenet_dataset_to_mm():
    # dir_images = "C:/_koray/train_datasets/spacenet/SN6_buildings/train/AOI_11_Rotterdam/PS-RGB"
    # image_prefix = "SN6_Train_AOI_11_Rotterdam_PS-RGB_"
    # fn_summary = "C:/_koray/train_datasets/spacenet/SN6_buildings/train/AOI_11_Rotterdam/SummaryData/SN6_Train_AOI_11_Rotterdam_Buildings.csv"
    # image_suffix = ".tif"
    # dir_out_rgb = "C:/_koray/train_datasets/spacenet/SN6_buildings/train/AOI_11_Rotterdam/mm/rgb"
    # dir_out_map = "C:/_koray/train_datasets/spacenet/SN6_buildings/train/AOI_11_Rotterdam/mm/map"

    # dir_images = "C:/_koray/train_datasets/spacenet/mm/building/global/tiff"
    # image_prefix = ""
    # fn_summary = "C:/_koray/train_datasets/spacenet/mm/building/global/sn7_train_ground_truth_pix.csv"
    # image_suffix = ".tif"
    # dir_out_rgb = "C:/_koray/train_datasets/spacenet/mm/building/global/rgb"
    # dir_out_map = "C:/_koray/train_datasets/spacenet/mm/building/global/map"

    # delete_dir_out = True
    delete_dir_out = False
    # preview = True
    preview = False

    dir_out_rgb = "C:/_koray/train_datasets/spacenet/mm/roads/SV3_roads/rgb"
    dir_out_map = "C:/_koray/train_datasets/spacenet/mm/roads/SV3_roads/map"
    read_rgb = _read_image_uint16

    image_replace = ["_img", "_PS-RGB_img"]
    configs = [
        {
            "dir_images": "C:/_koray/train_datasets/spacenet/SV3_roads/2_Vegas/PS-RGB",
            "image_prefix": "",
            "fn_summary": "C:/_koray/train_datasets/spacenet/SV3_roads/2_Vegas/train_AOI_2_Vegas_geojson_roads_speed_wkt_weighted_raw.csv",
            "image_suffix": ".tif",
            "image_replace": image_replace,
            "dir_out_rgb": dir_out_rgb,
            "dir_out_map": dir_out_map,
            "line_thickness": 28
        },
        {
            "dir_images": "C:/_koray/train_datasets/spacenet/SV3_roads/3_Paris/PS-RGB",
            "image_prefix": "",
            "fn_summary": "C:/_koray/train_datasets/spacenet/SV3_roads/3_Paris/train_AOI_3_Paris_geojson_roads_speed_wkt_weighted_raw.csv",
            "image_suffix": ".tif",
            "image_replace": image_replace,
            "dir_out_rgb": dir_out_rgb,
            "dir_out_map": dir_out_map,
            "line_thickness": 20
        },
        {
            "dir_images": "C:/_koray/train_datasets/spacenet/SV3_roads/4_Shanghai/PS-RGB",
            "image_prefix": "",
            "fn_summary": "C:/_koray/train_datasets/spacenet/SV3_roads/4_Shanghai/train_AOI_4_Shanghai_geojson_roads_speed_wkt_weighted_raw.csv",
            "image_suffix": ".tif",
            "image_replace": image_replace,
            "dir_out_rgb": dir_out_rgb,
            "dir_out_map": dir_out_map,
            "line_thickness": 18
        },
        {
            "dir_images": "C:/_koray/train_datasets/spacenet/SV3_roads/5_Khartoum/PS-RGB",
            "image_prefix": "",
            "fn_summary": "C:/_koray/train_datasets/spacenet/SV3_roads/5_Khartoum/train_AOI_5_Khartoum_geojson_roads_speed_wkt_weighted_raw.csv",
            "image_suffix": ".tif",
            "image_replace": image_replace,
            "dir_out_rgb": dir_out_rgb,
            "dir_out_map": dir_out_map,
            "line_thickness": 15
        }
    ]
    check_out_dir = True
    for config in configs:
        _convert_spacenet_dataset_to_mm(config, check_out_dir, delete_dir_out, preview, read_rgb)
        delete_dir_out = False
        check_out_dir = False


def _convert_spacenet_dataset_to_mm(config, check_out_dir, delete_dir_out, preview, read_rgb):
    dir_images = config["dir_images"]
    image_prefix = config["image_prefix"]
    fn_summary = config["fn_summary"]
    image_suffix = config["image_suffix"]
    image_replace = config["image_replace"]
    dir_out_rgb = config["dir_out_rgb"]
    dir_out_map = config["dir_out_map"]
    line_thickness = config["line_thickness"]

    out_suffix_rgb = '_rgb.jpg'
    out_suffix_map = '_map.jpg'
    map_color = 1

    if check_out_dir:
        if delete_dir_out:
            for d in [dir_out_rgb, dir_out_map]:
                if d is not None and os.path.isdir(d):
                    delete_dir(d)
        for d in [dir_out_rgb, dir_out_map]:
            if d is not None and not os.path.isdir(d):
                create_dir(d)
            else:
                raise Exception("Directory exists! {}".format(d))

    i = 0
    image_id = 0
    last_fn_img_map = None
    img_map = None
    img_preview = None
    for line in read_lines(fn_summary):
        try:
            items = line.split(",")
            image_id = items[0]
            if image_replace is not None:
                image_id = image_id.replace(image_replace[0], image_replace[1])
            image_fn = path_join(dir_images, image_prefix + image_id + image_suffix)
            i += 1
            print("{} - {}".format(i, image_id), end=end_txt)
            if os.path.isfile(image_fn):
                fn_img_rgb = path_join(dir_out_rgb, image_id + out_suffix_rgb)
                fn_img_map = path_join(dir_out_map, image_id + out_suffix_map)
                if not os.path.isfile(fn_img_rgb):
                    img_rgb = read_rgb(image_fn)
                    cv2.imwrite(fn_img_rgb, img_rgb)
                    if last_fn_img_map is not None:
                        if not preview:
                            cv2.imwrite(last_fn_img_map, img_map)
                        else:
                            cv2.imwrite(last_fn_img_map, img_preview)
                    img_map = np.zeros([img_rgb.shape[0], img_rgb.shape[1], 1], dtype=np.uint8)
                    if preview:
                        img_preview = img_rgb.copy()
                    last_fn_img_map = fn_img_map
                # else:
                #     img_map = cv2.imread(fn_img_map, cv2.IMREAD_GRAYSCALE)

                # if "POLYGON EMPTY" not in line:
                if " EMPTY" not in line:
                    last_index = line.rindex('"')
                    pl_txt = line[line.index('"') + 1:last_index]

                    geom = shapely.wkt.loads(pl_txt)
                    if geom is not None:
                        coors = []
                        if geom.geom_type.lower() == "polygon":
                            for c in geom.exterior.coords:
                                coors.append([int(c[0]), int(c[1])])
                            int_coords = lambda x: np.array(x).round().astype(np.int32)
                            exterior = [int_coords(coors)]
                            # exterior = [int_coords(polygon.exterior.coords)]
                            cv2.fillPoly(img_map, exterior, color=map_color)
                            if preview:
                                cv2.fillPoly(img_preview, exterior, color=[0, 255, 255])

                        elif geom.geom_type.lower() == "linestring":
                            # w_txts = line[last_index+2:].split(",")
                            # w = int(float(w_txts[1]))
                            w = line_thickness

                            for c in geom.coords:
                                coors.append([int(c[0]), int(c[1])])
                            int_coords = lambda x: np.array(x).round().astype(np.int32)
                            pnts = [int_coords(coors)]

                            cv2.polylines(img_map, pnts, isClosed=False, color=map_color, thickness=w)
                            if preview:
                                cv2.polylines(img_preview, pnts, isClosed=False, color=[0, 255, 255], thickness=line_thickness)

                        if preview:
                            cv2.imshow("preview", img_preview)
                            cv2.waitKey(20)

                    # alpha = 1
                    # image = img_map
                    # overlay = image.copy()
                    # cv2.fillPoly(overlay, exterior, color=map_color)
                    # cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                    # # cv2.imshow("Polygon", image)
                    # # cv2.waitKey(1)

                # cv2.imwrite(fn_img_map, img_map)

            else:
                print("Cannot find file: " + image_id)
        except Exception as e:
            print("Error - i:{}  image_id:{}   err:{} ".format(str(i), image_id, str(e)))
    if last_fn_img_map is not None:
        cv2.imwrite(last_fn_img_map, img_map)


convert_spacenet_dataset_to_mm()

#
# def deneme():
#     # dir_input = "C:/_koray/korhun/mmsegmentation/data/space/ann/train_rgb"
#     # dir_output = "C:/_koray/korhun/mmsegmentation/data/space/ann/train"
#
#     dir_input = "C:/_koray/korhun/mmsegmentation/data/space/ann/val_rgb"
#     dir_output = "C:/_koray/korhun/mmsegmentation/data/space/ann/val"
#
#     first_delete_out = True
#     if first_delete_out:
#         if os.path.isdir(dir_output):
#             delete_dir(dir_output)
#     if not os.path.isdir(dir_output):
#         create_dir(dir_output)
#
#     i = 0
#     for fn in enumerate_files(dir_input):
#         try:
#             i += 1
#             print("{} - {}".format(i, fn), end=end_txt)
#             dir_name, name, extension = file_helper.get_file_name_extension(fn)
#             fn_out = file_helper.path_join(dir_output, name+extension)
#
#             bgr = cv2.imread(fn)
#             gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
#             mask = cv2.inRange(gray, 1, 255)
#             img = gray.copy()
#             img[mask != 0] = 1
#
#             cv2.imwrite(fn_out, img)
#
#         except Exception as e:
#             print("Error - i:{}  fn:{}   err:{} ".format(str(i), fn, str(e)))
#

# deneme()
