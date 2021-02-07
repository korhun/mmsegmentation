# import file_helper
from fnmatch import fnmatch

from typing import AnyStr

import io

import shutil
import cv2
import os
import shapely.wkt
import numpy as np

import file_helper


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
    # first_delete_out = True
    first_delete_out = False

    if first_delete_out:
        for d in [dir_out_rgb, dir_out_map]:
            if os.path.isdir(d):
                delete_dir(d)
    for d in [dir_out_rgb, dir_out_map]:
        if not os.path.isdir(d):
            create_dir(d)

    i = 0
    image_id = 0
    for line in read_lines(fn_summary):
        try:
            items = line.split(",")
            image_id = items[0]
            image_fn = path_join(dir_images, image_prefix + image_id + image_suffix)
            i += 1
            print("{} - {}".format(i, image_id), end=end_txt)
            if os.path.isfile(image_fn):
                fn_img_rgb = path_join(dir_out_rgb, image_id + out_suffix_rgb)
                fn_img_map = path_join(dir_out_map, image_id + out_suffix_map)
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


# convert_spacenet_dataset_to_mm()


def deneme():
    # dir_input = "C:/_koray/korhun/mmsegmentation/data/space/ann/train_rgb"
    # dir_output = "C:/_koray/korhun/mmsegmentation/data/space/ann/train"

    dir_input = "C:/_koray/korhun/mmsegmentation/data/space/ann/val_rgb"
    dir_output = "C:/_koray/korhun/mmsegmentation/data/space/ann/val"

    first_delete_out = True
    if first_delete_out:
        if os.path.isdir(dir_output):
            delete_dir(dir_output)
    if not os.path.isdir(dir_output):
        create_dir(dir_output)

    i = 0
    for fn in enumerate_files(dir_input):
        try:
            i += 1
            print("{} - {}".format(i, fn), end=end_txt)
            dir_name, name, extension = file_helper.get_file_name_extension(fn)
            fn_out = file_helper.path_join(dir_output, name+extension)

            bgr = cv2.imread(fn)
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            mask = cv2.inRange(gray, 1, 255)
            img = gray.copy()
            img[mask != 0] = 1

            cv2.imwrite(fn_out, img)

        except Exception as e:
            print("Error - i:{}  fn:{}   err:{} ".format(str(i), fn, str(e)))


# deneme()
