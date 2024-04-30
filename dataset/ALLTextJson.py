import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
from util.io import read_lines
import cv2
from pycocotools.coco import COCO
import scipy.io as scio


class ALLTextJson(TextDataset):
    def __init__(self, is_training=True, load_memory=False, transform=None):
        super().__init__(transform, is_training)
        self.is_training = is_training
        self.load_memory = load_memory
        image_root = [
            "data/MLT/train_images/",
            "data/SynthCurve/img_part1/emcs_imgs/",
            "data/SynthCurve/img_part2/syntext_word_eng/",
            "../FAST/data/SynthText/",
        ]
        gt_root = [
            "data/MLT/gts/",
            "data/SynthCurve/img_part1/train_poly_pos.json",
            "data/SynthCurve/img_part2/train_poly_pos.json",
            "../FAST/data/SynthText/gt.mat",
        ]
        image_list = []
        anno_list = []
        for path, gtpath in zip(image_root, gt_root):
            if ".json" in gtpath:
                imgfnames = sorted(os.listdir(path))
                image_list.extend([os.path.join(path, fname) for fname in imgfnames])
                coco_api = COCO(gtpath)
                img_ids = sorted(coco_api.imgs.keys())
                anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
                for anno in anns:
                    polygons = []
                    for label in anno:
                        poly = label["polys"]
                        poly = np.array(list(map(int, poly)))
                        poly = poly.reshape(-1, 2)
                        polygons.append(TextInstance(poly, "c", "word"))
                    anno_list.append(polygons)
            elif "SynthText" in gtpath:
                data = scio.loadmat(gtpath)
                imgfnames = data["imnames"][0]
                gts = data["wordBB"][0]
                image_list.extend([os.path.join(path, fname[0]) for fname in imgfnames])
                anno_list.extend([self.read_SynthText_gt(anno) for anno in gts])
            else:
                imgfnames = sorted(os.listdir(path))
                image_list.extend([os.path.join(path, fname) for fname in imgfnames])
                gtfnames = sorted(os.listdir(gtpath))
                anno_list.extend(
                    [self.read_txt(os.path.join(gtpath, fname)) for fname in gtfnames]
                )
        self.image_list = []
        self.anno_list = []
        for imgpath, gtpath in zip(image_list, anno_list):
            if ".jpg" in imgpath or ".png" in imgpath:
                self.image_list.append(imgpath)
                self.anno_list.append(gtpath)

    def read_txt(self, path):
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        polygons = []
        for line in lines:
            line = line.strip("\ufeff")
            line = line.replace("ï»¿", "")
            gt = line.split(",")
            poly = np.array(list(map(int, gt[:8]))).reshape(-1, 2)
            if gt[-1].strip() == "###":
                label = gt[-1].strip().replace("###", "#")
            else:
                label = "word"
            polygons.append(TextInstance(poly, "c", label))
        return polygons

    def read_SynthText_gt(self, bboxes):
        bboxes = np.array(bboxes)
        bboxes = np.reshape(
            bboxes, (tuple(bboxes.shape)[0], tuple(bboxes.shape)[1], -1)
        )
        bboxes = bboxes.transpose(2, 1, 0)
        polygons = []
        for bbox in bboxes:
            points = np.rint(bbox)
            polygon = TextInstance(points, "c", "abc")
            polygons.append(polygon)
        return polygons

    def load_img_gt(self, item):
        image_path = self.image_list[item]
        image_id = image_path.split("/")[-1]
        image = pil_load_img(image_path)
        try:
            assert tuple(image.shape)[-1] == 3
        except:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)
        polygons = self.anno_list[item]
        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id
        data["image_path"] = image_path
        return data

    def __getitem__(self, item):
        data = self.load_img_gt(item)
        return self.get_training_data(
            data["image"],
            data["polygons"],
            image_id=data["image_id"],
            image_path=data["image_path"],
        )

    def __len__(self):
        return len(self.image_list)


if __name__ == "__main__":
    from util.augmentation import Augmentation
    from util.misc import regularize_sin_cos
    from util.pbox import bbox_transfor_inv, minConnectPath
    from util import canvas as cav
    import time

    means = 0.485, 0.456, 0.406
    stds = 0.229, 0.224, 0.225
    transform = Augmentation(size=640, mean=means, std=stds)
    trainset = MLTTextJson(is_training=True, transform=transform)
    print(len(trainset.image_list))
    print(len(trainset.anno_list))
