import os
import numpy as np
from dataset.data_util import pil_load_img
from dataset.dataload import TextDataset, TextInstance
from util.io import read_lines
import lanms
import cv2


class Ctw1500Text(TextDataset):
    def __init__(
        self,
        data_root,
        is_training=True,
        load_memory=False,
        transform=None,
        ignore_list=None,
    ):
        super().__init__(transform, is_training)
        self.data_root = data_root
        self.is_training = is_training
        self.load_memory = load_memory
        self.image_root = os.path.join(
            data_root, "train" if is_training else "test", "text_image"
        )
        self.annotation_root = os.path.join(
            data_root, "train" if is_training else "test", "text_label_circum"
        )
        self.image_list = os.listdir(self.image_root)
        self.annotation_list = [
            "{}".format(img_name.replace(".jpg", "")) for img_name in self.image_list
        ]
        if self.load_memory:
            self.datas = list()
            for item in range(len(self.image_list)):
                self.datas.append(self.load_img_gt(item))

    @staticmethod
    def parse_carve_txt(gt_path):
        """
        .mat file parser
        :param gt_path: (str), mat file path
        :return: (list), TextInstance
        """
        lines = read_lines(gt_path + ".txt")
        polygons = []
        for line in lines:
            gt = list(map(int, line.split(",")))
            pts = np.stack([gt[4::2], gt[5::2]]).T.astype(np.int32)
            pts[:, 0] = pts[:, 0] + gt[0]
            pts[:, 1] = pts[:, 1] + gt[1]
            polygons.append(TextInstance(pts, "c", "**"))
        return polygons

    def load_img_gt(self, item):
        image_id = self.image_list[item]
        image_path = os.path.join(self.image_root, image_id)
        image = pil_load_img(image_path)
        try:
            h, w, c = tuple(image.shape)
            assert c == 3
        except:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.array(image)
        annotation_id = self.annotation_list[item]
        annotation_path = os.path.join(self.annotation_root, annotation_id)
        polygons = self.parse_carve_txt(annotation_path)
        data = dict()
        data["image"] = image
        data["polygons"] = polygons
        data["image_id"] = image_id.split("/")[-1]
        data["image_path"] = image_path
        return data

    def __getitem__(self, item):
        if self.load_memory:
            data = self.datas[item]
        else:
            data = self.load_img_gt(item)
        if self.is_training:
            return self.get_training_data(
                data["image"],
                data["polygons"],
                image_id=data["image_id"],
                image_path=data["image_path"],
            )
        else:
            return self.get_test_data(
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
    trainset = Ctw1500Text(
        data_root="../data/ctw1500", is_training=True, transform=transform
    )
    for idx in range(0, len(trainset)):
        t0 = time.time()
        (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi) = (
            trainset[idx]
        )
        (img, train_mask, tr_mask, tcl_mask, radius_map, sin_map, cos_map, gt_roi) = (
            map(
                lambda x: x.cpu().numpy(),
                (
                    img,
                    train_mask,
                    tr_mask,
                    tcl_mask,
                    radius_map,
                    sin_map,
                    cos_map,
                    gt_roi,
                ),
            )
        )
        img = img.transpose(1, 2, 0)
        img = ((img * stds + means) * 255).astype(np.uint8)
        print(idx, tuple(img.shape))
        top_map = radius_map[:, :, 0]
        bot_map = radius_map[:, :, 1]
        print(tuple(radius_map.shape))
        sin_map, cos_map = regularize_sin_cos(sin_map, cos_map)
        ret, labels = cv2.connectedComponents(
            tcl_mask[:, :, 0].astype(np.uint8), connectivity=8
        )
        cv2.imshow(
            "labels0",
            cav.heatmap(np.array(labels * 255 / np.max(labels), dtype=np.uint8)),
        )
        print(np.sum(tcl_mask[:, :, 1]))
        t0 = time.time()
        for bbox_idx in range(1, ret):
            bbox_mask = labels == bbox_idx
            text_map = tcl_mask[:, :, 0] * bbox_mask
            boxes = bbox_transfor_inv(
                radius_map, sin_map, cos_map, text_map, wclip=(2, 8)
            )
            boxes = lanms.merge_quadrangle_n9(boxes.astype("float32"), 0.25)
            boxes = boxes[:, :8].reshape((-1, 4, 2)).astype(np.int32)
            if tuple(boxes.shape)[0] > 1:
                center = np.mean(boxes, axis=1).astype(np.int32).tolist()
                paths, routes_path = minConnectPath(center)
                boxes = boxes[routes_path]
                top = np.mean(boxes[:, 0:2, :], axis=1).astype(np.int32).tolist()
                bot = np.mean(boxes[:, 2:4, :], axis=1).astype(np.int32).tolist()
                boundary_point = top + bot[::-1]
                for ip, pp in enumerate(top):
                    if ip == 0:
                        color = 0, 255, 255
                    elif ip == len(top) - 1:
                        color = 255, 255, 0
                    else:
                        color = 0, 0, 255
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, color, -1)
                for ip, pp in enumerate(bot):
                    if ip == 0:
                        color = 0, 255, 255
                    elif ip == len(top) - 1:
                        color = 255, 255, 0
                    else:
                        color = 0, 255, 0
                    cv2.circle(img, (int(pp[0]), int(pp[1])), 2, color, -1)
                cv2.drawContours(img, [np.array(boundary_point)], -1, (0, 255, 255), 1)
        cv2.imshow("imgs", img)
        cv2.imshow(
            "", cav.heatmap(np.array(labels * 255 / np.max(labels), dtype=np.uint8))
        )
        cv2.imshow(
            "tr_mask",
            cav.heatmap(np.array(tr_mask * 255 / np.max(tr_mask), dtype=np.uint8)),
        )
        cv2.imshow(
            "tcl_mask",
            cav.heatmap(
                np.array(
                    tcl_mask[:, :, 1] * 255 / np.max(tcl_mask[:, :, 1]), dtype=np.uint8
                )
            ),
        )
        cv2.waitKey(0)
