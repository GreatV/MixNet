import paddle
from cfglib.config import config as cfg
from network.Reg_loss import PolyMatchingLoss
from .emb_loss import EmbLoss_v2
from .overlap_loss import overlap_loss
from paddle_msssim import SSIM
import cv2


class TextLoss(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.MSE_loss = paddle.nn.MSELoss(reduction="none")
        self.BCE_loss = paddle.nn.BCELoss(reduction="none")
        self.PolyMatchingLoss = PolyMatchingLoss(cfg.num_points, cfg.device)
        if cfg.mid:
            self.midPolyMatchingLoss = PolyMatchingLoss(cfg.num_points // 2, cfg.device)
        self.ssim = SSIM
        self.overlap_loss = overlap_loss()

    @staticmethod
    def single_image_loss(pre_loss, loss_label):
        batch_size = tuple(pre_loss.shape)[0]
        sum_loss = paddle.mean(x=pre_loss.view(-1)) * 0
        pre_loss = pre_loss.view(batch_size, -1)
        loss_label = loss_label.view(batch_size, -1)
        eps = 0.001
        for i in range(batch_size):
            average_number = 0
            positive_pixel = len(pre_loss[i][loss_label[i] >= eps])
            average_number += positive_pixel
            if positive_pixel != 0:
                posi_loss = paddle.mean(x=pre_loss[i][loss_label[i] >= eps])
                sum_loss += posi_loss
                if len(pre_loss[i][loss_label[i] < eps]) < 3 * positive_pixel:
                    nega_loss = paddle.mean(x=pre_loss[i][loss_label[i] < eps])
                    average_number += len(pre_loss[i][loss_label[i] < eps])
                else:
                    nega_loss = paddle.mean(
                        x=paddle.topk(
                            k=3 * positive_pixel, x=pre_loss[i][loss_label[i] < eps]
                        )[0]
                    )
                    average_number += 3 * positive_pixel
                sum_loss += nega_loss
            else:
                nega_loss = paddle.mean(x=paddle.topk(k=100, x=pre_loss[i])[0])
                average_number += 100
                sum_loss += nega_loss
        return sum_loss / batch_size

    def cls_ohem(self, predict, target, train_mask, negative_ratio=3.0):
        pos = (target * train_mask).astype(dtype="bool")
        neg = ((1 - target) * train_mask).astype(dtype="bool")
        n_pos = pos.astype(dtype="float32").sum()
        if n_pos.item() > 0:
            loss_pos = self.BCE_loss(predict[pos], target[pos]).sum()
            loss_neg = self.BCE_loss(predict[neg], target[neg])
            n_neg = min(
                int(neg.astype(dtype="float32").sum().item()),
                int(negative_ratio * n_pos.astype(dtype="float32")),
            )
        else:
            loss_pos = paddle.to_tensor(data=0.0)
            loss_neg = self.BCE_loss(predict[neg], target[neg])
            n_neg = 100
        loss_neg, _ = paddle.topk(k=n_neg, x=loss_neg)
        return (loss_pos + loss_neg.sum()) / (n_pos + n_neg).astype(dtype="float32")

    @staticmethod
    def loss_calc_flux(pred_flux, gt_flux, weight_matrix, mask, train_mask):
        gt_flux = (
            0.999999 * gt_flux / (gt_flux.norm(p=2, axis=1).unsqueeze(axis=1) + 0.001)
        )
        norm_loss = (
            weight_matrix
            * paddle.mean(x=(pred_flux - gt_flux) ** 2, axis=1)
            * train_mask
        )
        norm_loss = norm_loss.sum(axis=-1).mean()
        mask = train_mask * mask
        pred_flux = (
            0.999999
            * pred_flux
            / (pred_flux.norm(p=2, axis=1).unsqueeze(axis=1) + 0.001)
        )
        angle_loss = 1 - paddle.nn.functional.cosine_similarity(
            x1=pred_flux, x2=gt_flux, axis=1
        )
        angle_loss = angle_loss[mask].mean()
        return norm_loss, angle_loss

    @staticmethod
    def get_poly_energy(energy_field, img_poly, ind, h, w):
        img_poly = img_poly.clone().astype(dtype="float32")
        img_poly[..., 0] = img_poly[..., 0] / (w / 2.0) - 1
        img_poly[..., 1] = img_poly[..., 1] / (h / 2.0) - 1
        batch_size = energy_field.shape[0]
        gcn_feature = paddle.zeros(
            shape=[img_poly.shape[0], energy_field.shape[1], img_poly.shape[1]]
        ).to(img_poly.place)
        for i in range(batch_size):
            poly = img_poly[ind == i].unsqueeze(axis=0)
            gcn_feature[ind == i] = paddle.nn.functional.grid_sample(
                x=energy_field[i : i + 1], grid=poly
            )[0].transpose(perm=[1, 0, 2])
        return gcn_feature

    def loss_energy_regularization(self, energy_field, img_poly, inds, h, w):
        energys = []
        for i, py in enumerate(img_poly):
            energy = self.get_poly_energy(
                energy_field.unsqueeze(axis=1), py, inds, h, w
            )
            energys.append(energy.squeeze(axis=1).sum(axis=-1))
        regular_loss = paddle.to_tensor(data=0.0)
        energy_loss = paddle.to_tensor(data=0.0)
        for i, e in enumerate(energys[1:]):
            regular_loss += paddle.clip(x=e - energys[i], min=0.0).mean()
            energy_loss += paddle.where(
                condition=e <= 0.01, x=paddle.to_tensor(data=0.0), y=e
            ).mean()
        return (energy_loss + regular_loss) / len(energys[1:])

    def dice_loss(self, x, target, mask):
        b = tuple(x.shape)[0]
        x = paddle.nn.functional.sigmoid(x=x)
        x = x.reshape(b, -1)
        target = target.reshape(b, -1)
        mask = mask.reshape(b, -1)
        x = x * mask
        target = target.astype(dtype="float32")
        target = target * mask
        a = paddle.sum(x=x * target, axis=1)
        b = paddle.sum(x=x * x, axis=1) + 0.001
        c = paddle.sum(x=target * target, axis=1) + 0.001
        d = 2 * a / (b + c)
        loss = 1 - d
        loss = paddle.mean(x=loss)
        return loss

    def forward(self, input_dict, output_dict, eps=None):
        """
        calculate boundary proposal network loss
        """
        fy_preds = output_dict["fy_preds"]
        if not cfg.onlybackbone:
            py_preds = output_dict["py_preds"]
            inds = output_dict["inds"]
        train_mask = input_dict["train_mask"].astype(dtype="float32")
        tr_mask = input_dict["tr_mask"] > 0
        distance_field = input_dict["distance_field"]
        direction_field = input_dict["direction_field"]
        weight_matrix = input_dict["weight_matrix"]
        gt_tags = input_dict["gt_points"]
        instance = input_dict["tr_mask"].astype(dtype="int64")
        conf = tr_mask.astype(dtype="float32")
        if cfg.scale > 1:
            train_mask = (
                paddle.nn.functional.interpolate(
                    x=train_mask.astype(dtype="float32").unsqueeze(axis=1),
                    scale_factor=1 / cfg.scale,
                    mode="bilinear",
                )
                .squeeze()
                .astype(dtype="bool")
            )
            tr_mask = (
                paddle.nn.functional.interpolate(
                    x=tr_mask.astype(dtype="float32").unsqueeze(axis=1),
                    scale_factor=1 / cfg.scale,
                    mode="bilinear",
                )
                .squeeze()
                .astype(dtype="bool")
            )
            distance_field = paddle.nn.functional.interpolate(
                x=distance_field.unsqueeze(axis=1),
                scale_factor=1 / cfg.scale,
                mode="bilinear",
            ).squeeze()
            direction_field = paddle.nn.functional.interpolate(
                x=direction_field, scale_factor=1 / cfg.scale, mode="bilinear"
            )
            weight_matrix = paddle.nn.functional.interpolate(
                x=weight_matrix.unsqueeze(axis=1),
                scale_factor=1 / cfg.scale,
                mode="bilinear",
            ).squeeze()
        cls_loss = self.BCE_loss(fy_preds[:, 0, :, :], conf)
        cls_loss = paddle.multiply(x=cls_loss, y=paddle.to_tensor(train_mask)).mean()
        dis_loss = self.MSE_loss(fy_preds[:, 1, :, :], distance_field)
        dis_loss = paddle.multiply(x=dis_loss, y=paddle.to_tensor(train_mask))
        dis_loss = self.single_image_loss(dis_loss, distance_field)
        train_mask = train_mask > 0
        norm_loss, angle_loss = self.loss_calc_flux(
            fy_preds[:, 2:4, :, :], direction_field, weight_matrix, tr_mask, train_mask
        )
        if cfg.onlybackbone:
            alpha = 1.0
            beta = 3.0
            theta = 0.5
            loss = alpha * cls_loss + beta * dis_loss + theta * (norm_loss + angle_loss)
            loss_dict = {
                "total_loss": loss,
                "cls_loss": alpha * cls_loss,
                "distance loss": beta * dis_loss,
                "dir_loss": theta * (norm_loss + angle_loss),
                "norm_loss": theta * norm_loss,
            }
            return loss_dict
        point_loss = self.PolyMatchingLoss(py_preds[1:], gt_tags[inds])
        if cfg.mid:
            midline = output_dict["midline"]
            gt_midline = input_dict["gt_mid_points"]
            midline_loss = 0.5 * self.midPolyMatchingLoss(midline, gt_midline[inds])
        if cfg.embed:
            embed = output_dict["embed"]
            edge_field = input_dict["edge_field"].astype(dtype="float32")
            embed_loss = self.overlap_loss(embed, conf, instance, edge_field, inds)
        h, w = distance_field.shape[1] * cfg.scale, distance_field.shape[2] * cfg.scale
        energy_loss = self.loss_energy_regularization(
            distance_field, py_preds, inds[0], h, w
        )
        alpha = 1.0
        beta = 3.0
        theta = 0.5
        embed_ratio = 0.5
        if eps is None:
            gama = 0.05
        else:
            gama = 0.1 * paddle.nn.functional.sigmoid(
                x=paddle.to_tensor(data=(eps - cfg.max_epoch) / cfg.max_epoch)
            )
        loss = (
            alpha * cls_loss
            + beta * dis_loss
            + theta * (norm_loss + angle_loss)
            + gama * (point_loss + energy_loss)
        )
        if cfg.mid:
            loss = loss + gama * midline_loss
        if cfg.embed:
            loss = loss + embed_ratio * embed_loss
        loss_dict = {
            "total_loss": loss,
            "cls_loss": alpha * cls_loss,
            "distance loss": beta * dis_loss,
            "dir_loss": theta * (norm_loss + angle_loss),
            "norm_loss": theta * norm_loss,
            "angle_loss": theta * angle_loss,
            "point_loss": gama * point_loss,
            "energy_loss": gama * energy_loss,
        }
        if cfg.embed:
            loss_dict["embed_loss"] = embed_ratio * embed_loss
        if cfg.mid:
            loss_dict["midline_loss"] = gama * midline_loss
        return loss_dict


class knowledge_loss(paddle.nn.Layer):
    def __init__(self, T):
        super().__init__()
        self.KLDloss = paddle.nn.KLDivLoss(reduction="sum")
        self.T = T

    def forward(self, pred, know):
        log_pred = paddle.nn.functional.log_softmax(x=pred / self.T, axis=1)
        sftknow = paddle.nn.functional.softmax(x=know / self.T, axis=1)
        kldloss = self.KLDloss(log_pred, sftknow)
        kldloss = (
            kldloss
            * self.T**2
            / (tuple(pred.shape)[0] * tuple(pred.shape)[2] * tuple(pred.shape)[3])
        )
        return kldloss
