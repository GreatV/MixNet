import paddle
import numpy as np
import time
import cv2
from cfglib.config import config as cfg
from network.layers.model_block import FPN
from network.layers.Transformer import Transformer
from network.layers.gcn_utils import get_node_feature
from util.misc import get_sample_point
import unittest
from shapely.geometry import LineString
from shapely.ops import unary_union


class midlinePredictor(paddle.nn.Layer):
    def __init__(self, seg_channel):
        super(midlinePredictor, self).__init__()
        self.seg_channel = seg_channel
        self.clip_dis = 100
        self.midline_preds = paddle.nn.LayerList()
        self.contour_preds = paddle.nn.LayerList()
        self.iter = 3
        for i in range(self.iter):
            self.midline_preds.append(
                Transformer(
                    seg_channel,
                    128,
                    num_heads=8,
                    dim_feedforward=1024,
                    drop_rate=0.0,
                    if_resi=True,
                    block_nums=3,
                    pred_num=2,
                    batch_first=False,
                )
            )
            self.contour_preds.append(
                Transformer(
                    seg_channel,
                    128,
                    num_heads=8,
                    dim_feedforward=1024,
                    drop_rate=0.0,
                    if_resi=True,
                    block_nums=3,
                    pred_num=2,
                    batch_first=False,
                )
            )
        if not self.training:
            self.iter = 1
        for m in self.sublayers():
            if isinstance(m, paddle.nn.Conv1D) or isinstance(m, paddle.nn.Conv2D):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    init_Constant = paddle.nn.initializer.Constant(value=0)
                    init_Constant(m.bias)

    def get_boundary_proposal(self, input=None):
        inds = paddle.where(input["ignore_tags"] > 0)
        init_polys = input["proposal_points"][inds]
        return init_polys, inds, None

    def get_boundary_proposal_eval(self, input=None, seg_preds=None):
        cls_preds = seg_preds[:, 0, :, :].detach().cpu().numpy()
        dis_preds = seg_preds[:, 1, :].detach().cpu().numpy()
        inds = []
        init_polys = []
        confidences = []
        for bid, dis_pred in enumerate(dis_preds):
            dis_mask = dis_pred > cfg.dis_threshold
            ret, labels = cv2.connectedComponents(
                dis_mask.astype(np.uint8), connectivity=8, ltype=cv2.CV_16U
            )
            for idx in range(1, ret):
                text_mask = labels == idx
                confidence = round(cls_preds[bid][text_mask].mean(), 3)
                if (
                    np.sum(text_mask) < 50 / (cfg.scale * cfg.scale)
                    or confidence < cfg.cls_threshold
                ):
                    continue
                confidences.append(confidence)
                inds.append([bid, 0])
                poly = get_sample_point(
                    text_mask,
                    cfg.num_points,
                    cfg.approx_factor,
                    scales=np.array([cfg.scale, cfg.scale]),
                )
                init_polys.append(poly)
        if len(inds) > 0:
            inds = (
                paddle.to_tensor(data=np.array(inds))
                .transpose(perm=[1, 0])
                .to(input["img"].place, blocking=not True)
            )
            init_polys = (
                paddle.to_tensor(data=np.array(init_polys))
                .to(input["img"].place, blocking=not True)
                .astype(dtype="float32")
            )
        else:
            init_polys = (
                paddle.to_tensor(data=np.array(init_polys))
                .to(input["img"].place, blocking=not True)
                .astype(dtype="float32")
            )
            inds = paddle.to_tensor(data=np.array(inds)).to(
                input["img"].place, blocking=not True
            )
        return init_polys, inds, confidences

    def forward(self, embed_feature, input=None, seg_preds=None, switch="gt"):
        if self.training:
            init_polys, inds, confidences = self.get_boundary_proposal(input=input)
        else:
            init_polys, inds, confidences = self.get_boundary_proposal_eval(
                input=input, seg_preds=seg_preds
            )
            if tuple(init_polys.shape)[0] == 0:
                return [init_polys, init_polys], inds, confidences, None
        if len(init_polys) == 0:
            py_preds = paddle.zeros_like(x=init_polys)
        h, w = tuple(embed_feature.shape)[2:4]
        mid_pt_num = tuple(init_polys.shape)[1] // 2
        contours = [init_polys]
        midlines = []
        for i in range(self.iter):
            node_feat = get_node_feature(embed_feature, contours[i], inds[0], h, w)
            midline = (
                contours[i][:, :mid_pt_num]
                + paddle.clip(
                    x=self.midline_preds[i](node_feat).transpose(perm=[0, 2, 1]),
                    min=-self.clip_dis,
                    max=self.clip_dis,
                )[:, :mid_pt_num]
            )
            midlines.append(midline)
            mid_feat = get_node_feature(embed_feature, midline, inds[0], h, w)
            node_feat = paddle.concat(x=(node_feat, mid_feat), axis=2)
            new_contour = (
                contours[i]
                + paddle.clip(
                    x=self.contour_preds[i](node_feat).transpose(perm=[0, 2, 1]),
                    min=-self.clip_dis,
                    max=self.clip_dis,
                )[:, : cfg.num_points]
            )
            contours.append(new_contour)
        return contours, inds, confidences, midlines
