import paddle
import numpy as np


class PolyMatchingLoss(paddle.nn.Layer):
    def __init__(self, pnum, device, loss_type="L1"):
        super(PolyMatchingLoss, self).__init__()
        self.pnum = pnum
        self.device = device
        self.loss_type = loss_type
        self.smooth_L1 = paddle.nn.functional.smooth_l1_loss
        self.L2_loss = paddle.nn.MSELoss(reduction="none")
        batch_size = 1
        pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
        for b in range(batch_size):
            for i in range(pnum):
                pidx = (np.arange(pnum) + i) % pnum
                pidxall[b, i] = pidx
        pidxall = paddle.to_tensor(
            data=np.reshape(pidxall, newshape=(batch_size, -1))
        ).to(device)
        self.feature_id = (
            pidxall.unsqueeze_(axis=2)
            .astype(dtype="int64")
            .expand(shape=[pidxall.shape[0], pidxall.shape[1], 2])
            .detach()
        )

    def match_loss(self, pred, gt):
        batch_size = tuple(pred.shape)[0]
        feature_id = self.feature_id.expand(
            shape=[batch_size, self.feature_id.shape[1], 2]
        )
        gt_expand = paddle.take_along_axis(arr=gt, axis=1, indices=feature_id).view(
            batch_size, self.pnum, self.pnum, 2
        )
        pred_expand = pred.unsqueeze(axis=1)
        if self.loss_type == "L2":
            dis = self.L2_loss(pred_expand, gt_expand)
            dis = dis.sum(axis=3).sqrt().mean(axis=2)
        elif self.loss_type == "L1":
            dis = self.smooth_L1(pred_expand, gt_expand, reduction="none")
            dis = dis.sum(axis=3).mean(axis=2)
        min_dis, min_id = (
            paddle.min(x=dis, axis=1, keepdim=True),
            paddle.argmin(x=dis, axis=1, keepdim=True),
        )
        return min_dis

    def forward(self, pred_list, gt):
        loss = paddle.to_tensor(data=0.0)
        for pred in pred_list:
            loss += paddle.mean(x=self.match_loss(pred, gt))
        return loss / paddle.to_tensor(data=len(pred_list))


class AttentionLoss(paddle.nn.Layer):
    def __init__(self, beta=4, gamma=0.5):
        super(AttentionLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, gt):
        num_pos = paddle.sum(x=gt)
        num_neg = paddle.sum(x=1 - gt)
        alpha = num_neg / (num_pos + num_neg)
        edge_beta = paddle.pow(x=self.beta, y=paddle.pow(x=1 - pred, y=self.gamma))
        bg_beta = paddle.pow(x=self.beta, y=paddle.pow(x=pred, y=self.gamma))
        loss = 0
        loss = loss - alpha * edge_beta * paddle.log(x=pred) * gt
        loss = loss - (1 - alpha) * bg_beta * paddle.log(x=1 - pred) * (1 - gt)
        return paddle.mean(x=loss)


class GeoCrossEntropyLoss(paddle.nn.Layer):
    def __init__(self):
        super(GeoCrossEntropyLoss, self).__init__()

    def forward(self, output, target, poly):
        output = paddle.nn.functional.softmax(x=output, axis=1)
        output = paddle.log(x=paddle.clip(x=output, min=0.0001))
        poly = poly.view(poly.shape[0], 4, poly.shape[1] // 4, 2)
        target = target[..., None, None].expand(
            shape=[poly.shape[0], poly.shape[1], 1, poly.shape[3]]
        )
        target_poly = paddle.take_along_axis(arr=poly, axis=2, indices=target)
        sigma = (poly[:, :, 0] - poly[:, :, 1]).pow(y=2).sum(axis=2, keepdim=True)
        kernel = paddle.exp(x=-(poly - target_poly).pow(y=2).sum(axis=3) / (sigma / 3))
        x = kernel
        perm_1 = list(range(x.ndim))
        perm_1[2] = 1
        perm_1[1] = 2
        loss = -(output * x.transpose(perm=perm_1)).sum(axis=1).mean()
        return loss


class AELoss(paddle.nn.Layer):
    def __init__(self):
        super(AELoss, self).__init__()

    def forward(self, ae, ind, ind_mask):
        """
        ae: [b, 1, h, w]
        ind: [b, max_objs, max_parts]
        ind_mask: [b, max_objs, max_parts]
        obj_mask: [b, max_objs]
        """
        b, _, h, w = tuple(ae.shape)
        b, max_objs, max_parts = tuple(ind.shape)
        obj_mask = paddle.sum(x=ind_mask, axis=2) != 0
        ae = ae.view(b, h * w, 1)
        seed_ind = ind.view(b, max_objs * max_parts, 1)
        tag = ae.take_along_axis(axis=1, indices=seed_ind).view(b, max_objs, max_parts)
        tag_mean = tag * ind_mask
        tag_mean = tag_mean.sum(axis=2) / (ind_mask.sum(axis=2) + 0.0001)
        pull_dist = (tag - tag_mean.unsqueeze(axis=2)).pow(y=2) * ind_mask
        obj_num = obj_mask.sum(axis=1).astype(dtype="float32")
        pull = (pull_dist.sum(axis=(1, 2)) / (obj_num + 0.0001)).sum()
        pull /= b
        push_dist = paddle.abs(
            x=tag_mean.unsqueeze(axis=1) - tag_mean.unsqueeze(axis=2)
        )
        push_dist = 1 - push_dist
        push_dist = paddle.nn.functional.relu(x=push_dist)
        obj_mask = obj_mask.unsqueeze(axis=1) + obj_mask.unsqueeze(axis=2) == 2
        push_dist = push_dist * obj_mask.astype(dtype="float32")
        push = (
            (push_dist.sum(axis=(1, 2)) - obj_num) / (obj_num * (obj_num - 1) + 0.0001)
        ).sum()
        push /= b
        return pull, push


def smooth_l1_loss(inputs, target, sigma=9.0):
    try:
        diff = paddle.abs(x=inputs - target)
        less_one = (diff < 1.0 / sigma).astype(dtype="float32")
        loss = less_one * 0.5 * diff**2 * sigma + paddle.abs(
            x=paddle.to_tensor(data=1.0) - less_one
        ) * (diff - 0.5 / sigma)
        loss = paddle.mean(x=loss) if loss.size > 0 else paddle.to_tensor(data=0.0)
    except Exception as e:
        print("RPN_REGR_Loss Exception:", e)
        loss = paddle.to_tensor(data=0.0)
    return loss


def _neg_loss(pred, gt):
    """Modified focal loss. Exactly the same as CornerNet.
    Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x c x h x w)
        gt_regr (batch x c x h x w)
    """
    pos_inds = gt.equal(y=1).astype(dtype="float32")
    neg_inds = gt.less_than(y=paddle.to_tensor(1)).astype(dtype="float32")
    neg_weights = paddle.pow(x=1 - gt, y=4)
    loss = 0
    pos_loss = paddle.log(x=pred) * paddle.pow(x=1 - pred, y=2) * pos_inds
    neg_loss = paddle.log(x=1 - pred) * paddle.pow(x=pred, y=2) * neg_weights * neg_inds
    num_pos = pos_inds.astype(dtype="float32").sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(paddle.nn.Layer):
    """nn.Module warpper for focal loss"""

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)
