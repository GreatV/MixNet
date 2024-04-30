import paddle
from cfglib.config import config as cfg


class overlap_loss(paddle.nn.Layer):
    def __init__(self):
        super().__init__()
        self.BCE_loss = paddle.nn.BCELoss(reduction="none")
        self.inst_loss = paddle.nn.MSELoss(reduction="sum")

    def forward(self, preds, conf, inst, overlap, inds):
        p1 = preds[:, 0]
        p2 = preds[:, 1]
        and_preds = p1 * p2
        and_loss = self.BCE_loss(and_preds, overlap)
        or_preds = paddle.maximum(x=p1, y=p2)
        or_loss = self.BCE_loss(or_preds, conf)
        and_overlap = and_preds * overlap
        op1 = paddle.maximum(x=p1, y=and_overlap)
        op2 = paddle.maximum(x=p2, y=and_overlap)
        inst_loss = paddle.to_tensor(data=0)
        b, h, w = tuple(p1.shape)
        for i in range(b):
            bop1 = op1[i]
            bop2 = op2[i]
            inst_label = inst[i]
            keys = paddle.unique(x=inst_label)
            tmp = paddle.to_tensor(data=0)
            for k in keys:
                inst_map = (inst_label == k).astype(dtype="float32")
                suminst = paddle.sum(x=inst_map)
                d1 = self.inst_loss(bop1 * inst_map, inst_map) / suminst
                d2 = self.inst_loss(bop2 * inst_map, inst_map) / suminst
                tmp = tmp + paddle.minimum(d1, d2) - paddle.maximum(d1, d2) + 1
            inst_loss = inst_loss + tmp / tuple(keys.shape)[0]
        and_loss = and_loss[conf == 1].mean() + and_loss[conf == 0].mean()
        or_loss = or_loss.mean()
        inst_loss = inst_loss / b
        loss = 0.5 * and_loss + 0.25 * or_loss + 0.25 * inst_loss
        return loss
