import paddle
import numpy as np


class EmbLoss_v2(paddle.nn.Layer):
    def __init__(self, feature_dim=4, loss_weight=1.0):
        super(EmbLoss_v2, self).__init__()
        self.feature_dim = feature_dim
        self.loss_weight = loss_weight
        self.delta_v = 0.5
        self.delta_d = 1.5
        self.weights = 1.0, 1.0

    def forward_single(self, emb, instance, kernel, training_mask):
        training_mask = (training_mask > 0.5).astype(dtype="int64")
        kernel = (kernel > 0.5).astype(dtype="int64")
        instance = instance * training_mask
        instance_kernel = (instance * kernel).view(-1)
        instance = instance.view(-1)
        emb = emb.view(self.feature_dim, -1)
        unique_labels, unique_ids = paddle.unique(
            instance_kernel, sorted=True, return_inverse=True
        )
        num_instance = unique_labels.shape[0]
        if num_instance <= 1:
            return 0
        emb_mean = paddle.zeros(shape=(self.feature_dim, num_instance), dtype="float32")
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind_k = instance_kernel == lb
            emb_mean[:, i] = paddle.mean(x=emb[:, ind_k], axis=1)
        l_agg = paddle.zeros(shape=num_instance, dtype="float32")
        for i, lb in enumerate(unique_labels):
            if lb == 0:
                continue
            ind = instance == lb
            emb_ = emb[:, ind]
            dist = (emb_ - emb_mean[:, i : i + 1]).norm(p=2, axis=0)
            dist = paddle.nn.functional.relu(x=dist - self.delta_v) ** 2
            l_agg[i] = paddle.mean(x=paddle.log(x=dist + 1.0))
        l_agg = paddle.mean(x=l_agg[1:])
        if num_instance > 2:
            emb_interleave = emb_mean.transpose(perm=[1, 0]).repeat(num_instance, 1)
            emb_band = (
                emb_mean.transpose(perm=[1, 0])
                .repeat(1, num_instance)
                .view(-1, self.feature_dim)
            )
            mask = (
                (1 - paddle.eye(num_rows=num_instance, dtype="int8"))
                .view(-1, 1)
                .repeat(1, self.feature_dim)
            )
            mask = mask.view(num_instance, num_instance, -1)
            mask[0, :, :] = 0
            mask[:, 0, :] = 0
            mask = mask.view(num_instance * num_instance, -1)
            dist = emb_interleave - emb_band
            dist = dist[mask > 0].view(-1, self.feature_dim).norm(p=2, axis=1)
            dist = paddle.nn.functional.relu(x=2 * self.delta_d - dist) ** 2
            l_dis = [paddle.log(x=dist + 1.0)]
            emb_bg = emb[:, instance == 0].view(self.feature_dim, -1)
            if emb_bg.shape[1] > 100:
                rand_ind = np.random.permutation(emb_bg.shape[1])[:100]
                emb_bg = emb_bg[:, rand_ind]
            if emb_bg.shape[1] > 0:
                for i, lb in enumerate(unique_labels):
                    if lb == 0:
                        continue
                    dist = (emb_bg - emb_mean[:, i : i + 1]).norm(p=2, axis=0)
                    dist = paddle.nn.functional.relu(x=2 * self.delta_d - dist) ** 2
                    l_dis_bg = paddle.mean(
                        x=paddle.log(x=dist + 1.0), axis=0, keepdim=True
                    )
                    l_dis.append(l_dis_bg)
            l_dis = paddle.mean(x=paddle.concat(x=l_dis))
        else:
            l_dis = 0
        l_agg = self.weights[0] * l_agg
        l_dis = self.weights[1] * l_dis
        l_reg = (
            paddle.mean(
                x=paddle.log(x=paddle.linalg.norm(x=emb_mean, p=2, axis=0) + 1.0)
            )
            * 0.001
        )
        loss = l_agg + l_dis + l_reg
        return loss

    def forward(self, emb, instance, kernel, training_mask, reduce=True):
        loss_batch = paddle.zeros(shape=emb.shape[0], dtype="float32")
        for i in range(loss_batch.shape[0]):
            loss_batch[i] = self.forward_single(
                emb[i], instance[i], kernel[i], training_mask[i]
            )
        loss_batch = self.loss_weight * loss_batch
        if reduce:
            loss_batch = paddle.mean(x=loss_batch)
        return loss_batch
