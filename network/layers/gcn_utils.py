import paddle
import numpy as np
import cv2
from numpy import pi


def normalize_adj(A, type="AD"):
    if type == "DAD":
        A = A + np.eye(tuple(A.shape)[0])
        d = np.sum(A, axis=0)
        d_inv = np.power(d, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv = np.diag(d_inv)
        G = A.dot(y=d_inv).transpose().dot(y=d_inv)
        G = paddle.to_tensor(data=G)
    elif type == "AD":
        A = A + np.eye(tuple(A.shape)[0])
        A = paddle.to_tensor(data=A)
        D = A.sum(axis=1, keepdim=True)
        G = A.div(D)
    else:
        A = A + np.eye(tuple(A.shape)[0])
        D = A.sum(axis=1, keepdim=True)
        D = np.diag(D)
        G = paddle.to_tensor(data=D - A)
    return G


def np_to_variable(x, is_cuda=True, dtype=paddle.float32):
    return x


def set_trainable(model, requires_grad):
    for param in model.parameters():
        param.stop_gradient = not requires_grad


def weights_normal_init(model, dev=0.01):
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.sublayers():
            if isinstance(m, paddle.nn.Conv2D):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, paddle.nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    """Computes a gradient clipping coefficient based on gradient norm."""
    totalnorm = 0
    for p in model.parameters():
        if not p.stop_gradient:
            modulenorm = p.grad.data.norm()
            totalnorm += modulenorm**2
    totalnorm = np.sqrt(totalnorm)
    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if not p.stop_gradient:
            p.grad.mul_(norm)


def EuclideanDistances(A, B):
    BT = B.transpose()
    vecProd = np.dot(A, BT)
    SqA = A**2
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, tuple(vecProd.shape)[1]))
    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (tuple(vecProd.shape)[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2 * vecProd
    SqED[SqED < 0] = 0.0
    ED = np.sqrt(SqED)
    return ED


def get_center_feature(cnn_feature, img_poly, ind, h, w):
    batch_size = cnn_feature.shape[0]
    for i in range(batch_size):
        poly = img_poly[ind == i].cpu().numpy()
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, poly.astype(np.int32), color=(1,))
    return None


def get_node_feature(cnn_feature, img_poly, ind, h, w):
    img_poly = img_poly.clone().astype(dtype="float32")
    img_poly[..., 0] = img_poly[..., 0] / (w / 2.0) - 1
    img_poly[..., 1] = img_poly[..., 1] / (h / 2.0) - 1
    batch_size = cnn_feature.shape[0]
    gcn_feature = paddle.zeros(
        shape=[img_poly.shape[0], cnn_feature.shape[1], img_poly.shape[1]]
    ).to(img_poly.place)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(axis=0)
        gcn_feature[ind == i] = paddle.nn.functional.grid_sample(
            x=cnn_feature[i : i + 1], grid=poly
        )[0].transpose(perm=[1, 0, 2])
    return gcn_feature


def get_adj_mat(n_adj, n_nodes):
    a = np.zeros([n_nodes, n_nodes], dtype=np.float)
    for i in range(n_nodes):
        for j in range(-n_adj // 2, n_adj // 2 + 1):
            if j != 0:
                a[i][(i + j) % n_nodes] = 1
                a[(i + j) % n_nodes][i] = 1
    return a


def get_adj_ind(n_adj, n_nodes, device):
    ind = paddle.to_tensor(
        data=[i for i in range(-n_adj // 2, n_adj // 2 + 1) if i != 0]
    ).astype(dtype="int64")
    ind = (paddle.arange(end=n_nodes)[:, None] + ind[None]) % n_nodes
    return ind.to(device)


def coord_embedding(b, w, h, device):
    x_range = paddle.linspace(start=0, stop=1, num=w)
    y_range = paddle.linspace(start=0, stop=1, num=h)
    y, x = paddle.meshgrid(y_range, x_range)
    y = y.expand(shape=[b, 1, -1, -1])
    x = x.expand(shape=[b, 1, -1, -1])
    coord_map = paddle.concat(x=[x, y], axis=1)
    return coord_map


def img_poly_to_can_poly(img_poly):
    if len(img_poly) == 0:
        return paddle.zeros_like(x=img_poly)
    x_min = (
        paddle.min(x=img_poly[..., 0], axis=-1),
        paddle.argmin(x=img_poly[..., 0], axis=-1),
    )[0]
    y_min = (
        paddle.min(x=img_poly[..., 1], axis=-1),
        paddle.argmin(x=img_poly[..., 1], axis=-1),
    )[0]
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
    can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
    return can_poly
