import paddle
import numpy as np
from cfglib.config import config as cfg


class Positional_encoding(paddle.nn.Layer):
    def __init__(self, PE_size, n_position=256):
        super(Positional_encoding, self).__init__()
        self.PE_size = PE_size
        self.n_position = n_position
        self.register_buffer(
            name="pos_table", tensor=self.get_encoding_table(n_position, PE_size)
        )

    def get_encoding_table(self, n_position, PE_size):
        position_table = np.array(
            [
                [
                    (pos / np.power(10000, 2.0 * i / self.PE_size))
                    for i in range(self.PE_size)
                ]
                for pos in range(n_position)
            ]
        )
        position_table[:, 0::2] = np.sin(position_table[:, 0::2])
        position_table[:, 1::2] = np.cos(position_table[:, 1::2])
        return paddle.to_tensor(data=position_table, dtype="float32").unsqueeze(axis=0)

    def forward(self, inputs):
        return inputs + self.pos_table[:, : inputs.shape[1], :].clone().detach()


class MultiHeadAttention(paddle.nn.Layer):
    def __init__(
        self, num_heads, embed_dim, dropout=0.1, if_resi=True, batch_first=False
    ):
        super(MultiHeadAttention, self).__init__()
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=embed_dim)
        self.MultiheadAttention = paddle.nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=batch_first
        )
        self.Q_proj = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=embed_dim, out_features=embed_dim),
            paddle.nn.ReLU(),
        )
        self.K_proj = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=embed_dim, out_features=embed_dim),
            paddle.nn.ReLU(),
        )
        self.V_proj = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=embed_dim, out_features=embed_dim),
            paddle.nn.ReLU(),
        )
        self.if_resi = if_resi

    def forward(self, inputs):
        query = self.layer_norm(inputs)
        q = self.Q_proj(query)
        k = self.K_proj(query)
        v = self.V_proj(query)
        attn_output, attn_output_weights = self.MultiheadAttention(q, k, v)
        if self.if_resi:
            attn_output += inputs
        else:
            attn_output = attn_output
        return attn_output


class FeedForward(paddle.nn.Layer):
    def __init__(self, in_channel, FFN_channel, if_resi=True):
        super(FeedForward, self).__init__()
        """
        1024 2048
        """
        output_channel = FFN_channel, in_channel
        self.fc1 = paddle.nn.Sequential(
            paddle.nn.Linear(in_features=in_channel, out_features=output_channel[0]),
            paddle.nn.ReLU(),
        )
        self.fc2 = paddle.nn.Linear(
            in_features=output_channel[0], out_features=output_channel[1]
        )
        self.layer_norm = paddle.nn.LayerNorm(normalized_shape=in_channel)
        self.if_resi = if_resi

    def forward(self, inputs):
        outputs = self.layer_norm(inputs)
        outputs = self.fc1(outputs)
        outputs = self.fc2(outputs)
        if self.if_resi:
            outputs += inputs
        else:
            outputs = outputs
        return outputs


class TransformerLayer(paddle.nn.Layer):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads,
        attention_size,
        dim_feedforward=1024,
        drop_rate=0.1,
        if_resi=True,
        block_nums=3,
        batch_first=False,
    ):
        super(TransformerLayer, self).__init__()
        self.block_nums = block_nums
        self.if_resi = if_resi
        self.linear = paddle.nn.Linear(in_features=in_dim, out_features=attention_size)
        for i in range(self.block_nums):
            self.__setattr__(
                "MHA_self_%d" % i,
                MultiHeadAttention(
                    num_heads,
                    attention_size,
                    dropout=drop_rate,
                    if_resi=if_resi,
                    batch_first=batch_first,
                ),
            )
            self.__setattr__(
                "FFN_%d" % i, FeedForward(out_dim, dim_feedforward, if_resi=if_resi)
            )

    def forward(self, query):
        inputs = self.linear(query)
        for i in range(self.block_nums):
            outputs = self.__getattr__("MHA_self_%d" % i)(inputs)
            outputs = self.__getattr__("FFN_%d" % i)(outputs)
            if self.if_resi:
                inputs = inputs + outputs
            else:
                inputs = outputs
        return inputs


class Transformer(paddle.nn.Layer):
    def __init__(
        self,
        in_dim,
        out_dim,
        num_heads=8,
        dim_feedforward=1024,
        drop_rate=0.1,
        if_resi=False,
        block_nums=3,
        pred_num=2,
        batch_first=False,
    ):
        super().__init__()
        self.bn0 = paddle.nn.BatchNorm1D(
            num_features=in_dim, weight_attr=False, bias_attr=False
        )
        self.conv1 = paddle.nn.Conv1D(
            in_channels=in_dim, out_channels=out_dim, kernel_size=1, dilation=1
        )
        self.transformer = TransformerLayer(
            in_dim,
            out_dim,
            num_heads,
            attention_size=out_dim,
            dim_feedforward=dim_feedforward,
            drop_rate=drop_rate,
            if_resi=if_resi,
            block_nums=block_nums,
            batch_first=batch_first,
        )
        self.prediction = paddle.nn.Sequential(
            paddle.nn.Conv1D(in_channels=2 * out_dim, out_channels=128, kernel_size=1),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(p=0.1),
            paddle.nn.Conv1D(in_channels=128, out_channels=64, kernel_size=1),
            paddle.nn.ReLU(),
            paddle.nn.Conv1D(in_channels=64, out_channels=pred_num, kernel_size=1),
        )

    def forward(self, x):
        x = self.bn0(x)
        x1 = x.transpose(perm=[0, 2, 1])
        x1 = self.transformer(x1)
        x1 = x1.transpose(perm=[0, 2, 1])
        x = paddle.concat(x=[x1, self.conv1(x)], axis=1)
        pred = self.prediction(x)
        return pred
