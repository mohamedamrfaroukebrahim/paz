from keras import ops
from keras.layers import Dense, Dropout, Layer


class Attention(Layer):
    def __init__(
        self,
        dimension,
        num_heads=8,
        qkv_bias=False,
        projection_bias=True,
        attention_drop_rate=0.0,
        projection_drop_rate=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        self.projection_bias = projection_bias
        self.attention_drop_rate = attention_drop_rate
        self.projection_drop_rate = projection_drop_rate

    def build(self, input_shape):
        head_dim = self.dimension // self.num_heads
        self.scale = head_dim**-0.5

        self.qkv = Dense(self.dimension * 3, use_bias=self.qkv_bias)
        self.attention_drop_rate = Dropout(self.attention_drop_rate)
        self.proj = Dense(self.dimension, use_bias=self.projection_bias)
        self.projection_drop_rate = Dropout(self.projection_drop_rate)

    def call(self, x):
        B, N, C = ops.shape(x)[0], ops.shape(x)[1], ops.shape(x)[2]
        qkv = self.qkv(x)
        qkv = ops.reshape(qkv, (B, N, 3, self.num_heads, C // self.num_heads))
        qkv = ops.transpose(qkv, axes=[2, 0, 3, 1, 4])

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = ops.matmul(q, ops.transpose(k, axes=[0, 1, 3, 2]))

        attn = ops.softmax(attn, axis=-1)
        attn = self.attention_drop_rate(attn)

        x = ops.matmul(attn, v)
        x = ops.transpose(x, axes=[0, 2, 1, 3])
        x = ops.reshape(x, (B, N, C))
        x = self.proj(x)
        x = self.projection_drop_rate(x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "dimension": self.dimension,
                "num_heads": self.num_heads,
                "qkv_bias": self.qkv_bias,
                "projection_bias": self.projection_bias,
                "attention_drop_rate": self.attention_drop_rate,
                "projection_drop_rate": self.projection_drop_rate,
            }
        )
        return config
