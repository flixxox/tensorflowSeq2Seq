import tensorflow as tf

from tensorflowSeq2Seq.util.timer import Timer


class MultiHeadAttention(tf.Module):

    def __init__(self, H, D, dropout, name='multihead_attention'):
        super(MultiHeadAttention, self).__init__(name=name)

        with self.name_scope:

            self.H = H
            self.D = D
            self.Dh = D // H

            self.att = DotProductAttention(dropout)
            
            self.W_q = Timer(tf.keras.layers.Dense(D, name='W_q'))
            self.W_k = Timer(tf.keras.layers.Dense(D, name='W_k'))
            self.W_v = Timer(tf.keras.layers.Dense(D, name='W_v'))
            self.W_o = Timer(tf.keras.layers.Dense(D, name='W_o'))

            self.reshape = Timer(Reshape())
            self.transpose = Timer(Transpose())

    @tf.Module.with_name_scope
    def __call__(self, q, k, v, training, m=None):
        
        B = tf.shape(q)[0]

        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        q = self.reshape(q, (B, -1, self.H, self.Dh))
        k = self.reshape(k, (B, -1, self.H, self.Dh))
        v = self.reshape(v, (B, -1, self.H, self.Dh))

        q = self.transpose(q, perm=(0, 2, 1, 3))
        k = self.transpose(k, perm=(0, 2, 1, 3))
        v = self.transpose(v, perm=(0, 2, 1, 3))

        o, a = self.att(q, k, v, training, m=m)

        o = self.transpose(o, perm=(0, 2, 1, 3))
        o = self.reshape(o, (B, -1, self.D))
        o = self.W_o(o)

        return o, a


class DotProductAttention(tf.Module):

    def __init__(self, dropout, name='dot_attention'):
        super(DotProductAttention, self).__init__(name=name)

        with self.name_scope:

            self.reshape = Timer(Reshape())
            self.transpose = Timer(Transpose())
            self.matmul = Timer(MatMul())
            self.where = Timer(Where())

            self.softmax = Timer(tf.keras.layers.Softmax(-1, name='softmax'))
            self.dropout = Timer(tf.keras.layers.Dropout(dropout, name='dropout'))

    @tf.Module.with_name_scope
    def __call__(self, q, k, v, training, m=None):
        
        D = tf.shape(q)[-1]
        D = tf.cast(D, dtype=tf.float32)

        a = self.matmul(q, k, transpose_b=True)
        a = a / tf.math.sqrt(D)

        if m is not None:
            a = self.where(m, -float('inf'), a)

        a = self.softmax(a)
        a = self.dropout(a, training=training)

        o = self.matmul(a, v)

        return o, a


class PositionalEmbedding(tf.Module):

    def __init__(self, V, model_dim, maxI, dropout, word_embed=None, name='positional_embed'):
        super(PositionalEmbedding, self).__init__(name=name)

        with self.name_scope:

            self.model_dim = model_dim

            if word_embed is None:
                self.word_embed = tf.keras.layers.Embedding(V, model_dim, name='word_embed')
            else:
                self.word_embed = word_embed

            self.pos_embed = tf.keras.layers.Embedding(maxI, model_dim, name='pos_embed')

            self.dropout = tf.keras.layers.Dropout(dropout, name='dropout')

            self.rng = tf.Variable(
                tf.range(maxI),
                trainable=False,
                name='embed_range'
            )

    @tf.Module.with_name_scope
    def __call__(self, x, training):

        B = tf.shape(x)[0]
        J = tf.shape(x)[1]
        D = self.model_dim

        x = self.word_embed(x)

        pos = self.pos_embed(self.rng[:J])

        pos.set_shape(pos.shape.as_list()[:-1] + [D])

        pos = tf.expand_dims(pos, 0)
        pos = tf.repeat(pos, [B], axis=0)

        x = x + pos
        
        x = x * tf.math.sqrt(float(D))

        x = self.dropout(x, training=training)

        return x


class LayerNormalization(tf.Module):

    def __init__(self, model_dim, name='layer_norm'):
        super(LayerNormalization, self).__init__(name=name)

        with self.name_scope:

            self.a = tf.Variable(initial_value=tf.ones(model_dim), name='a')
            self.b = tf.Variable(initial_value=tf.zeros(model_dim), name='b')

    @tf.Module.with_name_scope
    def __call__(self, x):
        
        mu = tf.reduce_mean(x, axis=-1, keepdims=True)
        sg = tf.math.reduce_variance(x, axis=-1, keepdims=True)

        x = (x - mu) / tf.sqrt(sg + 1e-8)
        x = x * self.a + self.b

        return x


class TiedDenseLayer(tf.Module):

    def __init__(self, reference_layer, output_dim, name='tied_dense'):
        super(TiedDenseLayer, self).__init__(name=name)

        self.output_dim = output_dim
        self.reference_layer = reference_layer

        with self.name_scope:

            self.bias = tf.Variable(initial_value=tf.zeros(output_dim), trainable=True, name='bias')

    @tf.Module.with_name_scope
    def __call__(self, x):

        x = tf.tensordot(x, self.reference_layer.weights[0], [[2], [1]])
 
        x = tf.nn.bias_add(x, self.bias)
        
        return x


class Reshape(tf.Module):

    def __init__(self, name='reshape'):
        super(Reshape, self).__init__(name=name)

    def __call__(self, *args, **kwargs):
        return tf.reshape(*args, **kwargs)


class Transpose(tf.Module):

    def __init__(self, name='transpose'):
        super(Transpose, self).__init__(name=name)

    def __call__(self, *args, **kwargs):
        return tf.transpose(*args, **kwargs)


class MatMul(tf.Module):

    def __init__(self, name='matmul'):
        super(MatMul, self).__init__(name=name)

    def __call__(self, *args, **kwargs):
        return tf.linalg.matmul(*args, **kwargs)


class Where(tf.Module):

    def __init__(self, name='where'):
        super(Where, self).__init__(name=name)

    def __call__(self, *args, **kwargs):
        return tf.where(*args, **kwargs)


class LogSoftmax(tf.Module):

    def __init__(self, name='log_softmax'):
        super(LogSoftmax, self).__init__(name=name)

    def __call__(self, *args, **kwargs):
        return tf.nn.log_softmax(*args, **kwargs)