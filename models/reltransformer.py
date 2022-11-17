
from numpy import dtype
import tensorflow as tf

from . import register_model
from tensorflowSeq2Seq.util.timer import Timer
from tensorflowSeq2Seq.util.debug import my_print
from tensorflowSeq2Seq.model.modules import (
    MultiHeadAttention,
    PositionalEmbedding,
    LayerNormalization,
    TiedDenseLayer,
    LogSoftmax
)

@register_model("RelTransformer")
class RelTransformer(tf.Module):

    def __init__(self, name='reltransformer', **kwargs):
        super(RelTransformer, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.encoder = RelTransformerEncoder(**kwargs)

            if self.tiew:
                word_embed = self.encoder.src_embed.object_to_time.word_embed
            else:
                word_embed = None

            self.decoder = RelTransformerDecoder(word_embed=word_embed,**kwargs)

    @staticmethod
    def create_model_from_config(config, vocab_src, vocab_tgt):
        
        model =  RelTransformer(
            srcV = vocab_src.vocab_size,
            tgtV = vocab_tgt.vocab_size,
            encL = config['encL'],
            decL = config['decL'],
            model_dim = config['model_dim'],
            nHeads = config['nHeads'],
            ff_dim = config['ff_dim'],
            dropout = config['dropout'],
            maxI = config['max_sentence_length'],
            tiew = config['tiew'],
            initializer = config['initializer'],
            variance_scaling_scale = config['variance_scaling_scale'],
            K = config['K']
        )

        my_print('Start trace of model.__call__ to make tf aware of the keras variables!')
        model.__call__.get_concrete_function(
            src=tf.TensorSpec((None, None), tf.int32),
            tgt=tf.TensorSpec((None, None), tf.int32),
            src_mask=tf.TensorSpec((None, 1, 1, None), tf.bool),
            tgt_mask=tf.TensorSpec((1, 1, None, None), tf.bool),
        )

        return model

    def init_weights(self, seed):

        bias_initializer = tf.keras.initializers.RandomUniform

        if self.initializer == 'VarianceScaling':

            my_print('Initialize model paramaters with VarianceScaling')
            initializer = tf.keras.initializers.VarianceScaling(
                scale=self.variance_scaling_scale,
                mode='fan_in',
                distribution='uniform',
                seed=seed
            )

        else:

            my_print('Initialize model paramaters with GlorotUniform')
            initializer = tf.keras.initializers.GlorotUniform(
                seed=seed
            )

        for var in self.trainable_variables:
            
            if len(var.shape) > 1:
                var.assign(initializer(shape=var.shape))

            if '/bias:0' in var.name:
                stdv = 1. / tf.math.sqrt(tf.cast(var.shape[0], dtype=tf.float32))
                var.assign(bias_initializer(minval=-stdv, maxval=stdv, seed=seed)(shape=var.shape))
        
    @tf.function
    def __call__(self, src, tgt, training=False, src_mask=None, tgt_mask=None):
        
        h = self.encoder(src, training=training, src_mask=src_mask, tgt_mask=tgt_mask)

        s = self.decoder(tgt, h, training=training, src_mask=src_mask, tgt_mask=tgt_mask)

        return s, h

    @tf.function
    def create_masks(self, src, tgt, pad_index):

        tgtT = tf.shape(tgt)[1]

        src_mask = (src == pad_index)
        src_mask = tf.expand_dims(src_mask, axis=1)
        src_mask = tf.expand_dims(src_mask, axis=1)
        src_mask = tf.cast(src_mask, dtype=tf.bool)

        tgt_mask = tf.linalg.band_part(tf.ones((tgtT, tgtT)), -1, 0)
        tgt_mask = tf.expand_dims(tgt_mask, axis=0)
        tgt_mask = tf.expand_dims(tgt_mask, axis=0)
        tgt_mask = (tgt_mask == 0)

        out_mask = (tgt != pad_index)

        return {
            'src_mask': src_mask,
            'tgt_mask': tgt_mask
        }, out_mask


class RelTransformerEncoder(tf.Module):

    def __init__(self, name='encoder', **kwargs):
        super(RelTransformerEncoder, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.src_embed = Timer(PositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, name='embed_src'))
            
            self.enc_layers = []
            for l in range(1, self.encL+1):

                layer = RelTransformerEncoderLayer(name=f'encoder_layer_{l:0>2}', **kwargs)

                self.enc_layers.append(layer)

            self.lnorm = Timer(LayerNormalization(self.model_dim))

    @tf.Module.with_name_scope
    def __call__(self, src, training=False, src_mask=None, tgt_mask=None):

        h = self.src_embed(src, training)

        for layer in self.enc_layers:

            h = layer(h, training, src_mask=src_mask)

        h = self.lnorm(h)

        return h


class RelTransformerDecoder(tf.Module):

    def __init__(self, word_embed=None, name='decoder', **kwargs):
        super(RelTransformerDecoder, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.tgt_embed = Timer(PositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, word_embed=word_embed, name='embed_tgt'))

            self.dec_layers = []
            for l in range(self.decL):

                layer = RelTransformerDecoderLayer(name=f'decoder_layer_{l:0>2}', **kwargs)

                self.dec_layers.append(layer)

            self.lnorm              = Timer(LayerNormalization(self.model_dim))
            self.log_softmax        = Timer(LogSoftmax())

            if word_embed is None:
                self.output_projection  = Timer(tf.keras.layers.Dense(self.tgtV, name='output_projection'))
            else:
                self.output_projection  = Timer(TiedDenseLayer(word_embed, self.tgtV, name='output_projection'))

    @tf.Module.with_name_scope
    def __call__(self, tgt, h, training=False, src_mask=None, tgt_mask=None):

        s = self.tgt_embed(tgt, training)

        for layer in self.dec_layers:

            s = layer(s, h, training, src_mask=src_mask, tgt_mask=tgt_mask)

        s = self.lnorm(s)
        s = self.output_projection(s)
        s = self.log_softmax(s, axis=-1)

        return s


class RelTransformerEncoderLayer(tf.Module):

    def __init__(self, name='encoder_layer', **kwargs):
        super(RelTransformerEncoderLayer, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:
        
            self.lnorm1     = Timer(LayerNormalization(self.model_dim, name='lnorm1'))
            self.att        = Timer(RelativeMultiHeadAttention(self.nHeads, self.model_dim, self.K, self.dropout, name='self_att'))
            self.dropout    = Timer(tf.keras.layers.Dropout(self.dropout, name='drop'))

            self.lnorm2     = Timer(LayerNormalization(self.model_dim, name='lnorm2'))
            self.ff1        = Timer(tf.keras.layers.Dense(self.ff_dim, activation='relu', name='ff1'))
            self.ff2        = Timer(tf.keras.layers.Dense(self.model_dim, name='ff2'))

    @tf.Module.with_name_scope
    def __call__(self, x, training, src_mask=None):
        
        r = x
        x = self.lnorm1(x)
        x, _ = self.att(x, x, x, training, m=src_mask)
        x = self.dropout(x, training=training)
        x = x + r

        r = x
        x = self.lnorm2(x)
        x = self.ff1(x)
        x = self.ff2(x)
        x = self.dropout(x, training=training)
        x = x + r

        return x


class RelTransformerDecoderLayer(tf.Module):

    def __init__(self, name='decoder_layer', **kwargs):
        super(RelTransformerDecoderLayer, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.lnorm1     = Timer(LayerNormalization(self.model_dim, name='lnorm1'))
            self.self_att   = Timer(RelativeMultiHeadAttention(self.nHeads, self.model_dim, self.K, self.dropout, name='self_att'))

            self.lnorm2     = Timer(LayerNormalization(self.model_dim, name='lnorm2'))
            self.cross_att  = Timer(MultiHeadAttention(self.nHeads, self.model_dim, self.dropout, name='cross_att'))

            self.lnorm3     = Timer(LayerNormalization(self.model_dim, name='lnorm3'))
            self.ff1        = Timer(tf.keras.layers.Dense(self.ff_dim, activation='relu', name='ff1'))
            self.ff2        = Timer(tf.keras.layers.Dense(self.model_dim, name='ff2'))
            self.dropout    = Timer(tf.keras.layers.Dropout(self.dropout, name='drop'))

    @tf.Module.with_name_scope
    def __call__(self, s, h, training, src_mask=None, tgt_mask=None):

        r = s
        s = self.lnorm1(s)
        s, _ = self.self_att(s, s, s, training, m=tgt_mask)
        s = self.dropout(s, training=training)
        s = s + r

        r = s
        s = self.lnorm2(s)
        s, _ = self.cross_att(s, h, h, training, m=src_mask)
        s = self.dropout(s, training=training)
        s = s + r

        r = s
        s = self.lnorm3(s)
        s = self.ff1(s)
        s = self.ff2(s)
        s = self.dropout(s, training=training)
        s = s + r

        return s


class RelativeMultiHeadAttention(tf.Module):

    def __init__(self, H, D, K, dropout, name='multihead_attention'):
        super(RelativeMultiHeadAttention, self).__init__(name=name)

        with self.name_scope:
            
            self.K = K
            self.H = H
            self.D = D
            self.Dh = D // H
            
            self.W_q = tf.keras.layers.Dense(D, name='W_q')
            self.W_k = tf.keras.layers.Dense(D, name='W_k')
            self.W_v = tf.keras.layers.Dense(D, name='W_v')
            self.W_o = tf.keras.layers.Dense(D, name='W_o')

            self.dropout = tf.keras.layers.Dropout(dropout, name='dropout')

            self.rel_embed = tf.keras.layers.Embedding(K, self.Dh, name='rel_embed')
            self.rng = tf.Variable(
                tf.range(K),
                trainable=False,
                name='rel_range'
            )

    @tf.Module.with_name_scope
    def __call__(self, q, k, v, training, m=None):
        
        B  = tf.shape(q)[0]
        I  = tf.shape(q)[1]
        K  = self.K
        H  = self.H
        D  = self.D
        Dh = self.Dh

        q = self.W_q(q)
        k = self.W_k(k)
        v = self.W_v(v)

        q = tf.reshape(q, (B, -1, H, Dh))
        k = tf.reshape(k, (B, -1, H, Dh))
        v = tf.reshape(v, (B, -1, H, Dh))

        q = tf.transpose(q, perm=(0, 2, 1, 3))
        k = tf.transpose(k, perm=(0, 2, 1, 3))
        v = tf.transpose(v, perm=(0, 2, 1, 3))

        a = tf.matmul(q, k, transpose_b=True)

        r = self.rel_embed(self.rng)
        r = tf.matmul(q, r, transpose_b=True)

        indices = tf.expand_dims(tf.range(I), axis=0)
        indices = tf.repeat(indices, I, axis=0)
        indices = tf.math.abs(indices - tf.expand_dims(tf.range(I), axis=1))
        indices = tf.math.maximum(-K+1, tf.math.minimum(K-1, indices))
        indices = tf.repeat(tf.expand_dims(indices, axis=0), H, axis=0)
        indices = tf.repeat(tf.expand_dims(indices, axis=0), B, axis=0) 

        r = tf.gather(r, indices, batch_dims=3, axis=-1)

        a = a + r

        a = a / tf.math.sqrt(tf.cast(Dh, dtype=tf.float32))

        if m is not None:
            a = tf.where(m, -float('inf'), a)

        a = tf.nn.softmax(a, axis=-1)
        a = self.dropout(a, training=training)

        o = tf.matmul(a, v)

        o = tf.transpose(o, perm=(0, 2, 1, 3))
        o = tf.reshape(o, (B, -1, D))
        o = self.W_o(o)

        return o, a