
from numpy import dtype
import tensorflow as tf

from . import register_model
from tensorflowSeq2Seq.util.timer import Timer
from tensorflowSeq2Seq.util.debug import my_print
from tensorflowSeq2Seq.model.modules import (
    PositionalEmbedding,
    MultiHeadAttention,
    LayerNormalization,
    TiedDenseLayer,
    LogSoftmax
)

@register_model("Transformer")
class Transformer(tf.Module):

    def __init__(self, name='transformer', **kwargs):
        super(Transformer, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.encoder = TransformerEncoder(**kwargs)

            if self.tiew:
                word_embed = self.encoder.src_embed.object_to_time.word_embed
            else:
                word_embed = None

            self.decoder = TransformerDecoder(word_embed=word_embed,**kwargs)

    @staticmethod
    def create_model_from_config(config, vocab_src, vocab_tgt):
        
        model =  Transformer(
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
    @tf.Module.with_name_scope
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


class TransformerEncoder(tf.Module):

    def __init__(self, name='encoder', **kwargs):
        super(TransformerEncoder, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.src_embed  = Timer(PositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, name='embed_src'))
            
            self.enc_layers = []
            for l in range(1, self.encL+1):

                layer = TransformerEncoderLayer(name=f'encoder_layer_{l:0>2}', **kwargs)

                self.enc_layers.append(layer)

            self.lnorm = Timer(LayerNormalization(self.model_dim))

    @tf.Module.with_name_scope
    def __call__(self, src, training=False, src_mask=None, tgt_mask=None):

        h = self.src_embed(src, training)

        for layer in self.enc_layers:

            h = layer(h, training, src_mask=src_mask)

        h = self.lnorm(h)

        return h


class TransformerDecoder(tf.Module):

    def __init__(self, word_embed=None, name='decoder', **kwargs):
        super(TransformerDecoder, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.tgt_embed = Timer(PositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, word_embed=word_embed, name='embed_tgt'))

            self.dec_layers = []
            for l in range(self.decL):

                layer = TransformerDecoderLayer(name=f'decoder_layer_{l:0>2}', **kwargs)

                self.dec_layers.append(layer)

            self.lnorm              = Timer(LayerNormalization(self.model_dim))
            self.log_softmax        = Timer(LogSoftmax())

            if word_embed is None:
                self.output_projection  = Timer(tf.keras.layers.Dense(self.tgtV, name='output_projection'))
            else:
                self.output_projection  = Timer(TiedDenseLayer(self.tgt_embed.object_to_time.word_embed, self.tgtV, name='output_projection'))

    @tf.Module.with_name_scope
    def __call__(self, tgt, h, training=False, src_mask=None, tgt_mask=None):

        s = self.tgt_embed(tgt, training)

        for layer in self.dec_layers:

            s = layer(s, h, training, src_mask=src_mask, tgt_mask=tgt_mask)

        s = self.lnorm(s)
        s = self.output_projection(s)
        s = self.log_softmax(s, axis=-1)

        return s


class TransformerEncoderLayer(tf.Module):

    def __init__(self, name='encoder_layer', **kwargs):
        super(TransformerEncoderLayer, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:
        
            self.lnorm1     = Timer(LayerNormalization(self.model_dim, name='lnorm1'))
            self.att        = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout, name='self_att')
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


class TransformerDecoderLayer(tf.Module):


    def __init__(self, name='decoder_layer', **kwargs):
        super(TransformerDecoderLayer, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.lnorm1     = Timer(LayerNormalization(self.model_dim, name='lnorm1'))
            self.self_att   = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout, name='self_att')

            self.lnorm2     = Timer(LayerNormalization(self.model_dim, name='lnorm2'))
            self.cross_att  = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout, name='cross_att')

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