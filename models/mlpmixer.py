
from numpy import dtype
import tensorflow as tf

from . import register_model
from tensorflowSeq2Seq.util.timer import Timer
from tensorflowSeq2Seq.util.debug import my_print
from tensorflowSeq2Seq.model.modules import (
    PositionalEmbedding,
    LayerNormalization,
    TiedDenseLayer,
    LogSoftmax
)


@register_model("MLPMixer")
class MLPMixer(tf.Module):

    def __init__(self, name='mlpmixer', **kwargs):
        super(MLPMixer, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.encoder = MLPMixerEncoder(**kwargs)

            if self.tiew:
                word_embed = self.encoder.src_embed.object_to_time.word_embed
            else:
                word_embed = None

            self.decoder = MLPMixerDecoder(word_embed=word_embed,**kwargs)

    @staticmethod
    def create_model_from_config(config, vocab_src, vocab_tgt):
        
        model =  MLPMixer(
            srcV = vocab_src.vocab_size,
            tgtV = vocab_tgt.vocab_size,
            pad_index = vocab_src.PAD,
            encL = config['encL'],
            decL = config['decL'],
            model_dim = config['model_dim'],
            nHeads = config['nHeads'],
            dropout = config['dropout'],
            maxI = config['max_sentence_length'],
            tiew = config['tiew'],
            initializer = config['initializer'],
            variance_scaling_scale = config['variance_scaling_scale'],
            expansion_factor = config['expansion_factor']
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
        src_mask = tf.cast(src_mask, dtype=tf.bool)

        tgt_mask = tf.linalg.band_part(tf.ones((tgtT, tgtT)), -1, 0)
        tgt_mask = tf.expand_dims(tgt_mask, axis=0)
        tgt_mask = (tgt_mask == 0)

        out_mask = (tgt != pad_index)

        return {
            'src_mask': src_mask,
            'tgt_mask': tgt_mask
        }, out_mask


class MLPMixerEncoder(tf.Module):

    def __init__(self, name='encoder', **kwargs):
        super(MLPMixerEncoder, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.src_embed  = Timer(PositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, name='embed_src'))
            
            self.enc_layers = []
            for l in range(1, self.encL+1):

                layer = MLPMixerEncoderLayer(name=f'encoder_layer_{l:0>2}', **kwargs)

                self.enc_layers.append(layer)

            self.lnorm = Timer(LayerNormalization(self.model_dim))

    @tf.Module.with_name_scope
    def __call__(self, src, training=False, src_mask=None, tgt_mask=None):

        h = self.src_embed(src, training)

        for layer in self.enc_layers:

            h = layer(h, training, src_mask=src_mask)

        h = self.lnorm(h)

        return h


class MLPMixerDecoder(tf.Module):

    def __init__(self, word_embed=None, name='decoder', **kwargs):
        super(MLPMixerDecoder, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.tgt_embed = Timer(PositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, word_embed=word_embed, name='embed_tgt'))

            self.dec_layers = []
            for l in range(self.decL):

                layer = MLPMixerDecoderLayer(name=f'decoder_layer_{l:0>2}', **kwargs)

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


class MLPMixerEncoderLayer(tf.Module):

    def __init__(self, name='encoder_layer', **kwargs):
        super(MLPMixerEncoderLayer, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:
        
            self.lnorm1      = Timer(LayerNormalization(self.model_dim, name='lnorm1'))
            self.token_mlp   = Timer(TokenMixer(self.dropout, self.maxI, self.pad_index))

            self.lnorm2      = Timer(LayerNormalization(self.model_dim, name='lnorm2'))
            self.feature_mlp = Timer(FeatureMixer(self.dropout, self.model_dim, self.expansion_factor))

    @tf.Module.with_name_scope
    def __call__(self, x, training, src_mask=None):
        
        J = tf.shape(x)[1]

        r = x
        x = self.lnorm1(x)
        x = self.token_mlp(x, J, training, m1=src_mask)
        x = x + r

        r = x
        x = self.lnorm2(x)
        x = self.feature_mlp(x, training)
        x = x + r

        return x


class MLPMixerDecoderLayer(tf.Module):

    def __init__(self, name='decoder_layer', **kwargs):
        super(MLPMixerDecoderLayer, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.lnorm1      = Timer(LayerNormalization(self.model_dim, name='lnorm1'))
            self.dec_mlp     = Timer(TokenMixer(self.dropout, self.maxI, self.pad_index))

            self.enc_mlp     = Timer(TokenMixer(self.dropout, self.maxI, self.pad_index))

            self.lnorm3      = Timer(LayerNormalization(self.model_dim, name='lnorm3'))
            self.feature_mlp = Timer(FeatureMixer(self.dropout, self.model_dim, self.expansion_factor))

    @tf.Module.with_name_scope
    def __call__(self, s, h, training, src_mask=None, tgt_mask=None):

        I = tf.shape(s)[1]

        r = s
        s = self.lnorm1(s)
        s = self.dec_mlp(s, I, training, m1=tgt_mask, m2=tgt_mask)
        s = s + r

        r = s
        s = self.enc_mlp(h, I, training, m1=src_mask)
        s = s + r

        r = s
        s = self.lnorm3(s)
        s = self.feature_mlp(s, training)
        s = s + r

        return s


class TokenMixer(tf.Module):

    def __init__(self, dropout, maxI, pad_index, name='token_mixer'):
        super(TokenMixer, self).__init__(name=name)

        self.pad_index  = pad_index
        self.maxI       = maxI

        self.W1 = tf.Variable(
            tf.ones((maxI, maxI), dtype=tf.float32), # Will be initialized later
            trainable=True
        )

        self.dropout1 = tf.keras.layers.Dropout(dropout, name='drop1')
        
        self.W2 = tf.Variable(
            tf.ones((maxI, maxI), dtype=tf.float32), # Will be initialized later
            trainable=True
        )

        self.dropout2 = tf.keras.layers.Dropout(dropout, name='drop2')

    @tf.Module.with_name_scope
    def __call__(self, x, L_out, training, m1=None, m2=None):

        B    = tf.shape(x)[0]
        L_in = tf.shape(x)[1]
        D    = tf.shape(x)[2]

        W = self.W1[:L_out,:L_in]

        if m1 is not None:
            W = tf.repeat(tf.expand_dims(W, axis=0), B, axis=0)
            W = tf.where(m1, 0., W)

        x = tf.matmul(W, x) # Shape [B, L_out, D]
        x = tf.keras.activations.relu(x)
        x = self.dropout1(x, training=training)

        W = self.W2[:L_out,:L_out]

        if m2 is not None:
            W = tf.repeat(tf.expand_dims(W, axis=0), B, axis=0)
            W = tf.where(m2, 0., W)

        x = tf.matmul(W, x) # Shape [B, L_out, D]
        x = self.dropout2(x, training=training)

        return x


class FeatureMixer(tf.Module):

    def __init__(self, dropout, model_dim, expansion_factor, name='feature_mixer'):
        super(FeatureMixer, self).__init__(name=name)

        self.W1         = tf.keras.layers.Dense(model_dim*expansion_factor, name='W1')
        self.dropout1   = tf.keras.layers.Dropout(dropout, name='drop1')
        self.W2         = tf.keras.layers.Dense(model_dim, name='W2')
        self.dropout2   = tf.keras.layers.Dropout(dropout, name='drop2')

    @tf.Module.with_name_scope
    def __call__(self, x, training):

        x = self.W1(x)
        x = tf.keras.activations.relu(x)
        x = self.dropout1(x, training=training)

        x = self.W2(x)
        x = self.dropout2(x, training=training)

        return x

