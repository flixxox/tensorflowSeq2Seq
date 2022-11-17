
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


@register_model("NNGram")
class NNGram(tf.Module):

    def __init__(self, name='NNGram', **kwargs):
        super(NNGram, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.encoder = NNGramEncoder(**kwargs)

            if self.tiew:
                word_embed = self.encoder.src_embed.object_to_time.word_embed
            else:
                word_embed = None

            self.decoder = NNGramDecoder(word_embed=word_embed,**kwargs)

    @staticmethod
    def create_model_from_config(config, vocab_src, vocab_tgt):
        
        model =  NNGram(
            srcV = vocab_src.vocab_size,
            tgtV = vocab_tgt.vocab_size,
            pad_index = vocab_src.PAD,
            encL = config['encL'],
            decL = config['decL'],
            model_dim = config['model_dim'],
            nHeads = config['nHeads'],
            ff_dim = config['ff_dim'],
            dropout = config['dropout'],
            maxI = config['max_sentence_length'],
            tiew = config['tiew'],
            initializer = config['initializer'],
            N = config['N'],
            nn_gram_encoder = config['nn_gram_encoder'],
            layers_with_nn_gram_encoder = config['layers_with_nn_gram_encoder'],
            nn_gram_decoder = config['nn_gram_decoder'],
            layers_with_nn_gram_decoder = config['layers_with_nn_gram_decoder'],
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
    def __call__(self, src, tgt, training=False, src_mask=None, src_mask_nngram=None, tgt_mask=None):
        
        h = self.encoder(src, training=training, src_mask=src_mask, src_mask_nngram=src_mask_nngram)

        s = self.decoder(tgt, h, training=training, src_mask=src_mask, src_mask_nngram=src_mask_nngram, tgt_mask=tgt_mask)

        return s, h

    @tf.function
    def create_masks(self, src, tgt, pad_index):

        tgtT = tf.shape(tgt)[1]

        src_mask = (src == pad_index)

        src_mask_nngram = tf.expand_dims(src_mask, axis=-1)
        src_mask_nngram = tf.cast(src_mask_nngram, dtype=tf.bool)

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
            'src_mask_nngram': src_mask_nngram,
            'tgt_mask': tgt_mask
        }, out_mask


class NNGramEncoder(tf.Module):

    def __init__(self, name='encoder', **kwargs):
        super(NNGramEncoder, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.src_embed  = Timer(PositionalEmbedding(self.srcV, self.model_dim, self.maxI, self.dropout, name='embed_src'))
            
            self.enc_layers = []
            for l in range(1, self.encL+1):

                if self.nn_gram_encoder and l+1 <= self.layers_with_nn_gram_encoder:

                    layer = NNGramEncoderLayer(name=f'encoder_layer_{l:0>2}', **kwargs)
                
                else:

                    layer = TransformerEncoderLayer(name=f'encoder_layer_{l:0>2}', **kwargs)

                self.enc_layers.append(layer)

            self.lnorm = Timer(LayerNormalization(self.model_dim))

    @tf.Module.with_name_scope
    def __call__(self, src, training=False, src_mask=None, src_mask_nngram=None, tgt_mask=None):

        h = self.src_embed(src, training)

        for layer in self.enc_layers:

            h = layer(h, training, src_mask=src_mask, src_mask_nngram=src_mask_nngram)

        h = self.lnorm(h)

        return h


class NNGramDecoder(tf.Module):

    def __init__(self, word_embed=None, name='decoder', **kwargs):
        super(NNGramDecoder, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.tgt_embed = Timer(PositionalEmbedding(self.tgtV, self.model_dim, self.maxI, self.dropout, word_embed=word_embed, name='embed_tgt'))

            self.dec_layers = []
            for l in range(self.decL):

                if self.nn_gram_decoder and l+1 <= self.layers_with_nn_gram_decoder:

                    layer = NNGramDecoderLayer(name=f'decoder_layer_{l:0>2}', **kwargs)
                
                else:

                    layer = TransformerDecoderLayer(name=f'decoder_layer_{l:0>2}', **kwargs)

                self.dec_layers.append(layer)

            self.lnorm              = Timer(LayerNormalization(self.model_dim))
            self.log_softmax        = Timer(LogSoftmax())

            if word_embed is None:
                self.output_projection  = Timer(tf.keras.layers.Dense(self.tgtV, name='output_projection'))
            else:
                self.output_projection  = Timer(TiedDenseLayer(word_embed, self.tgtV, name='output_projection'))

    @tf.Module.with_name_scope
    def __call__(self, tgt, h, training=False, src_mask=None, src_mask_nngram=None, tgt_mask=None):

        s = self.tgt_embed(tgt, training)

        for layer in self.dec_layers:

            s = layer(s, h, training, src_mask=src_mask, src_mask_nngram=src_mask_nngram, tgt_mask=tgt_mask)

        s = self.lnorm(s)
        s = self.output_projection(s)
        s = self.log_softmax(s, axis=-1)

        return s


class NNGramEncoderLayer(tf.Module):

    def __init__(self, name='encoder_layer', **kwargs):
        super(NNGramEncoderLayer, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:
        
            self.lnorm1     = Timer(LayerNormalization(self.model_dim, name='lnorm1'))
            self.att        = Timer(EncoderNNGramAttention(self.model_dim, self.nHeads, self.N, self.dropout, self.pad_index))
            self.dropout    = Timer(tf.keras.layers.Dropout(self.dropout, name='drop'))

            self.lnorm2     = Timer(LayerNormalization(self.model_dim, name='lnorm2'))
            self.ff1        = Timer(tf.keras.layers.Dense(self.ff_dim, activation='relu', name='ff1'))
            self.ff2        = Timer(tf.keras.layers.Dense(self.model_dim, name='ff2'))

    @tf.Module.with_name_scope
    def __call__(self, x, training, src_mask=None, src_mask_nngram=None):
        
        r = x
        x = self.lnorm1(x)
        x = self.att(x, training, m=src_mask_nngram)
        x = self.dropout(x, training=training)
        x = x + r

        r = x
        x = self.lnorm2(x)
        x = self.ff1(x)
        x = self.ff2(x)
        x = self.dropout(x, training=training)
        x = x + r

        return x


class NNGramDecoderLayer(tf.Module):

    def __init__(self, name='decoder_layer', **kwargs):
        super(NNGramDecoderLayer, self).__init__(name=name)

        for k, v in kwargs.items():
            setattr(self, k, v)

        with self.name_scope:

            self.lnorm1     = Timer(LayerNormalization(self.model_dim, name='lnorm1'))
            self.self_att   = Timer(DecoderNNGramAttention(self.model_dim, self.nHeads, self.N, self.dropout, self.pad_index))

            self.lnorm2     = Timer(LayerNormalization(self.model_dim, name='lnorm2'))
            self.cross_att  = MultiHeadAttention(self.nHeads, self.model_dim, self.dropout, name='cross_att')

            self.lnorm3     = Timer(LayerNormalization(self.model_dim, name='lnorm1'))
            self.ff1        = Timer(tf.keras.layers.Dense(self.ff_dim, activation='relu', name='ff1'))
            self.ff2        = Timer(tf.keras.layers.Dense(self.model_dim, name='ff2'))
            self.dropout    = Timer(tf.keras.layers.Dropout(self.dropout, name='drop'))

    @tf.Module.with_name_scope
    def __call__(self, s, h, training, src_mask=None, src_mask_nngram=None, tgt_mask=None):

        r = s
        s = self.lnorm1(s)
        s = self.self_att(s, training)
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
    def __call__(self, x, training, src_mask=None, src_mask_nngram=None):
        
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

            self.lnorm3     = Timer(LayerNormalization(self.model_dim, name='lnorm1'))
            self.ff1        = Timer(tf.keras.layers.Dense(self.ff_dim, activation='relu', name='ff1'))
            self.ff2        = Timer(tf.keras.layers.Dense(self.model_dim, name='ff2'))
            self.dropout    = Timer(tf.keras.layers.Dropout(self.dropout, name='drop'))

    @tf.Module.with_name_scope
    def __call__(self, s, h, training, src_mask=None, src_mask_nngram=None, tgt_mask=None):

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


class DecoderNNGramAttention(tf.Module):

    def __init__(self, model_dim, nHeads, N, dropout, pad_index, name='nngram_attention'):
        super(DecoderNNGramAttention, self).__init__(name=name)

        self.pad_index = pad_index
        self.N = N
        self.D = model_dim
        self.H = nHeads
        self.Dh = self.D // self.H
        
        self.W_k = tf.Variable(
            tf.ones((self.H, self.Dh, self.N*self.Dh), dtype=tf.float32), # Will be initialized later
            trainable=True
        )

        self.dropout = tf.keras.layers.Dropout(dropout, name='drop')

        self.W = tf.keras.layers.Dense(self.D, name='W')

    @tf.Module.with_name_scope
    def __call__(self, x, training):

        B   = tf.shape(x)[0]
        I   = tf.shape(x)[1]
        N   = self.N
        H   = self.H
        D   = self.D
        Dh  = self.Dh

        x = tf.reshape(x, (B, -1, H, Dh))
        x = tf.transpose(x, perm=(0, 2, 1, 3))


        # Add N-1 padding vectors to the front of the sentence
        padding = tf.ones((N-1, Dh), dtype=x.dtype) * self.pad_index
        padding = tf.repeat(tf.expand_dims(padding, axis=0), H, axis=0)
        padding = tf.repeat(tf.expand_dims(padding, axis=0), B, axis=0)

        x = tf.concat((padding, x), axis=2)

        indices = tf.expand_dims(tf.range(I), axis=1)
        indices = tf.repeat(indices, N, axis=1) + tf.range(N)

        x = tf.gather(x, indices, axis=2)           # Shape [B, H, I, N, Dh]
        x = tf.reshape(x, (B, H, I, N*Dh))          # Shape [B, H, I, N*Dh]
        x = tf.transpose(x, perm=(0, 2, 1, 3))      # Shape [B, I, H, N*Dh]

        x = tf.linalg.matvec(self.W_k, x)           # Shape [B, I, H, Dh]


        x = tf.reshape(x, (B, I, D))

        x = self.dropout(x)

        x = self.W(x)

        return x


class EncoderNNGramAttention(tf.Module):

    def __init__(self, model_dim, nHeads, N, dropout, pad_index, name='nngram_attention'):
        super(EncoderNNGramAttention, self).__init__(name=name)

        self.pad_index = pad_index
        self.N = 2*N-1
        self.D = model_dim
        self.H = nHeads
        self.Dh = self.D // self.H
        
        self.W_k = tf.Variable(
            tf.ones((self.H, self.Dh, self.N*self.Dh), dtype=tf.float32), # Will be initialized later
            trainable=True
        )

        self.dropout = tf.keras.layers.Dropout(dropout, name='drop')

        self.W = tf.keras.layers.Dense(self.D, name='W')

    @tf.Module.with_name_scope
    def __call__(self, x, training, m=None):

        B   = tf.shape(x)[0]
        I   = tf.shape(x)[1]
        N   = self.N
        H   = self.H
        D   = self.D
        Dh  = self.Dh

        if m is not None:
            x = tf.where(m, 0., x)

        x = tf.reshape(x, (B, -1, H, Dh))
        x = tf.transpose(x, perm=(0, 2, 1, 3))


        # Add N-1 padding vectors to the front and back of the sentence
        padding = tf.ones((N-1, Dh), dtype=x.dtype) * self.pad_index
        padding = tf.repeat(tf.expand_dims(padding, axis=0), H, axis=0)
        padding = tf.repeat(tf.expand_dims(padding, axis=0), B, axis=0)

        x = tf.concat((padding, x, padding), axis=2)

        indices = tf.expand_dims(tf.range(I), axis=1)
        indices = tf.repeat(indices, N, axis=1) + tf.range(N)

        x = tf.gather(x, indices, axis=2)           # Shape [B, H, I, N, Dh]
        x = tf.reshape(x, (B, H, I, N*Dh))          # Shape [B, H, I, N*Dh]
        x = tf.transpose(x, perm=(0, 2, 1, 3))      # Shape [B, I, H, N*Dh]

        x = tf.linalg.matvec(self.W_k, x)           # Shape [B, I, H, Dh]


        x = tf.reshape(x, (B, I, D))

        x = self.dropout(x)

        x = self.W(x)

        return x