import tensorflow as tf
from transformers import TFViTModel


def vit_small(patch_size=16, **kwargs):
    if patch_size == 8:
        model = TFViTModel.from_pretrained('facebook/dino-vits8', from_pt = True)
    elif patch_size == 16:
        model = TFViTModel.from_pretrained('facebook/dino-vits16', from_pt = True)
    return model


def vit_base(patch_size=16, **kwargs):
    if patch_size == 8:
        model = TFViTModel.from_pretrained('facebook/dino-vitb8', from_pt = True)
    elif patch_size == 16:
        model = TFViTModel.from_pretrained('facebook/dino-vitb16', from_pt = True)
    return model


class ViTFeat(tf.keras.Model):
    """ Vision Transformer """
    def __init__(self, feat_dim, vit_arch='base', vit_feat='k', patch_size=16):
        super().__init__()

        if vit_arch == 'base':
            model_name = 'facebook/dino-vit-base'
            model = vit_base(patch_size=patch_size)
        else:
            model_name = 'facebook/dino-vit-small'
            model = vit_small(patch_size=patch_size)

        self.model = model
        self.feat_dim = feat_dim
        self.vit_feat = vit_feat
        self.patch_size = patch_size

    def call(self, img):
        feat_out = {}
        outputs = self.model(img, output_attentions = True, output_hidden_states = True)

        # feat_out['qkv'] = tf.stack([self.model.layers[-1].encoder.layer[-1].attention.self_attention.query(inputs = outputs.last_hidden_state),
        # self.model.layers[-1].encoder.layer[-1].attention.self_attention.key(inputs = outputs.last_hidden_state),
        # self.model.layers[-1].encoder.layer[-1].attention.self_attention.value(inputs = outputs.last_hidden_state)])
        layerInput = self.model.vit.encoder.layer[-1].layernorm_before(outputs.hidden_states[-1])
        raw_q = self.model.vit.encoder.layer[-1].attention.self_attention.query(layerInput)[0]
        raw_k = self.model.vit.encoder.layer[-1].attention.self_attention.key(layerInput)[0]
        raw_v = self.model.vit.encoder.layer[-1].attention.self_attention.value(layerInput)[0]

        feat_out['qkv'] = tf.expand_dims(tf.concat([tf.concat([raw_q, raw_k], axis=1), raw_v], axis=1), 0)

        # Forward pass in the model
        
        bs, nb_head, nb_token, _ = outputs.attentions[-1].shape
        qkv = tf.transpose(tf.reshape(feat_out['qkv'], (bs, nb_token, 3, nb_head, -1)), (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]

        # k = tf.linalg.matmul(outputs.attentions[-1], outputs.last_hidden_state)
        
        q = tf.reshape(tf.transpose(q, perm=[0, 2, 1, 3]), (bs, nb_token, -1))
        k = tf.reshape(tf.transpose(k, perm=[0, 2, 1, 3]), (bs, nb_token, -1))
        v = tf.reshape(tf.transpose(v, perm=[0, 2, 1, 3]), (bs, nb_token, -1))

        h, w = img.shape[2], img.shape[3]
        feat_h, feat_w = h // self.patch_size, w // self.patch_size

        if self.vit_feat == "k":
            feats = tf.reshape(tf.transpose(k[:, 1:], perm=[0, 2, 1]), (bs, self.feat_dim, feat_h * feat_w))
        elif self.vit_feat == "q":
            feats = tf.reshape(tf.transpose(q[:, 1:], perm=[0, 2, 1]), (bs, self.feat_dim, feat_h * feat_w))
        elif self.vit_feat == "v":
            feats = tf.reshape(tf.transpose(v[:, 1:], perm=[0, 2, 1]), (bs, self.feat_dim, feat_h * feat_w))
        elif self.vit_feat == "kqv":
            k = tf.transpose(k[:, 1:], perm=[0, 2, 1])
            q = tf.transpose(q[:, 1:], perm=[0, 2, 1])
            v = tf.transpose(v[:, 1:], perm=[0, 2, 1])
            feats = tf.reshape(tf.concat([k, q, v], axis=1), (bs, self.feat_dim, feat_h * feat_w))

        return feats