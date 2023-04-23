import numpy as np
from PIL import Image
from datasets import load_dataset
import tensorflow as tf
from transformers import TFViTModel
from scipy.linalg import eigh
from scipy import ndimage
# from crf import densecrf
from torch_crf import densecrf
import matplotlib.pyplot as plt

def detect_box(bipartition, seed,  dims, initial_im_size=None, scales=None, principle_object=True):

    w_featmap, h_featmap = dims
    objects, num_objects = ndimage.label(bipartition)
    cc = objects[np.unravel_index(seed, dims)]


    if principle_object:
        mask = np.where(objects == cc)
       # Add +1 because excluded max
        ymin, ymax = min(mask[0]), max(mask[0]) + 1
        xmin, xmax = min(mask[1]), max(mask[1]) + 1
        # Rescale to image size
        r_xmin, r_xmax = scales[1] * xmin, scales[1] * xmax
        r_ymin, r_ymax = scales[0] * ymin, scales[0] * ymax
        pred = [r_xmin, r_ymin, r_xmax, r_ymax]

        # Check not out of image size (used when padding)
        if initial_im_size:
            pred[2] = min(pred[2], initial_im_size[1])
            pred[3] = min(pred[3], initial_im_size[0])

        # Coordinate predictions for the feature space
        # Axis different then in image space
        pred_feats = [ymin, xmin, ymax, xmax]

        return pred, pred_feats, objects, mask
    else:
        raise NotImplementedError
             
def IoU(mask1, mask2):
    threshold = 0.5
    mask1, mask2 = tf.where(mask1 > threshold, 1, 0), tf.where(mask2 > threshold, 1, 0)
    intersection = tf.squeeze(tf.reduce_sum(tf.cast(mask1 * tf.cast(mask1==mask2, tf.int32), tf.float32), axis=[-1, -2]))
    union = tf.squeeze(tf.reduce_sum(tf.cast(mask1 + mask2, tf.float32), axis=[-1, -2]))
    return tf.reduce_mean(intersection / union).numpy()

def get_mean_and_std(x):
    mm, var=tf.nn.moments(x,axes=[0,1])
    return mm, var

def resize_pil(I, patch_size=16) : 
    w, h = I.size

    new_w, new_h = int(round(w / patch_size)) * patch_size, int(round(h / patch_size)) * patch_size
    feat_w, feat_h = new_w // patch_size, new_h // patch_size

    return I.resize((new_w, new_h), resample=Image.LANCZOS), w, h, feat_w, feat_h
    

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
        raw_q = self.model.vit.encoder.layer[11].attention.self_attention.query(layerInput)[0]
        raw_k = self.model.vit.encoder.layer[11].attention.self_attention.key(layerInput)[0]
        raw_v = self.model.vit.encoder.layer[11].attention.self_attention.value(layerInput)[0]

        feat_out['qkv'] = tf.expand_dims(tf.concat([tf.concat([raw_q, raw_k], axis=1), raw_v], axis=1), 0)

        # feat_out['qkv'] = tf.stack([self.model.vit.encoder.layer[-1].attention.self_attention.query(inputs = inputs),
        #                             self.model.vit.encoder.layer[-1].attention.self_attention.key(inputs = inputs),
        #                             self.model.vit.encoder.layer[-1].attention.self_attention.value(inputs = inputs)])

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

def get_affinity_matrix(feats, tau, eps=1e-5):
    # get affinity matrix via measuring patch-wise cosine similarity
    feats = tf.math.l2_normalize(feats, axis=0)
    A = tf.matmul(feats, feats, transpose_a=True).numpy()
    # convert the affinity matrix to a binary one.
    A = A > tau
    A = tf.where(tf.dtypes.cast(A, tf.float32) == 0, eps, A)
    d_i = tf.reduce_sum(tf.dtypes.cast(A, tf.float32), axis=1)
    D = tf.linalg.diag(d_i)
    return A, D

def second_smallest_eigenvector(A, D):
    # get the second smallest eigenvector from affinity matrix
    _, eigenvectors = eigh(D-A, D, subset_by_index=[1,2])
    eigenvec = np.copy(eigenvectors[:, 0])
    second_smallest_vec = eigenvectors[:, 0]
    return eigenvec, second_smallest_vec

def get_salient_areas(second_smallest_vec):
    # get the area corresponding to salient objects.
    avg = np.sum(second_smallest_vec) / len(second_smallest_vec)
    bipartition = second_smallest_vec > avg
    return bipartition

def check_num_fg_corners(bipartition, dims):
    # check number of corners belonging to the foreground
    bipartition_ = bipartition.reshape(dims)
    top_l, top_r, bottom_l, bottom_r = bipartition_[0][0], bipartition_[0][-1], bipartition_[-1][0], bipartition_[-1][-1]
    nc = int(top_l) + int(top_r) + int(bottom_l) + int(bottom_r)
    return nc

def get_masked_affinity_matrix(painting, feats, mask, ps):
    # mask out affinity matrix based on the painting matrix 
    dim, num_patch = feats.shape[0], feats.shape[1]
    painting = painting + tf.expand_dims(mask, 0)
    painting = tf.where(painting > 0, 1, 0)
    feats = tf.cast(tf.reshape(feats, [dim, ps, ps]), tf.float32)
    feats = tf.reshape(feats * tf.cast(1 - painting, tf.float32), [dim, num_patch])
    painting = tf.cast(painting, tf.float64)
    return feats, painting

def maskcut_forward(feats, dimensions, scales, init_image_size, tau=0, N=3):
    bipartitions = []
    eigvecs = []

    for i in range(N):
        print('i=', i)
        if i == 0:
            painting = np.zeros(dimensions)
            painting = tf.convert_to_tensor(painting)
        else:
            feats, painting = get_masked_affinity_matrix(painting, feats, current_mask, ps)
        
        # construct the affinity matrix
        A, D = get_affinity_matrix(feats, tau)
        # get the second smallest eigenvector
        eigenvec, second_smallest_vec = second_smallest_eigenvector(A, D)
        # get salient area
        bipartition = get_salient_areas(second_smallest_vec)
        # check if we should reverse the partition based on:
        # 1) peak of the 2nd smallest eigvec 2) object centric bias
        seed = np.argmax(np.abs(second_smallest_vec))
        nc = check_num_fg_corners(bipartition, dimensions)
        # print(nc)
        if nc >= 3:
            reverse = True
        else:
            reverse = bipartition[seed] != 1

        if reverse:
            # reverse bipartition, eigenvector and get new seed
            eigenvec = eigenvec * -1
            bipartition = np.logical_not(bipartition)
            seed = np.argmax(eigenvec)
        else:
            seed = np.argmax(second_smallest_vec)
        
        # get pixels corresponding to the seed
        bipartition = bipartition.reshape(dimensions).astype(int)
        _, _, _, cc = detect_box(bipartition, seed, dimensions, scales=scales, initial_im_size=init_image_size)
        pseudo_mask = np.zeros(dimensions)
        pseudo_mask[cc[0],cc[1]] = 1
        pseudo_mask = tf.convert_to_tensor(pseudo_mask)
        ps = pseudo_mask.shape[0]

        # check if the extra mask is heavily overlapped with the previous one or is too small.
        if i >= 1:
            ratio = tf.cast(tf.reduce_sum(pseudo_mask), tf.float32) / tf.cast(pseudo_mask.shape[0]*pseudo_mask.shape[1], tf.float32)
            if IoU(current_mask, pseudo_mask) > 0.5 or ratio <= 0.01:
                pseudo_mask = np.zeros(dimensions)
                pseudo_mask = tf.convert_to_tensor(pseudo_mask)
        
        current_mask = pseudo_mask

        # mask out foreground areas in previous stages
        masked_out = 0 if len(bipartitions) == 0 else np.sum(bipartitions, axis=0)
        bipartition = tf.squeeze(tf.image.resize(tf.expand_dims(tf.expand_dims(pseudo_mask, 0), -1), size=init_image_size, method='nearest'))
        bipartition_masked = bipartition.numpy() - masked_out
        bipartition_masked[bipartition_masked <= 0] = 0
        bipartitions.append(bipartition_masked)

        # unsample the eigenvec
        eigvec = second_smallest_vec.reshape(dimensions)
        eigvec = tf.convert_to_tensor(eigvec)
        eigvec = tf.squeeze(tf.image.resize(tf.expand_dims(tf.expand_dims(eigvec, 0), -1), size=init_image_size, method='nearest'))
        eigvecs.append(eigvec.numpy())

    return seed, bipartitions, eigvecs
    
def maskcut(img_path, backbone, patch_size, tau, N, fixed_size):
    bipartitions, eigvecs = [], []

    img = Image.open(img_path).convert('RGB')
    imgNew = img.resize((int(fixed_size), int(fixed_size)), Image.LANCZOS)
    imgResize, w, h, feat_w, feat_h = resize_pil(imgNew, patch_size)
    imgTensor = tf.cast(tf.convert_to_tensor(imgResize), tf.float32)
    imgTensor = imgTensor / 255.0
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    imgTensor = (imgTensor - mean) / std
    imgTensor = tf.transpose(imgTensor, perm=[2,0,1])
    # imgTensor = tf.reshape(tf.expand_dims(imgTensor, 0), (1, 3, h, w))
    imgTensor = tf.expand_dims(imgTensor, 0)

    feat = backbone(imgTensor)[0]

    _, bipartition, eigvec = maskcut_forward(feat, [feat_h, feat_w], [patch_size, patch_size], [h,w], tau, N=N)

    bipartitions += bipartition
    eigvecs += eigvec

    return bipartitions, eigvecs, imgNew

def vis_mask(input, mask, mask_color) :
    fg = mask > 0.5
    rgb = np.copy(input)
    rgb[fg] = (rgb[fg] * 0.3 + np.array(mask_color) * 0.7).astype(np.uint8)
    return Image.fromarray(rgb)

def main():
    img_path = '/Users/sumukhiganesan/OneDrive - Northeastern University/Img2.jpg'
    fixed_size = 224
    patch_size = 8
    vit_arch = 'base'
    vit_feat = 'k'
    tau = 0.15
    N = 6

    if vit_arch == 'base' and patch_size == 8:
        feat_dim = 768
    elif vit_arch == 'small' and patch_size == 8:
        feat_dim = 384

    backbone = ViTFeat(feat_dim, vit_arch, vit_feat, patch_size)
     
    bipartitions, _, imgNew = maskcut(img_path, backbone, patch_size, tau, N, fixed_size)

    pseudo_mask_list = []
    I = Image.open(img_path).convert('RGB')
    width, height = I.size

    for idx, bipartition in enumerate(bipartitions):
        # post-process pesudo-masks with CRF

        pseudo_mask = densecrf(np.array(imgNew), bipartition)

        pseudo_mask = ndimage.binary_fill_holes(pseudo_mask>=0.5).astype(float)

        mask1 = tf.convert_to_tensor(bipartition)
        mask2 = tf.convert_to_tensor(pseudo_mask)

        # print(pseudo_mask.any())

        print(IoU(mask1, mask2))

        if IoU(mask1, mask2) < 0.4:
            pseudo_mask = pseudo_mask * -1
    
        pseudo_mask[pseudo_mask < 0] = 0

        pseudo_mask = Image.fromarray(np.uint8(pseudo_mask*255))
        pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

        pseudo_mask = pseudo_mask.astype(np.uint8)
        upper = np.max(pseudo_mask)
        lower = np.min(pseudo_mask)
        thresh = upper / 2.0
        pseudo_mask[pseudo_mask > thresh] = upper
        pseudo_mask[pseudo_mask <= thresh] = lower
        pseudo_mask_list.append(pseudo_mask)


    input = np.array(I)
    for pseudo_mask in pseudo_mask_list:
        input = vis_mask(input, pseudo_mask, [0, 255, 0])
    plt.imshow(input)

if __name__ == "__main__":
    main()