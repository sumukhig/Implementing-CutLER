import numpy as np
from PIL import Image
import tensorflow as tf
from scipy.linalg import eigh
import utils

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
        # print('i=', i)
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
        _, _, _, cc = utils.detect_box(bipartition, seed, dimensions, scales=scales, initial_im_size=init_image_size)
        pseudo_mask = np.zeros(dimensions)
        pseudo_mask[cc[0],cc[1]] = 1
        pseudo_mask = tf.convert_to_tensor(pseudo_mask)
        ps = pseudo_mask.shape[0]

        # check if the extra mask is heavily overlapped with the previous one or is too small.
        if i >= 1:
            ratio = tf.cast(tf.reduce_sum(pseudo_mask), tf.float32) / tf.cast(pseudo_mask.shape[0]*pseudo_mask.shape[1], tf.float32)
            if utils.IoU(current_mask, pseudo_mask) > 0.5 or ratio <= 0.01:
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
    imgResize, w, h, feat_w, feat_h = utils.resize_pil(imgNew, patch_size)
    imgTensor = tf.cast(tf.convert_to_tensor(imgResize), tf.float32)
    imgTensor = imgTensor / 255.0
    mean, std = utils.get_mean_and_std(imgTensor)
    # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    imgTensor = (imgTensor - mean) / std
    imgTensor = tf.transpose(imgTensor, perm=[2,0,1])
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