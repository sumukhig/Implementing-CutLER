import numpy as np
from scipy import ndimage
import tensorflow as tf
from PIL import Image

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
    bool_mask1, bool_mask2 = tf.where(mask1 == 1, True, False), tf.where(mask2 == 1, True, False)
    intersection = tf.squeeze(tf.reduce_sum(tf.cast(mask1 * tf.cast(tf.math.logical_and(bool_mask1, bool_mask2), tf.int32), tf.float32), axis=[-1, -2]))
    union = tf.squeeze(tf.reduce_sum(tf.cast(tf.math.logical_or(bool_mask1, bool_mask2), tf.float32), axis=[-1, -2]))
    return (intersection / union).numpy()

def get_mean_and_std(x):
    mm, var=tf.nn.moments(x,axes=[0,1])
    return mm, var

def resize_pil(I, patch_size=16) : 
    w, h = I.size

    new_w, new_h = int(round(w / patch_size)) * patch_size, int(round(h / patch_size)) * patch_size
    feat_w, feat_h = new_w // patch_size, new_h // patch_size

    return I.resize((new_w, new_h), resample=Image.LANCZOS), w, h, feat_w, feat_h
    
