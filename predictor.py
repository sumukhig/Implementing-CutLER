import numpy as np
import PIL.Image as Image
from scipy import ndimage
from colormap import random_color
import tensorflow as tf
import json

import utils
import vit
from torch_crf import densecrf
from maskcut import maskcut, vis_mask
import save_outputs

from cog import BasePredictor, Input, Path

category_info = {
                "is_crowd": 0,
                "id": 1
}

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        # DINO pre-trained model
        vit_features = "k"
        self.patch_size = 8

        self.backbone_base = vit.ViTFeat(
            768,
            "base",
            vit_features,
            self.patch_size,
        )

        self.backbone_small = vit.ViTFeat(
            384,
            "small",
            vit_features,
            self.patch_size,
        )

    def predict(
        self,
        input_image_path: Path = Input(
            description="Input image path",
        ),
        output_path: Path = Input(
            description="Output folder",
        ),
        model: str = Input(
            description="Choose the model architecture",
            default="base",
            choices=["small", "base"]
        ),
        n_pseudo_masks: int = Input(
            description="The maximum number of pseudo-masks per image",
            default=3,
        ),
        tau: float = Input(
            description="Threshold used for producing binary graph",
            default=0.15,
        ),
        demo: bool = Input(
            description="Demo or not",
            default=False
        ),
    ) -> Path:
        """Run a single prediction on the model"""

        backbone = self.backbone_base if model == "base" else self.backbone_small

        if not demo:
            image_id = input_image_path[-10:-4]
            mskPath = output_path + '/masks/' + image_id + '.jpg'

        # MaskCut hyperparameters
        fixed_size = 224

        # get pesudo-masks with MaskCut
        bipartitions, _, I_new = maskcut(
            str(input_image_path),
            backbone,
            self.patch_size,
            tau,
            N=n_pseudo_masks,
            fixed_size=fixed_size,
        )

        I = Image.open(str(input_image_path)).convert("RGB")
        width, height = I.size
        pseudo_mask_list = []
        for idx, bipartition in enumerate(bipartitions):
            print(idx)
            # post-process pesudo-masks with CRF
            pseudo_mask = densecrf(np.array(I_new), bipartition)
            pseudo_mask = ndimage.binary_fill_holes(pseudo_mask >= 0.5).astype(float)

            # filter out the mask that have a very different pseudo-mask after the CRF
            mask1 = tf.convert_to_tensor(bipartition)
            mask2 = tf.convert_to_tensor(pseudo_mask)

            if utils.IoU(mask1, mask2) < 0.5:
                pseudo_mask = pseudo_mask * -1

            # construct binary pseudo-masks
            pseudo_mask[pseudo_mask < 0] = 0
            pseudo_mask = Image.fromarray(np.uint8(pseudo_mask * 255))
            pseudo_mask = np.asarray(pseudo_mask.resize((width, height)))

            pseudo_mask = pseudo_mask.astype(np.uint8)
            upper = np.max(pseudo_mask)
            lower = np.min(pseudo_mask)
            thresh = upper / 2.0
            pseudo_mask[pseudo_mask > thresh] = upper
            pseudo_mask[pseudo_mask <= thresh] = lower
            pseudo_mask_list.append(pseudo_mask)

            if not demo:
                segmentation_id = int(image_id) * 100 + idx
                json_name = 'annot_' + image_id + '_' + str(idx) + '.json'
                json_path = output_path + '/annotations/' + json_name

                annot = save_outputs.create_annotation_info(segmentation_id, image_id, category_info, pseudo_mask)

                if annot is not None:
                    with open(json_path, 'w') as output_json_file:
                        json.dump(annot, output_json_file)

        out = np.array(I)
        for pseudo_mask in pseudo_mask_list:
            out = vis_mask(out, pseudo_mask, random_color(rgb=True))

        out2 = np.array(I)
        for b in bipartitions:
            m = Image.fromarray(np.uint8(b * 255))
            m = np.asarray(m.resize((width, height)))
            out2 = vis_mask(out2, m, random_color(rgb=True))
        
        if demo:
            out.save(str(output_path), 'JPEG')
            p = '/Users/sumukhiganesan/OneDrive - Northeastern University/Output2.jpg'
            out2.save(str(p), 'JPEG')
        else:
            out.save(str(mskPath), 'JPEG')
            
        return True
