import numpy as np
import tensorflow as tf

import utils
import vit
import maskcut
from predictor import Predictor

def main():
    dir = '/Users/sumukhiganesan/OneDrive - Northeastern University/'
    imgPath = dir + '000027.jpg' #'DS5230/Project/data/data_object_image_2/training/image_2/' + '000020.png'
    outPath = dir + 'Output.jpg'

    pred = Predictor()

    pred.setup()

    ret = False

    ret = pred.predict(imgPath, outPath, model = 'base', n_pseudo_masks=3, tau = 0.15, demo = True)

    if not ret:
        print("Maskcut failed!")

if __name__ == "__main__":
    main()