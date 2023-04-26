import numpy as np
import tensorflow as tf
import glob

import utils
import vit
import maskcut
from predictor import Predictor

def main():
    dir = '/Users/sumukhiganesan/OneDrive - Northeastern University/DS5230/Project'
    inSubDir = '/data/data_object_image_2'
    inFolder = '/training/image_2'

    outSubDir = '/output/maskcut'
    outFolder = '/training'

    inPath = dir + inSubDir + inFolder + '/*'
    outPath = dir + outSubDir + outFolder + '/'

    outFormat = 'JPEG'

    maskcutPredictor = Predictor()

    maskcutPredictor.setup()

    ret = False

    inputImages = glob.glob(inPath)
    
    for imgPath in inputImages:
        imgId = imgPath[-10:-4]
        
        ret = maskcutPredictor.predict(imgPath, outPath, model = 'base', n_pseudo_masks=6, tau = 0.15, save = True)

        if not ret:
            print("Maskcut failed for image: ", imgId)

if __name__ == "__main__":
    main()