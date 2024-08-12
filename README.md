# Implementing CutLER for Unsupervised Object Detection and Instance Segmentation

This repository contains the implementation of the CutLER approach for unsupervised object detection and instance segmentation. The project focuses on reproducing and adapting the methodology described in the 2023 Meta AI paper "Cut and Learn for Unsupervised Object Detection and Instance Segmentation," with specific application to the KITTI dataset.

## Overview

Object detection and segmentation are critical components in computer vision, particularly in domains like autonomous driving, robotics, and surveillance. Traditional methods require extensive labeled data, making them expensive and time-consuming to implement. CutLER (Cut-and-LEaRn) addresses this challenge by proposing an unsupervised zero-shot learning approach, which does not require labeled data for training.

This project reproduces the CutLER method using the KITTI dataset and translates the MaskCut mechanism from PyTorch to TensorFlow, showcasing the method's applicability to real-world scenarios such as autonomous driving.

## Features

- **Unsupervised Object Detection**: Detects and segments objects without requiring labeled data.
- **KITTI Dataset**: Implements and tests the approach on the widely used KITTI Vision Benchmark Suite, a dataset designed for autonomous driving research.
- **TensorFlow Implementation**: Translates the MaskCut mechanism from the original PyTorch framework to TensorFlow.
- **Detectron2**: Utilizes the pre-trained Cascade Mask R-CNN model with ResNet50 FPN and DINO weights for object detection.

## Project Structure

- **Dataset Selection and Processing**: Preprocessing and selection of the KITTI dataset for object detection tasks.
- **MaskCut Implementation**: Translation of the MaskCut process from PyTorch to TensorFlow.
- **Detector Integration**: Integration of the MaskCut masks with the pre-trained Cascade Mask R-CNN model using Detectron2.
- **Demonstration Code**: Google Colab notebooks and scripts to run and test the model on KITTI images.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow 2.x
- Detectron2
- Other dependencies specified in the `requirements.txt` file

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/Implementing-CutLER.git
   ```
2. Install the required Python packages:
  ```bash
  pip install -r requirements.txt
  ```
3. Download the KITTI dataset and place it in the appropriate directory.

### Usage
1. Preprocess the Dataset: Run the preprocessing script to prepare the KITTI dataset for object detection.
  ```bash
  python preprocess_kitti.py
  ```
2. Run the Model: Use the provided Jupyter notebook or Python script to execute the CutLER method on the KITTI dataset.
  ```bash
  jupyter notebook Implementing_CutLER_Detector.ipynb
  ```
3. Evaluate the Results: The output will include images with overlayed masks and bounding boxes, demonstrating the object detection and segmentation results.

### Results

The model demonstrates high accuracy in pixel assignment to multiple objects within the KITTI images, showcasing the scalability of the CutLER approach for different use cases.

### Contributing

Contributions are welcome! Please fork the repository and create a pull request for any improvements or new features.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Acknowledgments

- Meta AI for the original CutLER method and the related research paper.
- The KITTI Vision Benchmark Suite for providing the dataset.
- Detectron2 by Facebook AI Research (FAIR) for the object detection framework.
