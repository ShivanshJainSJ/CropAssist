# Super-Resolution Models

This directory is for storing super-resolution models used by the application to enhance low-resolution plant images.

## Download Instructions

For the EDSR super-resolution model used in this application, you can download the pre-trained model from:
https://github.com/opencv/opencv_contrib/tree/master/modules/dnn_superres/models

Specifically, you need:
- EDSR_x4.pb - For 4x upscaling

Place the downloaded .pb file in this directory.

## Usage

The application will automatically use this model when processing low-resolution images (smaller than 224x224 pixels). If the model file is not found, the application will fall back to using standard bicubic interpolation for upscaling.

## Alternative Models

Other super-resolution models compatible with OpenCV's dnn_superres module:
- ESPCN
- FSRCNN
- LapSRN

These can be used as alternatives if EDSR is too computationally intensive or if you wish to experiment with different approaches.
