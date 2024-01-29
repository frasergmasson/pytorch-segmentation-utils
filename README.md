# Utilities for Performing Semantic Segmentation with Pytorch

These are a set of utility functions and classes that are useful when training semantic segmentation networks in Pytorch. I created these as part of my final year project at university, which invloved training a neural network to perform semantic segmentation on images of sea ice.
This repository was originally on my old account 'nofareyoker' which most of the commits are attributed to.

## ImageMaskDataset

This is an extension of the Pytorch Datset class which pairs images to corresponding masks. It also has functionality for applying different transformations to images and masks.

## Evaluations

A number of evaluation metrics are included in these utilities, namely: pixel accuracy, true positive count, intersect over union, and F1 score. Functions for calculating all of these metrics on individual classes are provided.
The MetricManager class can keep track of multiple different evaluation metrics over the course of training. A function that calculates a confusion matrix is also included.

## Mask To Image

This function converts a Pytorch tensor representing a pixel mask into an RGB image by assigning a colour to each class.

## Pair Transform

Classes that perform an identical transformation on an image and its mask. FunctionalPairTransform takes a deterministic transformation function and PairTransform takes a transformation with randomised parameters and ensures the same parameters are used to transform both image and mask.

## Weights
The 'calculate_class_weights' function gives a weight for each class in the dataset. The weight of a class is inversely proportional to its total pixel count within the dataset. 
