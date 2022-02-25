#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np

def get_steer_matrix_left_lane_markings(shape):
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_left_lane: The steering (angular rate) matrix for Braitenberg-like control
                                    using the masked left lane markings (numpy.ndarray)
    """
    h, w = shape
    w_half = w // 2
    steer_matrix_left_lane = np.zeros(shape=shape, dtype="float32")
    # Steer matrix
    steer_matrix_left_lane[int(h * 5 / 8):, :w_half] = -0.01

    return steer_matrix_left_lane

def get_steer_matrix_right_lane_markings(shape):
    """
        Args:
            shape: The shape of the steer matrix (tuple of ints)
        Return:
            steer_matrix_right_lane: The steering (angular rate) matrix for Braitenberg-like control
                                     using the masked right lane markings (numpy.ndarray)
    """
    h, w = shape
    w_half = w // 2
    steer_matrix_right_lane = np.zeros(shape=shape, dtype="float32")
    # Steer matrix
    steer_matrix_right_lane[int(h * 5 / 8):, w_half:] = 0.01

    return steer_matrix_right_lane


def detect_lane_markings(mask):
    """
        Args:
            mask: Segmentation result after masking (numpy.ndarray)
        Return:
            left_masked_img:   Masked image for the dashed-yellow line (numpy.ndarray)
            right_masked_img:  Masked image for the solid-white line (numpy.ndarray)
    """

    h, w = mask.shape

    # ####### Edge based masking #######
    mask_left = np.ones(mask.shape)
    mask_left[:, int(np.floor(w / 2)):w + 1] = 0
    mask_right = np.ones(mask.shape)
    mask_right[:, 0:int(np.floor(w / 2))] = 0

    # ####### Final edge masking #######
    mask_left_edge = mask * mask_left
    mask_right_edge = mask * mask_right

    return mask_left_edge, mask_right_edge


def DT_TOKEN():
    # todo change this to your duckietown token
    dt_token = "dt1-3nT8KSoxVh4MdLnE1Bq2mTkhRpbR35G8mmbjExKF7zGm6g4-43dzqWFnWd8KBa1yev1g3UKnzVxZkkTbfYtfGWrfSxeihNZvYVNfNmnCBP28LeqDxL"
    return dt_token

def MODEL_NAME():
    # todo change this to your model's name that you used to upload it on google colab.
    # if you didn't change it, it should be "yolov5"
    return "1_mlp_frozen_42"

# In[2]:


def NUMBER_FRAMES_SKIPPED():
    # todo change this number to drop more frames
    # (must be a positive integer)
    return 5

# In[3]:


# `class` is the class of a prediction
def filter_by_classes(clas):
    # Right now, this returns True for every object's class
    # Change this to only return True for duckies!
    # In other words, returning False means that this prediction is ignored.
    return True

# In[4]:


# `scor` is the confidence score of a prediction
def filter_by_scores(scor):
    # Right now, this returns True for every object's confidence
    # Change this to filter the scores, or not at all
    # (returning True for all of them might be the right thing to do!)
    return True

# In[5]:


# `bbox` is the bounding box of a prediction, in xyxy format
# So it is of the shape (leftmost x pixel, topmost y pixel, rightmost x pixel, bottommost y pixel)
def filter_by_bboxes(bbox):
    # Like in the other cases, return False if the bbox should not be considered.
    return True

