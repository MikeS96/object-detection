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


def rescale(a: float, L: float, U: float):
    if np.allclose(L, U):
        return 0.0
    return (a - L) / (U - L)


def vanilla_servoing_mask(mask, class2int):
    weighted_mask = np.zeros(mask.shape)
    weighted_mask[:] = (mask == class2int['white-lane']) + (mask == class2int['yellow-lane'])
    return weighted_mask


def obstables_servoing_mask(mask, class2int):
    # TODO add weights?
    weighted_mask = np.zeros(mask.shape)
    weighted_mask[:] = (mask == class2int['white-lane']) + (mask == class2int['duckiebot']) + \
                       (mask == class2int['duck']) + (mask == class2int['sign'])
    return weighted_mask
