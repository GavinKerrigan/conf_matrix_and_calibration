#!/usr/local/bin/python3
# Copyright (c) 2021 Robert Bosch GmbH Copyright holder of the paper "Multi-Class Uncertainty Calibration via Mutual Information Maximization-based Binning" accepted at ICLR 2021.
# All rights reserved.
###
# The paper "Multi-Class Uncertainty Calibration via Mutual Information Maximization-based Binning" accepted at ICLR 2021.
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Author: Kanil Patel
# -*- coding: utf-8 -*-
'''
 utils.py
 imax_calib

 Created by Kanil Patel on 07/06/20.
 Copyright 2020. Kanil Patel. All rights reserved.
'''
import numpy as np
import sklearn.model_selection

#EPS = np.finfo(float).eps # used to avoid division by zero
EPS = 1e-50

def is_sorted(a):
    for i in range(a.size-1):
        if a[i+1] < a[i] :  return False
    return True


def to_softmax(x, axis=-1):
    """
    Stable softmax in numpy. Will be applied across last dimension by default.
    Takes care of numerical instabilities like division by zero or log(0).

    Parameters
    ----------
    x : numpy ndarray 
       Logits of the network as numpy array. 
    axis: int
       Dimension along which to apply the operation (default: last one) 

    Returns
    -------
    softmax: numpy ndarray 
       Softmax output 
    """
    z = x - np.max(x, axis=axis, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=axis, keepdims=True)
    softmax = numerator / denominator
    assert np.all( np.isfinite(softmax) ) == True , "Softmax output contains NaNs. Handle this."
    return softmax

def to_sigmoid(x): 
    """
    Stable sigmoid in numpy. Uses tanh for a more stable sigmoid function. 
    
    Parameters
    ----------
    x : numpy ndarray 
       Logits of the network as numpy array. 

    Returns
    -------
    sigmoid: numpy ndarray 
       Sigmoid output 
    """
    sigmoid = 0.5 + 0.5 * np.tanh(x/2)
    assert np.all( np.isfinite(sigmoid) ) == True , "Sigmoid output contains NaNs. Handle this."
    return sigmoid 

def to_logodds(x):
    """

    Convert probabilities to logodds using:

    .. math:: 
        \\log\\frac{p}{1-p} ~ \\text{where} ~ p \\in [0,1]

    Natural log.

    Parameters
    ----------
    x : numpy ndarray 
       Class probabilties as numpy array. 

    Returns
    -------
    logodds : numpy ndarray 
       Logodds output 

    """
    x = np.clip(x, 1e-10, 1 - 1e-10)
    assert x.max() <= 1 and x.min() >= 0
    numerator = x
    denominator = 1-x
    #numerator[numerator==0] = EPS
    #denominator[denominator==0] = EPS # 1-EPS is basically 1 so not stable!
    logodds = safe_log_diff(numerator, denominator, np.log) # logodds = np.log( numerator/denominator   )
    assert np.all(np.isfinite(logodds))==True, "Logodds output contains NaNs. Handle this."
    return logodds

def safe_log_diff(A, B, log_func=np.log):
    """
    Numerically stable log difference function. Avoids log(0). Will compute log(A/B) safely where the log is determined by the log_func
    """
    if np.isscalar(A):
        if A==0 and B==0:
            return log_func(EPS)
        elif A==0:
            return log_func(  EPS   )  - log_func(B)
        elif B==0:
            return log_func(  A   )  - log_func( EPS )
        else:
            return log_func(A) - log_func(B)
    else:
        # log(A) - log(B)
        with np.errstate(divide='ignore'):
            output = np.where(A==0, log_func(EPS), log_func(A) ) - np.where(B==0, log_func(EPS), log_func(B))
            output[ np.logical_or(A==0, B==0)] = log_func(EPS)
            assert np.all(np.isfinite(output))
        return output




def quick_logits_to_logodds(logits, probs=None):
    """
    Using the log-sum-exp trick can be slow to convert from logits to logodds. This function will use the faster prob_to_logodds if n_classes is large.
    """
    n_classes = logits.shape[-1]
    if n_classes <=100:   # n_classes are reasonable as use this slow way to get marginal
        logodds = logits_to_logodds(logits) 
    else: # imagenet case will always come here! 
        if probs is None:   probs = to_softmax(logits)
        logodds = probs_to_logodds(probs)
    return logodds

def probs_to_logodds(x):
    """
    Use probabilities to convert to logodds. Faster than logits_to_logodds.
    """
    assert x.max() <= 1 and x.min() >= 0
    logodds = to_logodds(x)
    assert np.all(np.isfinite(logodds))
    return logodds

def logits_to_logodds(x):
    """
    Convert network logits directly to logodds (without conversion to probabilities and then back to logodds) using:

    .. math:: 
        \\lambda_k=z_k-\\log\\sum\\nolimits_{k'\\not = k}e^{z_{k'}}

    Parameters
    ----------
    x: numpy ndarray 
       Network logits as numpy array 

    axis: int
        Dimension with classes

    Returns
    -------
    logodds : numpy ndarray 
       Logodds output 
    """
    n_classes = x.shape[1]
    all_logodds = []
    for class_id in range(n_classes):
        logodds_c = x[...,class_id][..., np.newaxis] - custom_logsumexp(  np.delete(x, class_id, axis=-1) , axis=-1)
        all_logodds.append(logodds_c.reshape(-1))
    logodds = np.stack( all_logodds, axis=1 )
    assert np.all(np.isfinite(logodds))
    return logodds

def custom_logsumexp(x, axis=-1):
    """
    Uses the log-sum-exp trick.

    Parameters
    ----------
    x: numpy ndarray 
       Network logits as numpy array 
    
    axis: int (default -1)
        axis along which to take the sum

    Returns
    -------
    out: numpy ndarray
        log-sum-exp of x along some axis
    """
    x_max = np.amax(x, axis=axis, keepdims=True)
    x_max[~np.isfinite(x_max)] = 0
    tmp = np.exp(x - x_max)
    s = np.sum(tmp, axis=axis, keepdims=True)
    s[s<=0] = EPS # only add epsilon when argument is zero
    out = np.log(s)
    out += x_max
    return out






def to_onehot(y, num_classes):
    """
    Convert 1D targets to one-hot repr.

    Parameters
    ----------
    y: numpy 1D-array 
        Array with sample target ids (i.e. 0 to <num_classes>-1)
    num_classes: int
        Number of classes

    Returns
    -------
    y_onehot: numpy ndarray
        One-hot representation of target labels
    """
    assert len(y.shape)==1
    y_onehot = np.eye(num_classes)[y]
    return y_onehot


def binary_convertor(logodds, y, cal_setting, class_idx):
    """
    Function to convert the logodds data (in multi-class setting) to binary setting. The following options are available:
    1) CW - slice out some class: cal_setting="CW" and class_idx is not None (int)
    2) top1 - max class for each sample: get the top1 prediction: cal_setting="top1" and class_idx is None
    3) sCW - merge marginal setting where data is combined: cal_setting="sCW" and class_idx is None
    """

    if cal_setting=="CW":
        assert class_idx is not None, "class_idx needs to be an integer to slice out class needed for CW calibration setting"
        logodds_c = logodds[..., class_idx]
        y_c = y[..., class_idx] if y is not None else None
    elif cal_setting=="top1":
        assert class_idx is None, "class_idx needs to be None - check"
        top1_indices = logodds.argmax(axis=-1)
        logodds_c = logodds[np.arange(top1_indices.shape[0]), top1_indices]
        y_c = y.argmax(axis=-1) == top1_indices if y is not None else None 
    elif cal_setting=="sCW":
        assert class_idx is None, "class_idx needs to be None - check"
        logodds_c = np.concatenate(logodds.T)
        y_c = np.concatenate(y.T) if y is not None else None
    else:
        raise Exception("Calibration setting (%s) not recognized!"%(cal_setting))
    
    return logodds_c, y_c












































