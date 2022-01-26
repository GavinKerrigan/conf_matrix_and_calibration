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
 calibration_metrics.py
 evaluations

 Created by Kanil Patel on 07/27/20.
 Copyright 2020. Kanil Patel. All rights reserved.
'''
import numpy as np
import imax_calib.hb_utils as hb_utils
import imax_calib.utils as utils
import imax_calib.io as io
from scipy.cluster.vq import kmeans,vq
import scipy.cluster.vq
import os
import contextlib

from imax_calib.calibrators.binners import run_imax



def compute_top_1_and_CW_ECEs(multi_cls_probs, multi_cls_labels, list_approximators=["dECE", "mECE", "iECE", "kECE"], num_bins=100, threshold_mode='class'):
    """
    Given the multi-class predictions and labels, this function computes the top1 and CW ECEs. Will compute it by calling the other functions in this script.

    Parameters:
    -----------
    multi_cls_probs: 2D ndarray
        predicted probabilities
    multi_cls_labels: 1D or 2D ndarray
        label indices or one-hot labels. Will be converted to one-hot

    Return:
    -------
    ece_dict: dict
        Dictionary with all the ECE estimates

    """
    assert len(multi_cls_probs.shape)==2
    if len(multi_cls_labels.shape)==1: # not one-hot. so convert to one-hot
        multi_cls_labels = np.eye(multi_cls_probs.shape[1])[multi_cls_labels]

    ece_evals_dict = io.AttrDict({})
    
    n_classes = multi_cls_probs.shape[1]
    for ece_approx in list_approximators:
        top_1_preds  = multi_cls_probs.max(axis=-1)
        top_1_correct=multi_cls_probs.argmax(axis=-1) == multi_cls_labels.argmax(axis=-1)

        top_1_ECE = eval("measure_%s_calibration"%(ece_approx))(pred_probs=top_1_preds, correct=top_1_correct, num_bins=num_bins)["ece"]
    
        cw_ECEs = []
        if threshold_mode == 'class':
            threshold = 1.0/n_classes
        elif threshold_mode is None:
            threshold = 0.
        for class_idx in range(n_classes):
            cw_ECE = eval("measure_%s_calibration"%(ece_approx))(pred_probs=multi_cls_probs[:, class_idx],
                                                                 correct=multi_cls_labels[:, class_idx],
                                                                 num_bins=num_bins, threshold=threshold)["ece"]
            cw_ECEs.append(cw_ECE)
        mean_cw_ECE = np.mean(cw_ECEs)
        
        ece_evals_dict["top_1_%s"%(ece_approx)] = top_1_ECE
        ece_evals_dict["cw_%s"%(ece_approx)] = mean_cw_ECE 

    return ece_evals_dict


def _ece(avg_confs, avg_accs, counts):
    """
    Helper function to compute the Expected Calibration Error.

    Parameters
    ----------
    avg_confs: Averaged probability of predictions per bin (confidence)
    avg_accs: Averaged true accuracy of predictions per bin
    counts: Number of predictions per bin

    Returns
    -------
    ece: float - calibration error
    """
    return np.sum((counts / counts.sum()) * np.absolute(avg_confs- avg_accs))


def measure_iECE_calibration(pred_probs, correct, num_bins, threshold=-1):
    """
    Compute the calibration curve using I-Max binning scheme. This will run the I-Max algorithm on the TEST set and get the bin boundaries.

    Parameters
    ----------
    y: numpy binary array
        label indicating if sample is positive or negative
    
    for rest see calibration_error_and_curve()

    Returns
    -------
        see calibration_error_and_curve()

    """
    #print("Running iECE calc.: calling I-Max now!")
    logodds = utils.to_logodds(pred_probs)
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        logdata = run_imax(logodds, correct, num_bins, log_every_steps=None, logfpath=None )
    bin_boundaries = logdata.bin_boundaries[-1]
    assigned = hb_utils.bin_data(logodds, bin_boundaries)
    return calibration_error_and_curve(pred_probs, correct, assigned, num_bins, threshold)

def measure_dECE_calibration(pred_probs, correct, num_bins=100, threshold=-1):
    """
    Compute the calibration curve using the equal size binning scheme (i.e. equal size bins)and computes the calibration error given this binning scheme (i.e. dECE).

    Parameters
    ----------
        see calibration_error_and_curve()
    Returns
    -------
        see calibration_error_and_curve()

    """
    assert len(pred_probs.shape)==1
    bin_boundaries_prob = utils.to_sigmoid( hb_utils.nolearn_bin_boundaries(num_bins, binning_scheme="eqsize") )
    assigned = hb_utils.bin_data(pred_probs, bin_boundaries_prob)
    return calibration_error_and_curve(pred_probs, correct, assigned, num_bins, threshold)


def measure_mECE_calibration(pred_probs, correct, num_bins=100, threshold=-1):
    """
    Compute the calibration curve using the equal mass binning scheme (i.e. equal mass bins)and computes the calibration error given this binning scheme (i.e. mECE).

    Parameters
    ----------
        see calibration_error_and_curve()
    Returns
    -------
        see calibration_error_and_curve()
    """
    assert len(pred_probs.shape)==1
    logodds = utils.to_logodds(pred_probs)
    #if logodds.max()<=1 and logodds.min()>=0:
    bin_boundaries_prob = utils.to_sigmoid( hb_utils.nolearn_bin_boundaries(num_bins, binning_scheme="eqmass", x=logodds) )
    assigned = hb_utils.bin_data(pred_probs, bin_boundaries_prob)
    return calibration_error_and_curve(pred_probs, correct, assigned, num_bins, threshold)

def measure_kECE_calibration(pred_probs, correct, num_bins=100, threshold=-1):
    """
    Compute the calibration curve using the kmeans binning scheme (i.e. use kmeans to cluster the data and then determine the bin assignments) and computes the calibration error given this binning scheme (i.e. kECE).

    Parameters
    ----------
        see calibration_error_and_curve()
    Returns
    -------
        see calibration_error_and_curve()
    """

    assert len(pred_probs.shape)==1
    centroids,_ = scipy.cluster.vq.kmeans(pred_probs, num_bins)
    cluster_ids, _ = scipy.cluster.vq.vq(pred_probs, centroids)
    cluster_ids = cluster_ids.astype(np.int)
    return calibration_error_and_curve(pred_probs, correct, cluster_ids, num_bins, threshold)

 
def measure_quantized_calibration(pred_probs, correct, assigned, num_bins=100, threshold=-1):
    """
    Compute the calibration curve given the bin assignments (i.e. quantized values). 
    """
    assert len(pred_probs.shape)==1
    return calibration_error_and_curve(pred_probs, correct, assigned, num_bins, threshold)


def calibration_error_and_curve(pred_probs, correct, assigned, num_bins=100, threshold=-1):
    """
    Compute the calibration curve and calibration error. The threshold float will determine which samples to ignore because its confidence is very low.

    Parameters
    ----------
        see calibration_curve_quantized()

    Returns
    -------
    results: dict
        dictionary with calibration information
    """
    assert len(pred_probs.shape)==1
    mask = pred_probs>threshold
    pred_probs, correct, assigned = pred_probs[mask], correct[mask], assigned[mask]
    cov = mask.mean()
    prob_pred, prob_true, counts, counts_unfilt = calibration_curve_quantized(pred_probs, correct, assigned=assigned, num_bins=num_bins)
    ece = _ece(prob_pred, prob_true, counts)
    return {"ece": ece, "prob_pred":prob_pred, "prob_true":prob_true, "counts":counts, "counts_unfilt":counts_unfilt, "threshold":threshold, "cov":cov}


def calibration_curve_quantized(pred_probs, correct, assigned, num_bins=100):
    """
    Get the calibration curve given the bin assignments, samples and sample-correctness. 

    Parameters
    ----------
    pred_probs: numpy ndarray
        numpy array with predicted probabilities (i.e. confidences)
    correct: numpy ndarray
        0/1 indicating if the sample was correctly classified or not
    num_bins: int
        number of bins for quantization
    Returns
    -------
    prob_pred: for each bin the avg. confidence 
    prob_true: for each bin the avg. accuracy 
    counts: number of samples in each bin 
    counts_unfilt: same as `counts` but also including zero bins
    """
    assert len(pred_probs.shape)==1
    bin_sums_pred = np.bincount(assigned, weights=pred_probs,  minlength=num_bins)
    bin_sums_true = np.bincount(assigned, weights=correct, minlength=num_bins)
    counts = np.bincount(assigned, minlength=num_bins)
    filt = counts > 0
    prob_pred = (bin_sums_pred[filt] / counts[filt])
    prob_true = (bin_sums_true[filt] / counts[filt])
    counts_unfilt = counts
    counts = counts[filt]
    return prob_pred, prob_true, counts, counts_unfilt








































