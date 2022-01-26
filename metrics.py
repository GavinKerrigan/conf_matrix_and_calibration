import torch
import numpy as np
from torch import nn
from imax_calib.evaluations import calibration_metrics as cal_metrics  # Imax paper
import calibration as cal  # Kumar et al, Verified Uncertainty Calibration

# Implements various metrics.


def get_acc(y_pred, y_true):
    """ Computes the accuracy of predictions.
    If y_pred is 2D, it is assumed that it is a matrix of scores (e.g. probabilities) of shape (n_samples, n_classes)
    """
    if y_pred.ndim == 1:
        return np.mean(y_pred == y_true)
    elif y_pred.ndim == 2:
        return np.mean(np.argmax(y_pred, axis=1), y_true)


def get_cw_ECE(probs, y_true, mode='mass', threshold_mode='class', num_bins=15):
    """ Estimates the class-wise ECE by binning.

    Args:
        probs: shape (n_samples, n_classes)
        y_true: shape (n_samples, )
        mode: Either 'mass' or 'width' -- determines binning scheme
        threshold_mode: Either 'class' or None -- determines if thresholding is used in estimation
        num_bins: Number of bins used in estimation
    """

    if mode == 'mass':
        _mode = 'mECE'
    elif mode == 'width':
        _mode = 'dECE'

    evals = cal_metrics.compute_top_1_and_CW_ECEs(probs, y_true, list_approximators=[_mode],
                                                  num_bins=num_bins, threshold_mode=threshold_mode)
    return evals[f'cw_{_mode}']


def get_ECE(probs, y_true, mode='mass', num_bins=15):
    """ Estimates the top-label ECE by binning.

    Args:
        probs: shape (n_samples, n_classes)
        y_true: shape (n_samples, )
        mode: Either 'mass' or 'width' -- determines binning scheme
        num_bins: Number of bins used in estimation
    """
    if mode == 'mass':
        _mode = 'mECE'
    elif mode == 'width':
        _mode = 'dECE'

    evals = cal_metrics.compute_top_1_and_CW_ECEs(probs, y_true, list_approximators=[_mode], num_bins=num_bins)
    return evals[f'top_1_{_mode}']


def get_MCE(probs, y_true):
    """ Estimates the class-wise ECE. Not recommended for use.
    """
    return cal.get_calibration_error(probs, y_true,
                                     p=1, debias=False, mode='marginal')


def get_NLL(probs, y_true):
    """ Computes the negative log likelihood.
    """
    nll = nn.NLLLoss()
    _probs = np.clip(probs, 1e-100, 1)
    logprobs = torch.from_numpy(np.log(_probs))
    return nll(logprobs, torch.from_numpy(y_true)).item()
