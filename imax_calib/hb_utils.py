#!/usr/local/bin/python3
# Copyright (c) 2021 Robert Bosch GmbH Copyright holder of the paper "Multi-Class Uncertainty Calibration via Mutual Information Maximization-based Binning" accepted at ICLR 2021.
# All rights reserved.
##
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
 hb_utils.py
 imax_calib

Contains all util functions for any histogram binning (hb) operations.

 Created by Kanil Patel on 07/27/20.
 Copyright 2020. Kanil Patel. All rights reserved.
'''
import numpy as np
import scipy; import scipy.stats; import scipy.integrate as integrate
import imax_calib.utils as utils
import imax_calib.io as io

##################
# Binning utils
#################
def nolearn_bin_boundaries(num_bins, binning_scheme, x=None):
    """
    Get the bin boundaries (in logit space) of the <num_bins> bins. This function returns only the bin boundaries which do not include any type of learning. 
    For example: equal mass bins, equal size bins or overlap bins.
    
    Parameters
    ----------
    num_bins: int
        Number of bins
    binning_scheme: string
        The way the bins should be placed. 
            'eqmass': each bin has the same portion of samples assigned to it. Requires that `x is not None`.
            'eqsize': equal spaced bins in `probability` space. Will get equal spaced bins in range [0,1] and then convert to logodds.
            'custom_range[min_lambda,max_lambda]': equal spaced bins in `logit` space given some custom range.
    x: numpy array (1D,)
        array with the 1D data to determine the eqmass bins. 

    Returns
    -------
    bins: numpy array (num_bins-1,)
        Returns the bin boundaries. It will return num_bins-1 bin boundaries in logit space. Open ended range on both sides.
    """
    if binning_scheme=="eqmass": 
        assert x is not None and len(x.shape)==1
        bins = np.linspace(1.0/num_bins, 1 - 1.0 / num_bins, num_bins-1) # num_bins-1 boundaries for open ended sides
        bins = np.percentile(x, bins * 100, interpolation='lower') # data will ensure its in Logit space
    elif binning_scheme=="eqsize":  # equal spacing in logit space is not the same in prob space because of sigmoid non-linear transformation
        bins = utils.to_logodds(   np.linspace(1.0/num_bins, 1 - 1.0 / num_bins, num_bins-1)   ) # num_bins-1 boundaries for open ended sides
    elif "custom_range" in binning_scheme: # used for example when you want bins at overlap regions. then custom range should be [ min p(y=1), max p(y=0)  ]. e.g. custom_range[-5,8]
        custom_range = eval(binning_scheme.replace("custom_range", ""))
        assert type(custom_range)==list and (custom_range[0] <= custom_range[1])
        bins = np.linspace(custom_range[0], custom_range[1], num_bins-1) # num_bins-1 boundaries for open ended sides
    return bins

def bin_data(x, bins):
    """
    Given bin boundaries quantize the data (x). When ndims(x)>1 it will flatten the data, quantize and then reshape back to orig shape. 
    Returns the following quantized values for num_bins=10 and bins = [2.5, 5.0, 7.5, 1.0]\n
    quantize: \n
              (-inf, 2.5) -> 0\n
              [2.5, 5.0) -> 1\n
              [5.0, 7.5) -> 2\n
              [7.5, 1.0) -> 3\n
              [1.0, inf) -> 4\n

    Parameters
    ----------
    x: numpy ndarray 
       Network logits as numpy array 
    bins: numpy ndarray
        location of the (num_bins-1) bin boundaries

    Returns
    -------
    assigned: int numpy ndarray 
        For each sample, this contains the bin id (0-indexed) to which the sample belongs.
    """
    orig_shape = x.shape
    # if not 1D data. so need to reshape data, then quantize, then reshape back
    if len(orig_shape)>1 or orig_shape[-1]!=1:  x = x.flatten()
    assigned = np.digitize(x, bins) # bin each input in data. np.digitize will always return a valid index between 0 and num_bins-1 whenever bins has length (num_bins-1) to cater for the open range on both sides
    if len(orig_shape)>1 or orig_shape[-1]!=1:  assigned = np.reshape(assigned, orig_shape)
    return assigned.astype(np.int)



######### Quantize data
def quantize_logodds(x, bins, bin_reprs, return_probs=True):
    """ 
    Quantize logodds (x) using bin boundaries (bins) and reprs in logit space and then convert to prob space if `return_probs=True`.

    Parameters
    ----------
    x: numpy ndarray
       Network logits as numpy array 
    bins: numpy ndarray
        Location of the (num_bins-1) bin boundaries
    bin_reprs: numpy ndarray
        Bin representations in logodds space. Contains (num_bins-1)=len(bins)+1 entries.
    return_probs: boolean (default: True)
        All operations take place in logodds space. Setting this to true will ensure that the values returned are in probability space (i.e. it will convert the quantized values from logodds to sigmoid before returning them)

    Returns
    -------
    quant_output: numpy ndarray
        The output of the quantization based on the bins and bin_reprs. Either the output will be in logodds space (i.e. return_probs=False) or in probability space.
    assigned: int numpy ndarray
        The bin assignment integers for each sample.
    """
    assigned = bin_data(x, bins) # log space
    quant_output = bin_reprs[assigned] # fill up representations based on assignments
    if return_probs:    quant_output = utils.to_sigmoid(quant_output) # prob space 
    return pred_probs, assigned


########### Bin boundary update
def bin_boundary_update_closed_form(representations):
    """
    Closed form update of boundaries. stationary point when log(p(y=1|lambda)) - log(p(y=0|lambda)) = log(log(xxx)/log(xxx)) term. LHS side is logodds/boundaries when p(y|lambda) modelled with sigmoid (e.g. PPB )
    """
    temp_log = 1. + np.exp(-1*np.abs(representations))
    temp_log[temp_log==0] = utils.EPS
    logphi_a = np.maximum(0., representations)  + np.log(temp_log)
    logphi_b = np.maximum(0., -1*representations) + np.log(temp_log)
    assert np.any(np.sign(logphi_a[1:]-logphi_a[:-1])*np.sign(logphi_b[:-1]-logphi_b[1:])>=0.)
    temp_log1 = np.abs( logphi_a[1:] - logphi_a[:-1] )
    temp_log2 = np.abs( logphi_b[:-1] - logphi_b[1:] )
    temp_log1[temp_log1==0] = utils.EPS
    temp_log2[temp_log2==0] = utils.EPS
    bin_boundaries = np.log(temp_log1) - np.log(temp_log2)               
    bin_boundaries = np.sort(bin_boundaries)
    return bin_boundaries




######### Bin representation code
def bin_representation_calculation(x, y, num_bins, bin_repr_scheme="sample_based", bin_boundaries=None, assigned=None, return_probs=False):
    """
    Bin representations: frequency based: num_positive_samples/num_total_samples in each bin.
        or pred_prob based: average of the sigmoid of lambda
    Function gets the bin representation which can be used during the MI maximization.

    Parameters
    ----------
    x: numpy ndarray
        logodds data which needs to be binned using bin_boundaries. Only needed if assigned not given.
    y: numpy ndarray
        Binary label for each sample
    bin_repr_scheme: strig
        scheme to use to determine bin reprs. options: 'sample_based' and 'pred_prob_based'
    bin_boundaries: numpy array
        logodds bin boundaries. Only needed when assigned is not given.
    assigned: int numpy array
        bin id assignments for each sample

    Returns
    -------
    quant_reprs: numpy array
        quantized bin reprs for each sample

    """
    assert (bin_boundaries is None) != (assigned is None), "Cant have or not have both arguments. Need exactly one of them."
    if assigned is None:    assigned = bin_data(x, bin_boundaries)

    if bin_repr_scheme=="sample_based":
        quant_reprs = bin_repr_unknown_LLR(y, assigned, num_bins, return_probs) # frequency estimate of correct/incorrect
    elif bin_repr_scheme=="pred_prob_based":
        quant_reprs = bin_repr_unknown_LLR(utils.to_sigmoid(x), assigned, num_bins, return_probs) # softmax probability for bin reprs
    else:
        raise Exception("bin_repr_scheme=%s is not valid."%(bin_repr_scheme))
    return quant_reprs   

def bin_repr_unknown_LLR(sample_weights, assigned, num_bins, return_probs=False):
    """
    Unknown Bin reprs. Will take the average of the either the pred_probs or the binary labels.
    Determines the bin reprs by taking average of sample weights in each bin.
    For example for sample-based repr: sample_weights should be 0 or 1 indicating correctly classified or not.
    or for pred-probs-based repr: sample_weights should be the softmax output probabilities.
    Handles reshaping if sample_weights or assigned has more than 1 dim.
    
    Parameters
    ----------
    sample_weights: numpy ndarray
        array with the weight of each sample. These weights are used to calculate the bin representation by taking the averages of samples grouped together.
    assigned: int numpy array
        array with the bin ids of each sample
    return_probs: boolean (default: True)
        All operations take place in logodds space. Setting this to true will ensure that the values returned are in probability space (i.e. it will convert the quantized values from logodds to sigmoid before returning them)

    Returns
    -------
    representations: numpy ndarray
        representations of each sample based on the bin it was assigned to 
    """
    orig_shape = sample_weights.shape
    assert np.all(orig_shape==assigned.shape)
    assert sample_weights.max()<=1.0 and sample_weights.min()>=0.0, "make sure sample weights are probabilities"
    if len(orig_shape)>1:
        sample_weights = sample_weights.flatten()
        assigned = assigned.flatten()

    bin_sums_pos = np.bincount(assigned, weights=sample_weights, minlength=num_bins) # sum up all positive samples
    counts = np.bincount(assigned, minlength=num_bins) # sum up all samples in bin
    filt = counts>0
    prob_pos = np.ones(num_bins)*sample_weights.mean() # NOTE: important change: when no samples at all fall into any bin then default should be the prior
    prob_pos[filt] = bin_sums_pos[filt] / counts[filt] # get safe prob of pos samples over all samples
    representations = prob_pos 
    if return_probs==False:    representations = utils.to_logodds( representations)#NOTE: converting to logit domain again
    return representations 

def bin_repr_known_LLR(bin_boundaries, prior_y_pos, distr_kde_dict):
    """
    Known Bin reprs (i.e. density based representation). Will get the bin representations based on the density estimated by KDE. 
    Much slower than unknown LLR. so only used when calculating the MI.

    Parameters
    ----------
    logodds: numpy ndarray
       data which will be used to estimate the KDE 
    y: numpy ndarray
        labels of the samples also used to get the positive and negative KDEs
    assigned: int numpy array
        array with the bin ids of each sample
    return_probs: boolean (default: True)
        All operations take place in logodds space. Setting this to true will ensure that the values returned are in probability space (i.e. it will convert the quantized values from logodds to sigmoid before returning them)

    Returns
    -------
    representations: numpy ndarray
        representations of each sample based on the bin it was assigned to 
    """
    distr_pos = distr_kde_dict["pos"] # scipy.stats.gaussian_kde(logodds[y==1])
    distr_neg = distr_kde_dict["neg"] # scipy.stats.gaussian_kde(logodds[y==0])
    prior_y_neg = 1 - prior_y_pos
    new_boundaries = np.hstack([-100, bin_boundaries , 100])
    new_reprs = np.zeros(len(bin_boundaries)+1)

    p_ypos_given_lam = np.zeros( len(bin_boundaries)+1 )
    p_yneg_given_lam = np.zeros( len(bin_boundaries)+1 )
    for idx in range( len(bin_boundaries) + 1):
        numer = prior_y_pos*distr_pos.integrate_box_1d(new_boundaries[idx], new_boundaries[idx+1]) # p(lam|y=1)*p(y=1)
        denom = prior_y_neg*distr_neg.integrate_box_1d(new_boundaries[idx], new_boundaries[idx+1]) # p(lam|y=0)*p(y=0)
        new_reprs[idx] = utils.safe_log_diff(numer, denom, np.log)
        p_ypos_given_lam[idx] = numer
        p_yneg_given_lam[idx] = denom
    new_reprs[~np.isfinite(new_reprs)] = utils.EPS
    new_reprs[new_reprs==0] = utils.EPS
    return new_reprs, p_ypos_given_lam, p_yneg_given_lam






def MI_unknown_LLR(p_y_pos, logodds, bin_boundaries, representations):
    """logodds => the logodds which were used to bin. rewrote MI loss: sum_Y sum_B p(y'|lambda)p(lambda) for term outside log. Before it was p(lambda|y')p(y') """
    # NOTE: checked and matches impl of Dan: -1*MI_eval(**kwargs) => all good
    pred_probs = utils.to_sigmoid(logodds)
    prior_y = io.AttrDict( dict(pos=p_y_pos, neg=1-p_y_pos)  )
    num_bins = len(bin_boundaries)+1
    # get p(y|lambda)p(lambda).... first get mean pred. prob. per bin
    assigned = bin_data(logodds, bin_boundaries)
    bin_sums_pred_probs_pos = np.bincount( assigned, weights=pred_probs, minlength=num_bins) # get the reprs in prob space because of mean.
    p_y_pos_given_lambda_per_bin = bin_sums_pred_probs_pos / logodds.shape[0]
    bin_sums_pred_probs_neg = np.bincount( assigned, weights=1-pred_probs, minlength=num_bins) # get the reprs in prob space because of mean.
    p_y_neg_given_lambda_per_bin = bin_sums_pred_probs_neg / logodds.shape[0]
    p_y_given_lambda_dict = io.AttrDict(dict(pos=p_y_pos_given_lambda_per_bin, neg=p_y_neg_given_lambda_per_bin))
    mi_loss = 0.0
    for binary_class_str, binary_class in zip( ["neg","pos"], [0,1] ):
        terms_in_log = (   1 + np.exp((1-2*binary_class) * representations)  )       *    prior_y[binary_class_str]   # part 3
        bin_summation_term =  np.sum(  p_y_given_lambda_dict[binary_class_str] * np.log(  terms_in_log ) )
        mi_loss += bin_summation_term
    return -1*mi_loss




def MI_known_LLR(bin_boundaries, p_y_pos, distr_kde_dict):
    """
    Calculate the MI(lambda_hat, y)(using the known LLR), where lambda_hat is the quantized lambdas.
    This will compute the MI in bits (log2).
    It uses a KDE to estimate the density of the positive and negative samples.
    At the end it will perform some basic checks to see if the computations were correct.
    In addition to the MI it will compute the bit rate (R) (i.e. MI(z, lambda) where z is quantized lambda)
    

    Parameters
    ----------
    bin_boundaries: numpy array
        bin boundaries
    p_y_pos: float
        p(y=1) prior
    distr_kde_dict: dict
        dictionary containing the KDE objects used to estimate the density in each bin with keys 'pos' and 'neg'.

    Returns
    -------
    MI: float
        MI(z, y) where z is quantized lambda. This is the mutual information between the quantizer output to the label.
    R: float
        bin rate. This is MI(z, lambda). Mutual Information between lambda and quantized lambda.
    """
    distr_pos, distr_neg = distr_kde_dict["pos"], distr_kde_dict["neg"]        
    p_y_neg = 1 - p_y_pos


    new_boundaries = np.hstack([-100, bin_boundaries, 100])
    # lists for checks afterwards
    all_vs, all_intpos, all_intneg = [], [], []
    MI, R = 0.0, 0.0
    for idx in range( len(bin_boundaries) + 1):
        integral_pos = p_y_pos*distr_pos.integrate_box_1d(new_boundaries[idx], new_boundaries[idx+1]) # p(lam|y=1)*p(y=1) = p(lam|y=1)
        integral_neg = p_y_neg*distr_neg.integrate_box_1d(new_boundaries[idx], new_boundaries[idx+1]) # p(lam|y=1)*p(y=1) = p(lam|y=0)
        repr = utils.safe_log_diff(integral_pos, integral_neg, np.log)

        p_ypos_given_z = max( utils.EPS, utils.to_sigmoid(   repr) )
        p_yneg_given_z = max( utils.EPS, utils.to_sigmoid(-1*repr) )

        curr_MI_pos = integral_pos * ( utils.safe_log_diff( p_ypos_given_z, p_y_pos, np.log2    ) )
        curr_MI_neg = integral_neg * ( utils.safe_log_diff( p_yneg_given_z, p_y_neg, np.log2    ) )
        MI += curr_MI_pos + curr_MI_neg 

        v = max( utils.EPS, (integral_pos + integral_neg)     )
        curr_R = -1 * v * np.log2(v) # entropy of p(z) = p(z|y=1)p(y=1) + p(z|y=0)p(y=0)
        R += curr_R
        # gather for checks
        all_vs.append(v)
        all_intpos.append(integral_pos); all_intneg.append(integral_neg)
    np.testing.assert_almost_equal( np.sum(all_vs), 1.0 , decimal=1)
    np.testing.assert_almost_equal( np.sum(all_intpos), p_y_pos, decimal=1)
    np.testing.assert_almost_equal( np.sum(all_intneg), p_y_neg, decimal=1)
    return MI, R


def MI_upper_bounds(p_y_pos, distr_kde_dict):
    """
    Calculate the MI upper bound of MI(z, y) <= MI(lambda, y). As z is the quantized version of lambda, MI(z, y) is upper bounded by MI(lambda, y).
    This is a tigther bound than H(y). This function will return both upper bounds.

    Bound 1: MI(z, y) <= H(y) - H(y|z) <= H(y)
    Bound 2: MI(z, y) <= MI(lambda, y)

    Parameters
    ----------
    p_y_pos: float
        p(y=1) prior
    distr_kde_dict: dict
        dictionary containing the KDE objects used to estimate the density in each bin with keys 'pos' and 'neg'.

    Returns
    -------
    H_y: float
        Loose upper bound which is H(y)        
    MI_y_lambda: float
        Upper bound of MI(z, y) which is upper bounded by MI(lambda, y). Tigther bound than H(y)

    """
    tic = io.time.time()
    p_y_neg = 1 - p_y_pos

    # Bound 1
    H_y = -1*p_y_pos*np.log2(p_y_pos) + -1*p_y_neg*np.log2(p_y_neg)

    # Bound 2
    distr_pos, distr_neg = distr_kde_dict["pos"], distr_kde_dict["neg"]        
    def get_logodd_lambda(lam):
        log_term_1 = p_y_pos * distr_pos.pdf(lam)
        log_term_2 = p_y_neg * distr_neg.pdf(lam) 
        logodd_lambda = utils.safe_log_diff(log_term_1, log_term_2, np.log)
        return logodd_lambda

    def integral_pos(lam):
        logodd_lambda = get_logodd_lambda(lam)
        p_ypos_lambda = utils.to_sigmoid(    logodd_lambda  )
        return p_y_pos * distr_pos.pdf(lam) * utils.safe_log_diff( p_ypos_lambda, p_y_pos, np.log2)  #np.log2(  p_ypos_lambda    /   p_y_pos  )
	
    def integral_neg(lam):
        logodd_lambda = get_logodd_lambda(lam)
        p_yneg_lambda = utils.to_sigmoid(  -1*  logodd_lambda  )
        return p_y_neg * distr_neg.pdf(lam) * utils.safe_log_diff(p_yneg_lambda, p_y_neg, np.log2) #np.log2(  p_yneg_lambda    /   p_y_neg  )    
	
    term_pos = integrate.quad(integral_pos, -100, 100, limit=100)[0]
    term_neg = integrate.quad(integral_neg, -100, 100, limit=100)[0]
    MI_y_lambda =  term_pos + term_neg


    toc = io.time.time()
    print("Time elapsed: upper bound computation: ", (toc-tic), " seconds!")
    return H_y, MI_y_lambda










