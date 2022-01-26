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
 binners.py
 calibrators

 Created by Kanil Patel on 07/28/20.
 Copyright 2020. Kanil Patel. All rights reserved.
'''
import os
import contextlib
import numpy as np
import scipy; import scipy.stats; import scipy.integrate as integrate

import imax_calib.utils as utils
import imax_calib.hb_utils as hb_utils
from imax_calib.calibrators.scalers_np import BaseCalibrator
import imax_calib.clustering as clustering
import imax_calib.io as io
from tqdm import tqdm as tqdm


class HistogramBinninerCW(BaseCalibrator):
    def __init__(self, cfg, scaler_obj=None, load_params=False, **kwargs):
        """
        Histogram Binning Multi-Class binning by binning each class. 
        The binning stage determines if the RAW logodds will be binned or the scaled logodds (scaler_obj must not be None).
        """
        super(HistogramBinninerCW, self).__init__()
        self.cfg = cfg        
        self.list_binners = []
        self.parameter_list = ["bin_boundaries", "bin_representations_SB", "bin_representations_PPB"]# each individual scalers parameters. will be used to get the attr of each binary object
        self.binning_stage = cfg.Q_binning_stage 
        self.scaler_obj = scaler_obj
        assert self.binning_stage in ["raw", "scaled"]

        if self.scaler_obj is None: assert self.binning_stage=="raw", "If no scaler_obj is provided then binning stage can only be RAW logodds"
        assert self.cfg.cal_setting=="CW"

        for class_idx in range(self.cfg.n_classes):
            histbin_c = _HistogramBinniner_Binary(self.cfg, self.cfg.cal_setting, class_idx, cfg.num_bins, cfg.Q_method, cfg.Q_binning_repr_scheme, cfg.Q_bin_repr_during_optim)
            self.list_binners.append(histbin_c)
           
    def fit(self, logits, logodds, y, **kwargs):
        """
        Fits a binary histogram binner on each class idx. Input needs to be logits which will be converted to logodds before passing to binary binner.
        
        This function will also handle the scaling option. Options - Binn. Setting:
            1) Binn.: raw    logodds and Binn. reprs: raw    logodds ====> self.binning_stage='raw'    and self.scaler_obj is None 
            2) Binn.: raw    logodds and Binn. reprs: scaled logodds ====> self.binning_stage='raw'    and self.scaler_obj is not None 
            3) Binn.: scaled logodds and Binn. reprs: scaled logodds ====> self.binning_stage='scaled' and self.scaler_obj is not None
        """

        logodds_to_bin = logodds
        # handles scaling before binning or as bin reprs.
        binn_setting = 1
        if self.scaler_obj is not None:
            binn_setting = 2
            _, scaled_logodds, _ = self.scaler_obj(logits, logodds)
            if self.binning_stage=="scaled": # if binning_stage==scaled then set logodds to scaled logodds
                binn_setting = 3
                logodds_to_bin = scaled_logodds
        else:
            scaled_logodds = None


        logits = None # dont need it so set it to None to be sure its not used!
        tic = io.time.time()
        for class_idx, histbin_c in enumerate(self.list_binners):
            #with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            histbin_c.fit(logits, logodds_to_bin, y, logodds_for_bin_reprs=scaled_logodds)

        toc = io.time.time()




    def calibrate(self, logits, logodds, **kwargs):
        """
        Inputs need to be logits and NOT logodds.
        Will return scaled_logdds and scaled_probs.

        Parameters
        ----------
        logits: numpy ndarray
            raw network logits (before softmax)
        logodds: numpy ndarray
            logodds - binary setting
        kwargs: dict
            other parameters - not used here

        Returns
        -------
        cal_logits: numpy ndarray
            calibrated logits (in this case None as PlattScaling scales logodds and not logits)
        cal_logodds: numpy ndarray
            calibrated logodds
        cal_probs: numpy ndarray
            calibrated probabilities
        cal_assigned: numpy ndarray int
            the bin ids for each sample/class. This is required to calc. the ECE for binning methods so can be returned here as binning already performed in calibrator.
        """
        logits = None # dont need it so set it to None to be sure its not used!
        tic = io.time.time()
        assert len(logodds.shape)>1, "Need to send all logodds. Splitting into individual classes will be done in binary scalers."
        cal_logodds = np.zeros_like(logodds)
        cal_probs = np.zeros_like(logodds)
        cal_assigned = np.zeros_like(logodds, dtype=np.int)
        for class_idx, histbin_c in enumerate(self.list_binners):
            _, new_logodds, new_probs, new_assigned = histbin_c(logits, logodds)
            cal_logodds[..., class_idx] = new_logodds 
            cal_probs[..., class_idx] = new_probs
            cal_assigned[..., class_idx] = new_assigned
        toc = io.time.time()
        cal_logits = None
        return cal_logits, cal_logodds, cal_probs, cal_assigned 
    
    def save_params(self, fpath):
        """
        For this calibrator (HB-based): overwrite saving/loading functions. Save all binary HB parameters in a single hdf5 file.
        """
        if len(self.parameter_list)>0:
            data_to_save = io.AttrDict()
            for class_idx, histbin_c in enumerate(self.list_binners):
                data_to_save["class_%d"%(class_idx)] = io.AttrDict({})
                for key in self.parameter_list: data_to_save["class_%d"%(class_idx)][key] = getattr(histbin_c, key)
            io.deepdish_write(fpath, data_to_save)

    def load_params(self, fpath):
        """
        For this kind of binary scaler: overwrite saving/loading functions. Load all binary class parameters in a single hdf5 file.
        """
        data_to_save = io.deepdish_read(fpath)
        if len(self.parameter_list)>0:
            for class_idx, histbin_c in enumerate(self.list_binners):
                for key in self.parameter_list: 
                    curr_param = data_to_save["class_%d"%(class_idx)][key] 
                    setattr(histbin_c, key, curr_param)


    def get_calib_parameters(self):
        """ 
        Get all parameters as a dictionary of lists
        """
        all_params = io.AttrDict()
        for param_name in self.parameter_list:
            all_params[param_name] = np.array([getattr(histbin_c, param_name) for histbin_c in self.list_binners])

        if self.histbin_c.binning_repr_scheme=="pred_prob_based": 
            all_params["bin_representations"] = all_params["bin_representations_PPB"] 
        elif self.histbin_c.binning_repr_scheme=="sample_based":  
            all_params["bin_representations"] = all_params["bin_representations_SB"] 
        return all_params


class HistogramBinninerTop1(BaseCalibrator):
    def __init__(self, cfg, scaler_obj=None, load_params=False, **kwargs):
        """
        Histogram Binning Multi-Class binning by binning each class. 
        The binning stage determines if the RAW logodds will be binned or the scaled logodds (scaler_obj must not be None).
        """
        super(HistogramBinninerTop1, self).__init__()
        self.cfg = cfg        
        self.parameter_list = ["bin_boundaries", "bin_representations_SB", "bin_representations_PPB"]# each individual scalers parameters. will be used to get the attr of each binary object
        self.binning_stage = cfg.Q_binning_stage 
        self.scaler_obj = scaler_obj
        assert self.binning_stage in ["raw", "scaled"]

        if self.scaler_obj is None: assert self.binning_stage=="raw", "If no scaler_obj is provided then binning stage can only be RAW logodds"
        assert self.cfg.cal_setting=="top1"

        self.histbin_c = _HistogramBinniner_Binary(self.cfg, self.cfg.cal_setting, None, cfg.num_bins, cfg.Q_method, cfg.Q_binning_repr_scheme, cfg.Q_bin_repr_during_optim)
           
    def fit(self, logits, logodds, y, **kwargs):
        """
        Fits a binary histogram binner on each class idx. Input needs to be logits which will be converted to logodds before passing to binary binner.
        
        This function will also handle the scaling option. Options - Binn. Setting:
            1) Binn.: raw    logodds and Binn. reprs: raw    logodds ====> self.binning_stage='raw'    and self.scaler_obj is None 
            2) Binn.: raw    logodds and Binn. reprs: scaled logodds ====> self.binning_stage='raw'    and self.scaler_obj is not None 
            3) Binn.: scaled logodds and Binn. reprs: scaled logodds ====> self.binning_stage='scaled' and self.scaler_obj is not None
        """

        logodds_to_bin = logodds
        # handles scaling before binning or as bin reprs.
        binn_setting = 1
        if self.scaler_obj is not None:
            binn_setting = 2
            _, scaled_logodds, _ = self.scaler_obj(logits, logodds)
            if self.binning_stage=="scaled": # if binning_stage==scaled then set logodds to scaled logodds
                binn_setting = 3
                logodds_to_bin = scaled_logodds
        else:
            scaled_logodds = None

        logits = None # dont need it so set it to None to be sure its not used!

        tic = io.time.time()
        #with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        self.histbin_c.fit(logits, logodds_to_bin, y, logodds_for_bin_reprs=scaled_logodds)

        # now save parameters
        toc = io.time.time()


    def calibrate(self, logits, logodds, **kwargs):
        """
        Inputs need to be logits and NOT logodds.
        Will return scaled_logdds and scaled_probs.

        Parameters
        ----------
        logits: numpy ndarray
            raw network logits (before softmax)
        logodds: numpy ndarray
            logodds - binary setting
        kwargs: dict
            other parameters - not used here

        Returns
        -------
        cal_logits: numpy array (N,)
            calibrated logits (in this case None as PlattScaling scales logodds and not logits)
        cal_logodds: numpy array (N,)
            calibrated logodds
        cal_probs: numpy array (N,)
            calibrated probabilities
        cal_assigned: numpy array (N,)
            bin ids for each samples top prediction. Needed for ECE calc. and can be returned from here as the binning is already performed in the calibrator
        """
        assert len(logodds.shape)>1, "Need to send all logodds. Splitting into individual classes will be done in binary scalers."

        tic = io.time.time()
        _, cal_logodds, cal_probs, cal_assigned = self.histbin_c(logits, logodds)
        toc = io.time.time()
        cal_logits = None
        return cal_logits, cal_logodds, cal_probs, cal_assigned 
    
    def save_params(self, fpath):
        """
        For this calibrator (HB-based): overwrite saving/loading functions. Save all binary HB parameters in a single hdf5 file.
        """
        if len(self.parameter_list)>0:
            data_to_save = io.AttrDict()
            for key in self.parameter_list: data_to_save[key] = getattr(self.histbin_c, key)
            io.deepdish_write(fpath, data_to_save)

    def load_params(self, fpath):
        """
        For this kind of binary scaler: overwrite saving/loading functions. Load all binary class parameters in a single hdf5 file.
        """
        data_to_save = io.deepdish_read(fpath)
        if len(self.parameter_list)>0:
            for key in self.parameter_list: 
                curr_param = data_to_save[key] 
                setattr(self.histbin_c, key, curr_param)

    def get_calib_parameters(self):
        """ 
        Get all parameters as a dictionary of lists
        """
        all_params = io.AttrDict()
        for param_name in self.parameter_list:
            all_params[param_name] = np.array([getattr(self.histbin_c, param_name),])

        if self.histbin_c.binning_repr_scheme=="pred_prob_based": 
            all_params["bin_representations"] = all_params["bin_representations_PPB"] 
        elif self.histbin_c.binning_repr_scheme=="sample_based":  
            all_params["bin_representations"] = all_params["bin_representations_SB"] 
        return all_params



class HistogramBinninerSharedCW(BaseCalibrator):
    def __init__(self, cfg, scaler_obj=None, load_params=False, **kwargs):
        """
        Histogram Binning Multi-Class binning by binning each class. 
        The binning stage determines if the RAW logodds will be binned or the scaled logodds (scaler_obj must not be None).
        """
        super(HistogramBinninerSharedCW, self).__init__()
        self.cfg = cfg        
        self.parameter_list = ["bin_boundaries", "bin_representations_SB", "bin_representations_PPB"]# each individual scalers parameters. will be used to get the attr of each binary object
        self.binning_stage = cfg.Q_binning_stage 
        self.scaler_obj = scaler_obj
        assert self.binning_stage in ["raw", "scaled"]

        if self.scaler_obj is None: assert self.binning_stage=="raw", "If no scaler_obj is provided then binning stage can only be RAW logodds"
        assert self.cfg.cal_setting=="sCW"

        self.histbin_c = _HistogramBinniner_Binary(self.cfg, cfg.cal_setting, None, cfg.num_bins, cfg.Q_method, cfg.Q_binning_repr_scheme, cfg.Q_bin_repr_during_optim)
           
    def fit(self, logits, logodds, y, **kwargs):
        """
        Fits a binary histogram binner on each class idx. Input needs to be logits which will be converted to logodds before passing to binary binner.
        
        This function will also handle the scaling option. Options - Binn. Setting:
            1) Binn.: raw    logodds and Binn. reprs: raw    logodds ====> self.binning_stage='raw'    and self.scaler_obj is None 
            2) Binn.: raw    logodds and Binn. reprs: scaled logodds ====> self.binning_stage='raw'    and self.scaler_obj is not None 
            3) Binn.: scaled logodds and Binn. reprs: scaled logodds ====> self.binning_stage='scaled' and self.scaler_obj is not None
        """

        logodds_to_bin = logodds
        # handles scaling before binning or as bin reprs.
        binn_setting = 1
        if self.scaler_obj is not None:
            binn_setting = 2
            _, scaled_logodds, _ = self.scaler_obj(logits, logodds)
            if self.binning_stage=="scaled": # if binning_stage==scaled then set logodds to scaled logodds
                binn_setting = 3
                logodds_to_bin = scaled_logodds
        else:
            scaled_logodds = None

        logits = None # dont need it so set it to None to be sure its not used!

        tic = io.time.time()
        #with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        self.histbin_c.fit(logits, logodds_to_bin, y, logodds_for_bin_reprs=scaled_logodds)

        # now save parameters
        toc = io.time.time()


    def calibrate(self, logits, logodds, **kwargs):
        """
        Inputs need to be logits and NOT logodds.
        Will return scaled_logdds and scaled_probs.

        Parameters
        ----------
        logits: numpy ndarray
            raw network logits (before softmax)
        logodds: numpy ndarray
            logodds - binary setting
        kwargs: dict
            other parameters - not used here

        Returns
        -------
        cal_logits: numpy array (N,)
            calibrated logits (in this case None as PlattScaling scales logodds and not logits)
        cal_logodds: numpy array (N,)
            calibrated logodds
        cal_probs: numpy array (N,)
            calibrated probabilities
        cal_assigned: numpy array (N,)
            bin ids for each samples top prediction. Needed for ECE calc. and can be returned from here as the binning is already performed in the calibrator
        """
        assert len(logodds.shape)>1, "Need to send all logodds. Splitting into individual classes will be done in binary scalers."
        ### convert to top1
        logits = None # dont need it so set it to None to be sure its not used!

        tic = io.time.time()
        cal_logodds = np.zeros_like(logodds)
        cal_probs = np.zeros_like(logodds)
        cal_assigned = np.zeros_like(logodds, dtype=np.int)
        for class_idx in range(self.cfg.n_classes):

            # create a temp calib obj which will calibrate each class using the same binning parameters!
            temp_histbin_c = _HistogramBinniner_Binary(self.cfg, cal_setting="CW", class_idx=class_idx, 
                                num_bins=self.histbin_c.num_bins, binning_scheme=self.histbin_c.binning_scheme, 
                                binning_repr_scheme=self.histbin_c.binning_repr_scheme, bin_repr_during_optim=self.histbin_c.bin_repr_during_optim)
            for k in temp_histbin_c.parameter_list: setattr(temp_histbin_c, k, getattr(self.histbin_c, k)) # update each parameters of temp binner with sCW learned version
            _, new_logodds, new_probs, new_assigned = temp_histbin_c(logits, logodds)
            cal_logodds[..., class_idx] = new_logodds 
            cal_probs[..., class_idx] = new_probs
            cal_assigned[..., class_idx] = new_assigned
        toc = io.time.time()
        cal_logits = None
        return cal_logits, cal_logodds, cal_probs, cal_assigned 
    
    def save_params(self, fpath):
        """
        For this calibrator (HB-based): overwrite saving/loading functions. Save all binary HB parameters in a single hdf5 file.
        """
        if len(self.parameter_list)>0:
            data_to_save = io.AttrDict()
            for key in self.parameter_list: data_to_save[key] = getattr(self.histbin_c, key)
            io.deepdish_write(fpath, data_to_save)

    def load_params(self, fpath):
        """
        For this kind of binary scaler: overwrite saving/loading functions. Load all binary class parameters in a single hdf5 file.
        """
        data_to_save = io.deepdish_read(fpath)
        if len(self.parameter_list)>0:
            for key in self.parameter_list: 
                curr_param = data_to_save[key] 
                setattr(self.histbin_c, key, curr_param)

    def get_calib_parameters(self):
        """ 
        Get all parameters as a dictionary of lists
        """
        all_params = io.AttrDict()
        for param_name in self.parameter_list:
            all_params[param_name] = np.array([getattr(self.histbin_c, param_name),])

        if self.histbin_c.binning_repr_scheme=="pred_prob_based": 
            all_params["bin_representations"] = all_params["bin_representations_PPB"] 
        elif self.histbin_c.binning_repr_scheme=="sample_based":  
            all_params["bin_representations"] = all_params["bin_representations_SB"] 
        return all_params












class _HistogramBinniner_Binary(BaseCalibrator):
    def __init__(self, cfg, cal_setting, class_idx, num_bins=15, binning_scheme="imax", binning_repr_scheme="pred_prob_based", bin_repr_during_optim="pred_prob_based"):
        """
        Histogram Binning Calibrator. Can be using eqsize, eqmass or Imax binning scheme and different representations.       
        This object bins one specific class

        Parameters
        ----------
        cfg: io.AttriDict
            all configs from main script.
        class_idx: int or None
            determines which class this obj will bin
        cal_setting: string
            will determine how the data (multi-class) is converted to binary
        binning_scheme: String (default: 'imax')
            scheme to use to determine the binning scheme
        binning_repr_scheme: String (default: pred_prob_based)
            the representation to use when quantizing the logodds during inference
        bin_repr_during_optim: String
            scheme to use to determine the bin representations DURING optimization
        """
        self.cfg = cfg
        self.cal_setting = cal_setting
        self.class_idx = class_idx
        self.num_bins = num_bins
        self.binning_scheme = binning_scheme
        self.binning_repr_scheme = binning_repr_scheme
        self.bin_repr_during_optim = bin_repr_during_optim

        # checks
        assert self.binning_scheme in ["imax", "eqmass", "eqsize"] or "custom_range" in self.binning_scheme
        assert self.bin_repr_during_optim in ["pred_prob_based", "sample_based"]
        assert self.binning_repr_scheme in ["pred_prob_based", "sample_based"]
        
        self.bin_boundaries = np.zeros(num_bins-1)
        self.bin_representations_SB = np.zeros(num_bins)
        self.bin_representations_PPB = np.zeros(num_bins)
        self.parameter_list = ["bin_boundaries", "bin_representations_SB", "bin_representations_PPB"]
    
    def fit(self, logits, logodds, y, logodds_for_bin_reprs=None, **kwargs):
        """
        Fit Histogram Binning calibrator. This function will first learn the bin boundaries (IMAX will learn it the others will be determined).
        After learning the bin boundaries, two types of bin representations will be computed.

        logodds_for_bin_reprs can be used to send scaled_logodds (raw logodds scaled using some scaling method). This will then bin the logodds but use the for bin reprs.
        By default will use the logodds to determine the bin reprs.

        Parameters
        ----------

        Returns
        -------
        """
        logits = None # dont need it so set it to None to be sure its not used!
        if logodds_for_bin_reprs is None: 
            logodds_for_bin_reprs = logodds

        # get the class specific logodds
        logodds, y = utils.binary_convertor(logodds, y, self.cal_setting, self.class_idx)
        logodds_for_bin_reprs, _ = utils.binary_convertor( logodds_for_bin_reprs, None, self.cal_setting, self.class_idx)
        
        if self.binning_scheme=="imax":
            log = run_imax(logodds, y, self.num_bins, num_steps=200, init_mode=self.cfg.Q_init_mode, bin_repr_during_optim=self.bin_repr_during_optim, log_every_steps=100)
            self.bin_boundaries = log.bin_boundaries[-1]
            self.MI = log.MI
            self.Rbitrate = log.Rbitrate
        elif self.binning_scheme=="eqmass" or self.binning_scheme=="eqsize" or "custom_range" in self.binning_scheme:
            self.bin_boundaries = hb_utils.nolearn_bin_boundaries(self.num_bins, binning_scheme=self.binning_scheme, x=logodds)

            # calc. MI at the end now
            distr_kde_dict = fit_kde_distributions(logodds, y)
            self.MI, self.Rbitrate = hb_utils.MI_known_LLR(self.bin_boundaries, p_y_pos=y.mean(), distr_kde_dict=distr_kde_dict)
        else:
            raise Exception("Binning scheme %s unknown!"%(self.binning_scheme))
        
        assigned = hb_utils.bin_data(logodds, self.bin_boundaries) # bin the raw logodds and then use scaled logodds to get the predictions
        self.bin_representations_SB = hb_utils.bin_representation_calculation(None, y, self.num_bins, bin_repr_scheme="sample_based"   , assigned=assigned)
        self.bin_representations_PPB= hb_utils.bin_representation_calculation(logodds_for_bin_reprs, y, self.num_bins, bin_repr_scheme="pred_prob_based", assigned=assigned)

    def calibrate(self, logits, logodds, **kwargs):
        """
        Calibrate using HB. Will also return the bin assignements which are used to calculate the ECE. 
        Will always take in multi-class data but only returns the calibrated preds for a binary case which depends on cal_setting and class_idx.

        Parameters
        ----------
        logits: numpy ndarray
           Logits which need to be binned. They will be converted to logodds here. 
        Returns
        -------
        cal_logits: numpy ndarray
            calibrated logits (in this case None as bin logodds and not logits)
        cal_logodds: numpy ndarray
            calibrated logodds
        cal_probs: numpy ndarray
            calibrated probabilities
        assigned: numpy array
            bin id assignements. needed to calculate the ECE
        """
        logits = None # dont need it so set it to None to be sure its not used!
        
        assert self.cal_setting != "sCW"
        logodds, _ = utils.binary_convertor(logodds, None, self.cal_setting, self.class_idx)

        assigned = hb_utils.bin_data(logodds, self.bin_boundaries)
        if self.binning_repr_scheme=="pred_prob_based": bin_reprs = self.bin_representations_PPB 
        elif self.binning_repr_scheme=="sample_based":  bin_reprs = self.bin_representations_SB 
        cal_logodds = bin_reprs[assigned] # fill up representations based on assignments
        cal_probs = utils.to_sigmoid(cal_logodds) # prob space 
        cal_logits = None
        return cal_logits, cal_logodds, cal_probs, assigned 

    def __call__(self, *args, **kwargs):
        return self.calibrate(*args, **kwargs)

    def save_params(self, fpath):
        raise Exception("Save all binary parameters as one instead of single files")

    def load_params(self, fpath):
        raise Exception("Load all binary parameters from one file instead of single files")









def run_imax(logodds, y, num_bins=15, p_y_pos=None, num_steps=200, init_mode="kmeans", bin_repr_during_optim="pred_prob_based", log_every_steps=10, logfpath=None, skip_slow_evals=True ):
    #print("Starting I-Max MI Optimization!")
    if p_y_pos is None: p_y_pos = y.mean()
    p_y_neg = 1 - p_y_pos

    # fit KDE for MI estimation
    distr_kde_dict = fit_kde_distributions(logodds, y)

    bin_repr_func = bin_representation_function(logodds, y, num_bins, bin_repr_scheme=bin_repr_during_optim) # get the sample_based or pred_prob_based representations used during training
    bin_boundary_func = bin_boundary_function()
    loss_func= mutual_information_and_R_function(p_y_pos, distr_kde_dict)
    loss_func_unknown = unknown_LLR_mutual_information(p_y_pos, logodds)

    if skip_slow_evals==False:
        # get upper bounds
        H_y, MI_lambda_y = hb_utils.MI_upper_bounds(p_y_pos, distr_kde_dict)
        #print("Upper bounds: H_y: %.7f and MI(lambda, y): %.7f"%(H_y, MI_lambda_y))
    else:
        H_y, MI_lambda_y = -1, -1
    
    if init_mode=="kmeans":
        representations, _ = clustering.kmeans_pp_init(logodds[..., np.newaxis], num_bins, 755619, mode='jsd')
        representations = np.sort(np.squeeze(representations))
    elif init_mode=="eqmass" or init_mode=="eqsize" or "custom_range" in init_mode:
        boundaries = hb_utils.nolearn_bin_boundaries(num_bins, binning_scheme=init_mode, x=logodds)
        representations = bin_repr_func(boundaries) 
    else:
        raise Exception("I-Max init unknown!")


    LOG = io.Logger(fpath=logfpath)
    tic = io.time.time()
    #pbar = tqdm(range(num_steps))
    time_loss_cal=0.0
    MI, Rbitrate, MIunknownLLR = np.inf, np.inf, np.inf 
    for i, step in enumerate(range(num_steps)):
        # Boundary update
        boundaries = bin_boundary_func(representations)

        # Theta - bin repr update
        representations = bin_repr_func(boundaries) 

        # logging
        if log_every_steps is not None and (step%log_every_steps == 0 or step==0):
            # Loss calc.
            tic2=io.time.time()
            MI, Rbitrate = loss_func(boundaries)
            MIunknownLLR = loss_func_unknown(boundaries, representations)
            toc2=io.time.time()
            LOG.log("step", step); LOG.log("bin_boundaries", boundaries); LOG.log("bin_representations", representations)
            LOG.log("MIunknownLLR", MIunknownLLR)
            LOG.log("MI", MI)
            LOG.log("Rbitrate", Rbitrate)
            time_loss_cal=toc2-tic2

        print_str = "%d/%d, (MI, R) : (%.7f, %.3f) - MIunknownLLR: %.7f - time for loss calc: %.2f seconds"%(step, num_steps, MI, Rbitrate, MIunknownLLR, (time_loss_cal))
        #pbar.set_description(("%s"%(print_str)))
    ######################## end all learning steps!!!!            
    #del pbar
    MI, Rbitrate = loss_func(boundaries)
    MIunknownLLR = loss_func_unknown(boundaries, representations)
    #print("(MI, R) = (%.7f, %.3f) - MIunknownLLR: %.7f"%(MI, Rbitrate, MIunknownLLR))
    LOG.log("step", step); 
    LOG.log("bin_boundaries", boundaries); 
    LOG.log("bin_representations", representations)
    LOG.log("MIunknownLLR", MIunknownLLR)
    LOG.log("MI", MI)
    LOG.log("Rbitrate", Rbitrate)

    #print(print_str)
    #print ("\n")
    
    LOG.log("H_y", H_y)
    LOG.log("MI_lambda_y", MI_lambda_y)
    LOG.end_log()
    if LOG.fpath is not None:   LOG.save_log()
    return LOG.logdata 





















def bin_representation_function(logodds, labels, num_bins, bin_repr_scheme="sample_based"):
    """
    Get function which returns the sample based bin representations. The function will take in bin boundaries as well as the logodds and labels to return the representations.

    Parameters
    ----------
    logodds: numpy ndarray
        validation logodds
    labels: numpy logodds
        binary labels

    Returns
    -------
    get_bin_reprs: function
        returns a function which takes in bin_boundaries

    """
    def get_bin_reprs(bin_boundaries):
        return hb_utils.bin_representation_calculation(logodds, labels, num_bins, bin_repr_scheme, bin_boundaries=bin_boundaries)
    return get_bin_reprs


def bin_boundary_function():
    def get_bin_boundary(representations):
        return hb_utils.bin_boundary_update_closed_form(representations)
    return get_bin_boundary




def mutual_information_and_R_function(p_y_pos, distr_kde_dict):
    '''logodds_c => the logodds which were used to bin. rewrote MI loss: sum_Y sum_B p(y'|lambda)p(lambda) for term outside log. Before it was p(lambda|y')p(y') '''
    # NOTE: checked and matches impl of Dan: -1*MI_eval(**kwargs) => all good
    def get_MI(bin_boundaries):
        return hb_utils.MI_known_LLR(bin_boundaries, p_y_pos, distr_kde_dict)
    return get_MI

def unknown_LLR_mutual_information(p_y_pos, logodds):
    def get_MI(bin_boundaries, representations):
        return hb_utils.MI_unknown_LLR(p_y_pos, logodds, bin_boundaries, representations)
    return get_MI    

def fit_kde_distributions(logodds, y):
    """
    Fit KDEs to the data based on the labels. Get KDE-pos and KDE-neg.

    Parameters
    ----------
    logoddds: numpy ndarray
        logodds (lambda)
    y: numpy ndarray
        binary labels indicating positive or negative label

    Returns
    -------
    dict: 
        KDE dictionary with "pos" and "neg"
    """
    distr_pos = scipy.stats.gaussian_kde(logodds[y==1])
    distr_neg = scipy.stats.gaussian_kde(logodds[y==0])
    return io.AttrDict({"pos":distr_pos, "neg":distr_neg})
   















































