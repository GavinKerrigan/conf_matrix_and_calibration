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
 calibration.py
 imax_calib

 Created by Kanil Patel on 07/28/20.
 Copyright 2020. Kanil Patel. All rights reserved.
'''
import os
import numpy as np
import imax_calib.io as io
import imax_calib.utils as utils
import imax_calib.calibrators.binners as binners
import imax_calib.calibrators.scalers_np as scalers_np 

def learn_calibrator(cfg, logits, logodds, y, feats=None, **kwargs):
    """
    Use this function to access all calibrators (binning).
    Inputs are the raw network logits and one-hot labels.
    The kwargs can be used to send other arguments which some calibrators might need. 

    Parameters
    ----------
    cfg: io.AttrDict
        config dictionary containing all information. 
    logits: numpy ndarray
        raw network logits
    logodds: numpy ndarray
        raw network logodds. use utils.quick_logits_to_logodds(logits) to get them 
    y: numpy ndarray
        one-hot target labels
    kwargs: dict
        extra arguments which some calibrators require
    Returns
    -------

    cal_obj: calibrators_*.BaseCalibrator
        calibrator object. can be used given logits as input
    """
    binner_obj = learn_binning(cfg, logits, logodds, y, **kwargs)
    return binner_obj 

def learn_binning(cfg, logits, logodds, y, **kwargs):
    """
    Same as learn_calibrator() but this func specifically learns the logodds binning methods.
    """
    # set all seeds
    np.random.seed(cfg.Q_rnd_seed)

    if cfg.Q_method is None:
        CALIBRATOR = scalers_np.Raw
    elif cfg.Q_method=="imax" or cfg.Q_method=="eqmass" or cfg.Q_method=="eqsize":
        if cfg.cal_setting=="CW":
            CALIBRATOR = binners.HistogramBinninerCW
        elif cfg.cal_setting=="top1":
            CALIBRATOR = binners.HistogramBinninerTop1
        elif cfg.cal_setting=="sCW":
            CALIBRATOR = binners.HistogramBinninerSharedCW
    else:
        raise Exception("Quantization method unknown!")

    cal_obj = CALIBRATOR(cfg)
    #print("Learning calibration parameters!")
    cal_obj.fit(logits, logodds, y, **kwargs)
    return cal_obj




































