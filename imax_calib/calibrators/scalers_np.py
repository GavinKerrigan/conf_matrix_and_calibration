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
 calibrators_np.py
 imax_calib

All calibration methods which require numpy functions during learning of parameters.

 Created by Kanil Patel on 07/27/20.
 Copyright 2020. Kanil Patel. All rights reserved.
'''
import numpy as np
import imax_calib.io as io
import imax_calib.utils as utils

class BaseCalibrator():
    """
    A generic base class.   
    """
    def __init__(self):
        self.parameter_list = []

    def fit(self, logits, logodds, y, **kwargs):
        """
        Function to learn the model parameters using the input data X and labels y.

        Parameters
        ----------
        logits: numpy ndarray
            input data to the calibrator. 
        logodds: numpy ndarray
            input data to the calibrator. 
        y: numpy ndarray
            target labels
        Returns
        -------

        """
        raise NotImplementedError("Subclass must implement this method.")

    def calibrate(self, logits, logodds, **kwargs):
        """
        Calibrate the data using the learned parameters after fit was already called.
        """
        raise NotImplementedError("Subclass must implement this method.")

    def __call__(self, *args, **kwargs):
        return self.calibrate(*args, **kwargs)

    def save_params(self, fpath):
        """
        Save the parameters of the model. The parameters which need to be saved are determined by self.parameter_list.
        Saves a single hdf5 file with keys being the parameter names. 

        Parameters
        ----------
        fpath: string
            filepath to save the hdf5 file with model parameters
        Returns
        -------
        """
        if len(self.parameter_list)>0:
            data_to_save = io.AttrDict()
            for key in self.parameter_list:
                data_to_save[key] = getattr(self, key)
            io.deepdish_write(fpath, data_to_save)
            print(io.pc._OKGREEN("Parameters written to fpath: %s"%(fpath)))

    def load_params(self, fpath):
        """
        Load the parameters of the model. The parameters which need to be loaded are determined by self.parameter_list.
        Loads a single hdf5 file and assigns the attributes to the object using keys as the parameter names. 

        Parameters
        ----------
        fpath: string
            filepath to save the hdf5 file with model parameters
        Returns
        -------
        """
        if len(self.parameter_list)>0:
            data_to_load = io.deepdish_read(fpath)
            for key in self.parameter_list:
                setattr(self, key, data_to_load[key])
            print(io.pc._OKGREEN("Parameters loaded and updated from fpath: %s"%(fpath)))




class Raw(BaseCalibrator):
    """
    The raw outputs without any calibration. Identity function.
    """
    def __init__(self, cfg=None):
        super(Raw).__init__()

    def fit(self, logits, logodds, y, **kwargs):
        return self

    def calibrate(self, logits, logodds, **kwargs):
        probs = utils.to_sigmoid(logodds)
        return logits, logodds, probs

    def load_params(self, fpath):
        return None





































