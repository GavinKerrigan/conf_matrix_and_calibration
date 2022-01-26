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
"""
Created on Tue Mar 20 10:03:33 2018

@author: pak2rng
"""
import os
import numpy as np
from attrdict import AttrDict
import deepdish
import time

def deepdish_read(fpath, group=None):
    ''' Read all data inside the hdf5 file '''
    data = deepdish.io.load(fpath, group=group)
    if isinstance(data, dict):
        data = AttrDict(data)
    return data

def deepdish_write(fpath, data):
    ''' Save a dictionary as a hdf5 file! '''
    create_dir_for_fpath(fpath)
    deepdish.io.save(fpath, data, compression="None")
    



class Logger:
    def __init__(self, fpath):
        self.fpath = fpath
        self.logdata = AttrDict({})        

    def log(self, key, value):
        if key not in self.logdata:  self.logdata[key] = []
        self.logdata[key].append(value)

    def last(self, key):
        return self.logdata[key][-1]

    def log_dict(self, dictionary, suffix=""):
        # logging each element in the dictionary
        suffix = "_%s"%(suffix) if (suffix != "" and suffix[0]!="_") else suffix
        for k,v in dictionary.items():
            self.log(k+suffix,v)


    def end_log(self):
        for k,v in self.logdata.items():
            self.logdata[k] = np.array(v) if isinstance(v, list) else v

    def save_log(self):
        deepdish_write(self.fpath, self.logdata)













