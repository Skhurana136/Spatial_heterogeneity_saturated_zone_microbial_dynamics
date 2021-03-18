# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:05:06 2020

@author: khurana
"""
import os
import pandas as pd
import numpy as np

#Load data
parent_dir = "E:\Zenodo_spatial_heterogeneity"
path_data = os.path.join(parent_dir, "massflux_steadystate_BG.csv")
data = pd.read_csv(path_data)
data.columns
data.dtypes

regimes = data.Regime.unique().tolist()
chem_series = data.Chem.unique().tolist()
trial_series = data.Trial.unique().tolist()
spatial_base = 'H'

for r in regimes:
    for t in trial_series:
        for c in chem_series:
            spat_n_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H')]['normmassflux'].values[0]
            spat_r_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H')]['reldelmassflux'].values[0]
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'spatial_normmassflux_base'] = spat_n_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'spatial_reldelmassflux_base'] = spat_r_base
        
data['normmassflux_spatial_fraction'] = data['normmassflux']/data['spatial_normmassflux_base']
data['reldelmassflux_spatial_fraction'] = data['reldelmassflux']/data['spatial_reldelmassflux_base']
data["Da63"] = np.log(data.normmassflux)

results_file = os.path.join(parent_dir, "massflux_comparison_steadystate_BG.csv")
data.to_csv(results_file,index = False)
