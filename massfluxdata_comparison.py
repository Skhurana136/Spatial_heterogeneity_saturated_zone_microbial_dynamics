# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:05:06 2020

@author: khurana
"""

import pandas as pd
import numpy as np

#Load data
path_data = "Y:/Home/khurana/4. Publications/Restructuring/Paper1/Figurecodes/massflux_Original_complete_28012021.csv"
data = pd.read_csv(path_data, sep = "\t")
data.columns
data.dtypes

regimes = data.Regime.unique().tolist()
Time_series = data.Time_series.unique().tolist()
chem_series = data.Chem.unique().tolist()
trial_series = data.Trial.unique().tolist()
spatial_base = 'H'
temp_base = 0

for r in regimes:
    for t in trial_series:
        for c in chem_series:
            spat_n_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H') & (data.Time_series == 0)]['normmassflux'].values[0]
            tim_n_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t) & (data.Time_series == 0)]['normmassflux'].values[0]
            spat_r_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H') & (data.Time_series == 0)]['reldelmassflux'].values[0]
            tim_r_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t) & (data.Time_series == 0)]['reldelmassflux'].values[0]
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'temporal_normmassflux_base'] = tim_n_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'spatial_normmassflux_base'] = spat_n_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'temporal_reldelmassflux_base'] = tim_r_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'spatial_reldelmassflux_base'] = spat_r_base
        
data['normmassflux_temporal_fraction'] = data['normmassflux']/data['temporal_normmassflux_base']
data['normmassflux_spatial_fraction'] = data['normmassflux']/data['spatial_normmassflux_base']
data['reldelmassflux_temporal_fraction'] = data['reldelmassflux']/data['temporal_reldelmassflux_base']
data['reldelmassflux_spatial_fraction'] = data['reldelmassflux']/data['spatial_reldelmassflux_base']
data["Da63"] = np.log(data.normmassflux)

data.to_csv("Y:/Home/khurana/4. Publications/Restructuring/Paper1/Figurecodes/massflux_comparison_Original_complete_28012021.csv", sep ="\t", index = False)
