# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:05:06 2020

@author: khurana
"""

import pandas as pd

#Load data
path_data = "Y:/Home/khurana/4. Publications/Restructuring/Paper1/Figurecodes/biomass_Original_complete.csv"
data = pd.read_csv(path_data, sep = "\t")
data.columns
data.dtypes

regimes = data.Regime.unique().tolist()
Time_series = data.Time_series.unique().tolist()
chem_series = data.Chem.unique().tolist()
trial_series = data.Trial.unique().tolist()

for r in regimes:
    for t in trial_series:
        for c in chem_series:
            mass_temp_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t) & (data.Time_series == 0)]['Mass'].values[0]
            cont_temp_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t) & (data.Time_series == 0)]['Mass'].values[0]
            mass_spat_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H') & (data.Time_series == 0)]['Mass'].values[0]
            cont_spat_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial =='H') & (data.Time_series == 0)]['Mass'].values[0]
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'Mass_temporal_base'] = mass_temp_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'Cont_temporal_base'] = cont_temp_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'Mass_spatial_base'] = mass_spat_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'Cont_spatial_base'] = cont_spat_base
        
data['meanmass_temporal_fraction'] = data['Mass']/data['Mass_temporal_base']
data['contribution_temporal_fraction'] = data['Contribution']/data['Cont_temporal_base']

data['meanmass_spatial_fraction'] = data['Mass']/data['Mass_spatial_base']
data['contribution_spatial_fraction'] = data['Contribution']/data['Cont_spatial_base']


data.to_csv("//msg-filer2/scratch_60_days/khurana/biomass_comparison_Original_complete_06022021.csv", sep ="\t")
