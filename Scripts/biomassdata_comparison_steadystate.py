# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:05:06 2020

@author: khurana
"""
import os
import pandas as pd

#Load data
parent_dir = "E:\Zenodo_spatial_heterogeneity"
path_data = os.path.join(parent_dir, "Results","biomass_steadystate_BG.csv")
data = pd.read_csv(path_data)
data.columns
data.dtypes

regimes = data.Regime.unique().tolist()
chem_series = data.Chem.unique().tolist()
trial_series = data.Trial.unique().tolist()

for r in regimes:
    for t in trial_series:
        for c in chem_series:
            mass_spat_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == 'H')]['Mass'].values[0]
            cont_spat_base = data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial =='H')]['Contribution'].values[0]
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'Mass_spatial_base'] = mass_spat_base
            data.loc[(data.Regime == r) & (data.Chem == c) & (data.Trial == t), 'Cont_spatial_base'] = cont_spat_base
        
data['meanmass_spatial_fraction'] = data['Mass']/data['Mass_spatial_base']
data['contribution_spatial_fraction'] = data['Contribution']/data['Cont_spatial_base']

results_file = os.path.join(parent_dir, "Results","biomass_comparison_steadystate_BG.csv")
data.to_csv(results_file,index = False)