# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:05:06 2020

@author: khurana
"""
import os
import numpy as np
import data_reader.data_processing as proc
import pandas as pd
import analyses.steady_state as sssa

#File paths
parent_dir = "E:\Zenodo_spatial_heterogeneity"
#set up basic constants 
Regimes = ["Slow", "Equal", "Fast"]

fpre = "NS-A"
scdict = proc.masterscenarios() #master dictionary of all spatially heterogeneous scenarios that were run

# Scenarios to investigate:
Trial = list(scdict.keys())
#Trial = ["H", "37", "38", "39", "40", "41", "42", "43", "44", "45"]
reginvest = Regimes

vardict = proc.speciesdict("Saturated")
states = ["Active", "Inactive"]
gvarnames = list(t for t in vardict.keys() if vardict[t]["State"] in (states))

row = []
for Reg in reginvest:
    if Reg == "Equal":
        r = "Medium"
    else:
        r = Reg
    directory = os.path.join(parent_dir, "Data")
    print(Reg)
    for j in Trial:
        path = os.path.join(directory, Reg+"AR_0_NS-A"+j+"_df.npy")
        data = np.load(path)
        meanmass = sssa.sum_biomass(data, 0, -1, 0, -1, gvarnames, "Saturated")
        summass = sum(meanmass)
        masscontribution = meanmass/summass
        for g in gvarnames + ["Total"]:
            if g == "Total":
                row.append([j,scdict[j]['Het'], scdict[j]['Anis'], r, g, summass, 1])
            else:
                row.append([j,scdict[j]['Het'], scdict[j]['Anis'], r, g, meanmass[gvarnames.index(g)], masscontribution[gvarnames.index(g)]])

massdata = pd.DataFrame.from_records (row, columns = ["Trial", "Variance", "Anisotropy", "Regime", "Chem", "Mass", "Contribution"])

#Load tracer data
tracer_data_path = os.path.join(parent_dir, "Results", "tracer_combined_05032020.csv")
tr_data = pd.read_csv(tracer_data_path, sep = "\t")
tr_data.columns
tr_data['Regime'] = tr_data['Regime'].replace({'Equal':'Medium'})

#Merge the datasets and save
cdata = pd.merge(massdata, tr_data[["Trial", "Regime", "Time", "fraction"]], on = ["Regime", "Trial"])

biomass_path = os.path.join(parent_dir, "Results", "biomass_steadystate_BG.csv")
cdata.to_csv(biomass_path, index=False)