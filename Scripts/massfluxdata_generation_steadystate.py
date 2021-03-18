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

#set up basic constants 
Regimes = ["Slow", "Equal", "Fast"]
fpre = "NS-A"
scdict = proc.masterscenarios() #master dictionary of all spatially heterogeneous scenarios that were run

# Scenarios to investigate:
Trial = list(scdict.keys())
reginvest = Regimes

species = proc.speciesdict("Saturated")
gvarnames = ["DOC","DO","Nitrate", "Ammonium","Nitrogen", "TOC"]

parent_dir = "E:\Zenodo_spatial_heterogeneity"

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
        massfluxin, massfluxout = sssa.massflux(data, 0, -1, 0, -1, gvarnames, "Saturated")
        delmassflux = massfluxin - massfluxout
        reldelmassflux = 100*delmassflux/massfluxin
        normmassflux = massfluxout/massfluxin
        for g in gvarnames:
            row.append([j,scdict[j]['Het'], scdict[j]['Anis'], r, g, massfluxin[gvarnames.index(g)], massfluxout[gvarnames.index(g)],delmassflux[gvarnames.index(g)], reldelmassflux[gvarnames.index(g)], normmassflux[gvarnames.index(g)]])

massfluxdata = pd.DataFrame.from_records (row, columns = ["Trial", "Variance", "Anisotropy", "Regime", "Chem", "massflux_in", "massflux_out","delmassflux", "reldelmassflux", "normmassflux"])

#Load tracer data
tracer_data_path = os.path.join(parent_dir, "Results", "tracer_combined_05032020.csv")
tr_data = pd.read_csv(tracer_data_path, sep = "\t")
tr_data.columns
tr_data['Regime'] = tr_data['Regime'].replace({'Equal':'Medium'})

#Merge the datasets and save
cdata = pd.merge(massfluxdata, tr_data[["Trial", "Regime", "Time", "fraction"]], on = ["Regime", "Trial"])

massflux_path = os.path.join(parent_dir, "Results", "massflux_steadystate_BG.csv")
cdata.to_csv(massflux_path, index=False)
