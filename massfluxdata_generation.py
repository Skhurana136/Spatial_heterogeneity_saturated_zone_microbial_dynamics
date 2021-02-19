# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 12:05:06 2020

@author: khurana
"""

import numpy as np
import data_reader.data_processing as proc
import pandas as pd
import analyses.saturated_steady_state as sssa
import analyses.saturated_transient as sta

#set up basic constants 
Regimes = ["Slow", "Equal", "Fast"]
domains = ["Original", "Half", "Double", "Big"]
domainodes = {"Original": {'ynodes' : 51},
              "Big" : {'ynodes' : 126},
              "Double" : {'ynodes' : 101},
              "Half" : {'ynodes' : 26}}
fpre = "NS-A"
scdict = proc.masterscenarios() #master dictionary of all spatially heterogeneous scenarios that were run
fsuf = r"/"
filename = "model_domain_quad.tec" #same filename in each subfolder
horiznodes = 31

# Scenarios to investigate:
Trial = list(scdict.keys())
#Trial = ["H", "37", "38", "39", "40", "41", "42", "43", "44", "45"]
reginvest = Regimes
domaininvest = list(domainodes.keys())[:1]

species = proc.speciesdict("Saturated")
gvarnames = ["DOC","DO","Nitrate", "Ammonium","Nitrogen", "TOC"]

Trial.remove("43")

row = []
for Reg in reginvest:
    for domain in domaininvest:
        if domain != "Original":
            domadd = domain + "_"
        else:
            domadd = ""
        for t in ["0", "1", "2", "5"]:
            directory = "E:/Saturated_flow/EGUGoldschmidtdataset6/" + domadd + Reg + "AR_" + t + "/"
            #directory = "X:/Saturated_flow/changedkindox_transient/" + domadd + Reg + "AR_" + t + "/"#change directory as per flow regime
            print (Reg, domain, t)
            for j in Trial:
                data = np.load(directory + "NS-A"+j+"/NS-A"+j+"_df.npy")
                if t == "0":
                    massfluxin, massfluxout = sssa.calcmassfluxnew(data, 0, -1, 0, -1, gvarnames, "Saturated")
                else:
                    massfluxin, massfluxout = sta.calcmft_temp(data, 0, -1, 0, -1, gvarnames, "Saturated")
                delmassflux = massfluxin - massfluxout
                reldelmassflux = 100*delmassflux/massfluxin
                normmassflux = massfluxout/massfluxin
                for g in gvarnames:
                    row.append([j,scdict[j]['Het'], scdict[j]['Anis'], domain, Reg, t, g, massfluxin[gvarnames.index(g)], massfluxout[gvarnames.index(g)],delmassflux[gvarnames.index(g)], reldelmassflux[gvarnames.index(g)], normmassflux[gvarnames.index(g)]])

massfluxdata = pd.DataFrame.from_records (row, columns = ["Trial", "Variance", "Anisotropy", "Domain", "Regime", "Time_series", "Chem", "massflux_in", "massflux_out","delmassflux", "reldelmassflux", "normmassflux"])

#Load tracer data
path_tr_data = "Y:/Home/khurana/4. Publications/Restructuring/Paper1/Figurecodes/tracer_combined_05032020.csv"
tr_data = pd.read_csv(path_tr_data, sep = "\t")
tr_data.columns

#Merge the datasets and save
cdata = pd.merge(massfluxdata, tr_data[["Trial", "Regime", "Time", "fraction"]], on = ["Regime", "Trial"])

cdata.to_csv("Y:/Home/khurana/4. Publications/Restructuring/Paper1/Figurecodes/massflux_Original_complete_28012021.csv", sep = "\t", index=False)
