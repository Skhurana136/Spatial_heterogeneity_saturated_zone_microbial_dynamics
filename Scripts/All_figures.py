# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 17:36:05 2020

@author: khurana
"""
#loading required libraries
import os
import pandas as pd
#import plots.saturated_steady_state as sssp
import analyses.transient as sta
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl  
from data_reader import data_processing as proc
from data_reader.data_processing import tracerstudies

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

#File paths
DIR = "E:/Zenodo_spatial_heterogeneity"
results_dir = os.path.join(DIR,"Results")
raw_directory = os.path.join(DIR, "Data")
os.chdir(results_dir)
#Standard color and font options
my_pal = {2:"indianred", 11:"g", 22:"steelblue", "DO":"indianred", "Nitrate":"g", "Ammonium":"steelblue",
          "Slow":"indianred", "Medium":"g", "Fast":"steelblue"}
legendkw = {'fontsize' : 14}
labelkw = {'labelsize' : 14}
secondlabelkw = {'labelsize' : 16}
suptitlekw = {'fontsize' : 18}
titlekw = {'fontsize' : 16}
mpl.rc('font',family='Arial')
#plt.rcParams["axes.grid"] = False

#Figure 2 Impact on tracer breakthrough

filename = "tracer_combined_05032020.csv" #processed data saved in this file

#plotting boxplots to see variance of breakthrough from homogeneous scenario
combined_tracer = tracerstudies(filename)
combined_tracer["%fraction"] = combined_tracer["fraction"] * 100
combined_tracer["%fraction_withslow"] = combined_tracer["fraction_withslow"] * 100
sns.set(rc={"figure.figsize": (8, 4)})
sns.set_style("whitegrid",{'axes.grid' : False})
sns.boxplot(x="Xlabels",y="%fraction",hue="Regime",data=combined_tracer,hue_order=["Slow", "Medium", "Fast"],
        palette=["coral", "mediumseagreen", "steelblue"])
plt.xlabel("Variance:Anisotropy", **legendkw)
plt.ylabel("% of homogeneous scenario", **legendkw)
plt.title("Time taken for tracer breakthrough", **suptitlekw)
plt.savefig("Figure2_tracer_breakthrough_impact.png", dpi = 300, pad_inches = 0.1)
plt.savefig("Figure2_tracer_breakthrough_impact.pdf", dpi = 300, pad_inches = 0.1)

#Figure 3 Impact of spatial heterogeneity on mass flux 2x2 format

#Load mass flux data
massfluxdata = pd.read_csv("massflux_comparison_steadystate_BG.csv")

#Variables to plot
chemicalgvarnames = ["DOC", "DO","Nitrogen","TOC"]

Regimes = ["Slow", "Medium", "Fast"]
Chems = ["DOC", "DO", "Nitrogen", "TOC"]
colseries = ["indianred", "g", "steelblue"]
nrows = 2
ncols = 2
data = massfluxdata
data["fraction"] = data.fraction*100
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=[11, 8], sharex=True)
for k in Chems:
    dfc = data[data["Chem"] == k]
    colidx1 = Chems.index(k)
    for i in Regimes:
        dfctemp = dfc
        dfcr = dfctemp[dfctemp["Regime"] == i]
        print(i)
        axes.flat[colidx1].scatter(
                "fraction",
                "reldelmassflux_spatial_fraction",
                color=colseries[Regimes.index(i)],
                data=dfcr,
                label=i + " flow",
        )
        axes.flat[colidx1].set_ylabel(k, **legendkw)
        axes.flat[colidx1].tick_params(axis="y", labelsize = 14)
        axes.flat[colidx1].tick_params(axis="x", labelsize = 14)
plt.legend(loc="best", **legendkw)
plt.annotate(
    "Normalized reduction in mass flux in the domain",
    xy=(-1.2, 1),
    xytext=(-80, 0),
    xycoords="axes fraction",
    textcoords="offset points",
    size="large",
    ha="left",
    va="center",
    rotation="vertical",
    **suptitlekw)
plt.annotate(
    "Residence time of solutes (%)",
    xy=(-0.6, 0),
    xytext=(0, -50),
    xycoords="axes fraction",
    textcoords="offset points",
    size="large",
    ha="left",
    va="center",
    **suptitlekw)
plt.grid(False)
plt.savefig("Figure3_massflux_impact.png", dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
plt.savefig("Figure3_massflux_impact.pdf", dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#Figure 4 Impact of spatial heterogeneity on active biomass
#Load biomass data
biomassdata = pd.read_csv("biomass_comparison_steadystate_BG.csv")
#Variable names of interest
activebiomassgvarnames = ["Immobile active aerobic degraders",
                          "Immobile active ammonia oxidizers",
                          "Immobile active nitrate reducers"]

#Plot and save
Regimes = ["Slow", "Medium", "Fast"]
Chems = activebiomassgvarnames
species = ["Aerobes", "Ammonia oxidizers", "Nitrate reducers"]
colseries = ["indianred", "g", "steelblue"]
data = biomassdata
data.fraction = data.fraction*100
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=[11, 5], sharex = True)
for k in Chems:
    dfc = data[data["Chem"] == k]
    colidx1 = Chems.index(k)
    for i in Regimes:
        dfctemp = dfc
        dfcr = dfctemp[dfctemp["Regime"] == i]
        print(i, colidx1)
        axes.flat[colidx1].scatter(
            "fraction",
            "meanmass_spatial_fraction",
            color=colseries[Regimes.index(i)],
            data=dfcr,
            label=i + " flow",
        )
        axes.flat[colidx1].tick_params(axis="y", **labelkw)
        axes.flat[colidx1].tick_params(axis="x", **labelkw)
plt.legend(loc="best", **legendkw)
for ax, typsp in zip(axes, species):
    ax.set_title(typsp, **titlekw)
plt.annotate(
    "Normalized biomass in the domain",
    xy=(-2.5, 0.5),
    xytext=(-40, 0),
    xycoords="axes fraction",
    textcoords="offset points",
    size="large",
    ha="left",
    va="center",
    rotation="vertical",
    **suptitlekw)
plt.annotate(
    "Residence time of solutes (%)",
    xy=(-0.5, -0.1),
    xytext=(-40, -15),
    xycoords="axes fraction",
    textcoords="offset points",
    size="large",
    ha="center",
    va="center",
    **suptitlekw)
plt.xticks([20,60,100], [20,60,100])
plt.grid(False)
plt.savefig("Figure4_active_biomass_impact.png", dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
plt.savefig("Figure4__active_biomass_impact.pdf", dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#Figure5: Regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
grey_line = mlines.Line2D([], [], color='grey', markersize=15, label='Linear regression')
grey_dot = mlines.Line2D([], [], linestyle = '', marker = "o", markerfacecolor = "grey", markeredgecolor = "grey", markersize=10, label='Data', alpha = 0.5)
grey_triangle = mlines.Line2D([], [], linestyle = '', marker = "^", markerfacecolor = "grey", markeredgecolor = "grey",markersize=10, label='Test data', alpha = 0.5)
my_pal = {3:"indianred", 2: "g", 0:"steelblue", 1 :"orange"}
blue_patch = mpatches.Patch(color="steelblue", label= 'log$_{10}$Da < -1', alpha = 0.5)
orange_patch = mpatches.Patch(color = "orange", label =  r'-1 < log$_{10}$Da < 0', alpha = 0.5)
green_patch = mpatches.Patch(color="g", label='0 < log$_{10}$Da < 0.5', alpha = 0.5)
red_patch = mpatches.Patch(color="indianred", label= 'log$_{10}$Da > 0.5', alpha = 0.5)
patchlist = [blue_patch, orange_patch, green_patch, red_patch, grey_line, grey_dot]

path_da_data= os.path.join(results_dir, "Da_BG.csv")
#path_da_data= os.path.join(results_dir, "massflux_comparison_steadystate_BG.csv")
da = pd.read_csv(path_da_data)
gvarnames = ["DO", "Nitrogen", "DOC"]
data = da[da['Chem'].isin (gvarnames)]

data["logDa"] = np.log10(data.Da)
data.loc[data["logDa"] < -1, "PeDamark"] = 0
data.loc[(data["logDa"] > -1) & (data["logDa"] < 0), "PeDamark"] = 1
data.loc[(data["logDa"] > 0) & (data["logDa"] <0.5), "PeDamark"] = 2
data.loc[(data["logDa"] > 0.5), "PeDamark"] = 3
labels = {3 : "log$_{10}$Da > 0.5",
          2 : "0 < log$_{10}$Da < 0.5",
          1 : "-1 < log$_{10}$Da < 0",
         0 : "log$_{10}$Da < -1"}

data["pc_reldelmassflux_spatial"] = data.reldelmassflux_spatial_fraction * 100

mymarklist = ["^", "o", "s", "d"]

plt.figure (figsize = (8,8))
for frac in [0,1,2,3]:
    subset = data[data['PeDamark'] == frac]
    #y = subset["%fraction_rel_delmf"]
    y = subset["pc_reldelmassflux_spatial"]
    X = subset[["fraction"]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    reg = LinearRegression().fit(X_train, y_train)
    X_test_sorted = X_test.sort_values(by = "fraction")
    ypred = reg.predict(X_test_sorted)
    plt.plot(X_test_sorted*100, ypred, c = my_pal[frac], label = labels[frac]+ ": regression")
    plt.scatter(X*100, y, c = my_pal[frac], marker = 'o',alpha = 0.5, label = labels[frac])
    coef = reg.coef_
    inter = reg.intercept_
    rmse = mean_squared_error(y_test, reg.predict(X_test), squared = False)
    print("Category :",frac)
    print("Test score: {:.3f}".format(rmse))
    print("Coefficients: ", coef)
    print("Intercept: ", inter)
plt.legend(loc="best", handles = patchlist, **legendkw)
plt.xlabel ("Residence time of solutes (%)", **titlekw)
plt.ylabel("Removal of reactive species (%)", ha='center', va='center', rotation='vertical', labelpad = 10, **titlekw)
locs, labels1 = plt.yticks()
plt.yticks((10,40,100,300,600), (10,40,100,300,600), **legendkw)
plt.tick_params(axis="x", **labelkw)
plt.grid(False)
picname = "Figure5_Dat_removal.png"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = "Figure5_Dat_removal.pdf"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#Figure 6 comparison of DO and aerobic cells in the cross-section - requires access to raw data
fpre = 'NS-A'
limit = 3

#Default:
Trial = proc.masterscenarios()
species = proc.speciesdict("Saturated")
#Constants
yout = -1
yin = 0
xleft = 0
xright = -1

#Assign index to Variable
gvarnames = ["DO"]

#Classify oxic cells and then plot along Y axis for the medium flow regime
intsce = ["H", "50", "76", "73", "80", "84", "44", "63"]

oxiccells = np.zeros([len(intsce), 51])
for j in intsce:
    data = np.load(os.path.join(raw_directory,"EqualAR_0_"+fpre+j+"_df.npy"))
    conctime, TotalFlow, Headinlettime = sta.conc_time (data,yin,yout,xleft,xright, 51, gvarnames,"Saturated")
    c = []
    for k in range(51):
        c.append(np.count_nonzero(data[species["DO"]["TecIndex"], np.shape(data)[1] - 1, k, :]>limit))
    oxiccells[intsce.index(j), :] = c

nrows = 2
ncols = 4
figsize = [11, 8]
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize, sharex=True, sharey=True)
count = 0
for j in intsce:
    data = np.load(os.path.join(raw_directory,"EqualAR_0_"+fpre+str(j)+"_df.npy"))
    conctime, TotalFlow, Headinlettime = sta.conc_time (data,yin,yout,xleft,xright, 51, gvarnames,"Saturated")
    axes.flat[count].plot(conctime[-1, :, gvarnames.index("DO")], "r-")
    axes.flat[count].set_ylim(0, 260)
    axes.flat[count].tick_params(axis="y", colors="r", **labelkw)
    axes.flat[count].set_title(
        "Variance: "
        + str(Trial[j]["Het"])
        + " &\nAnisotropy: "
        + str(Trial[j]["Anis"]),
        **titlekw)
    ax2 = axes.flat[count].twinx()
    ax2.plot((oxiccells[intsce.index(j), :] / 31) * 100, "b-")
    ax2.set_ylim(0, 105)
    ax2.tick_params(axis="y", colors="b", **labelkw)
    if (count + 1) % ncols == 0:
        ax2.set_ylabel("% of oxic cells (-)", color="b", **titlekw)
    else:
        ax2.set_yticklabels([])  # turning off secondary y axis yticklabels
    count = count + 1
for ax in axes[:, 0]:
    ax.set_ylabel("DO (uM)", color="r", **titlekw)
for ax in axes[-1]:
    ax.set_xlabel("Y (cm)", **titlekw)
plt.grid(False)
picname = "Figure6_DO_and_aerobic_cells.png"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = "Figure6_DO_and_aerobic_cells.pdf"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#Figure S1: 1D profile dissolved species
Regimes = ["Slow", "Equal", "Fast"]
trialist = proc.masterscenarios()
Trial = ["50", "73", "63"]
species = proc.speciesdict("Saturated")
gvarnames = ["DO", "DOC", "Ammonium", "Nitrate"]
cvars = list(species[g]['TecIndex'] for g in gvarnames)
velindex = 2
colors = ["red", "black", "blue", "darkgreen"]
columntitles = ["Slow flow", "Medium flow", "Fast flow"]
pad = 230
figbig, axes = plt.subplots(3,3, figsize=(13, 10), sharey = True, sharex = True)
for t in Trial:
    for r in Regimes:
        if r == "Equal":
            rtitle = "Medium"
        else:
            rtitle = r
        fileh = os.path.join(raw_directory, r + "AR_0_NS-AH_df.npy")
        datah = np.load(fileh)
        i = Trial.index(t)*len(Regimes) + Regimes.index(r)
        host = axes.flat[i]
        file = os.path.join(raw_directory, r + "AR_0_NS-A"+str(t)+"_df.npy")
        data = np.load(file)
        #DIR = d + "/" + r + "AR_0/"
        conctime, TotalFlow, Headinlettime = sta.conc_time (data,0,50,0,30, 51, gvarnames,"Saturated")
        conctimeh, TotalFlowh, Headinlettimeh = sta.conc_time (datah,0,50,0,30, 51, gvarnames,"Saturated")
        yindex = list(range(51))
        #fig, host = axe.subplots()
        host.plot(conctimeh[-1, :, 0],yindex,label=gvarnames[0],color=colors[0],linestyle="-.")
        host.plot(conctime[-1, :, 0],yindex,label=gvarnames[0],color=colors[0],linestyle="-")
        host.plot(conctimeh[-1, :, 1],yindex,label=gvarnames[1],color=colors[1],linestyle="-.")
        host.plot(conctime[-1, :, 1],yindex,label=gvarnames[1],color=colors[1],linestyle="-",)
        par1 = host.twiny()
        par2 = host.twiny()

        # Offset the top spine of par2.  The ticks and label have already been
        # placed on the top by twiny above.
        par2.spines["top"].set_position(("axes", 1.2))
        # Having been created by twinx, par2 has its frame off, so the line of its
        # detached spine is invisible.  First, activate the frame but make the patch
        # and spines invisible.
        make_patch_spines_invisible(par2)
        # Second, show the right spine.

        par1.plot(conctimeh[-1, :, 2],yindex,label=gvarnames[2],color=colors[2],linestyle="-.")
        par1.plot(conctime[-1, :, 2],yindex,label=gvarnames[2],color=colors[2],linestyle="-")
        par2.plot(conctimeh[-1, :, 3],yindex,label=gvarnames[3],color=colors[3],linestyle="-.")
        par2.plot(conctime[-1, :, 3],yindex,label=gvarnames[3],color=colors[3],linestyle="-")

        host.set_ylim(0, 51)
        host.set_xlim(0, 800)
        par1.set_xlim(30, 60)
        par2.set_xlim(50, 260)
        host.xaxis.label.set_color("black")
        tkw = dict(size=4, width=1.5, labelsize=14)
        host.tick_params(axis="x", colors="black", **tkw)
        host.tick_params(axis="y", **tkw)
        if i < 3:
            host.set_title (rtitle + " flow", **titlekw)
            par2.spines["top"].set_visible(True)
            par1.xaxis.label.set_color("blue")
            par2.xaxis.label.set_color("darkgreen")
            par1.tick_params(axis="x", colors="blue", **tkw)
            par2.tick_params(axis="x", colors="darkgreen", **tkw)
            par1.set_xlabel(str(gvarnames[2]) + ' ($\mu$M)', **legendkw)
            par2.set_xlabel(str(gvarnames[3]) + ' ($\mu$M)', **legendkw)
        elif i > 5:
            host.set_xlabel("DOC, DO ($\mu$M)", **legendkw)
            par1.set_xticks([])
            par2.set_xticks([])
        else:
            par1.set_xticks([])
            par2.set_xticks([])
figbig.gca().invert_yaxis()
figbig.subplots_adjust(top=1.0, hspace = 0.2, wspace = 0.2)
for t,a in zip(Trial[::-1],range(3)):
    plt.annotate("Variance: " + str(trialist[t]["Het"])+ " &\nAnisotropy: " + str(trialist[t]["Anis"]),
                 xy=(0.1, 0.17), xytext=(-50, 0.7 + pad*a),
                xycoords='figure fraction', textcoords='offset points',
                rotation = "vertical",
                size='large', ha='center', va='baseline',
                fontsize = 16)
    axes.flat[3*a].set_ylabel("Y (cm)", **legendkw)
plt.legend(handles = patchlist, ncol = 3, **legendkw,
           bbox_to_anchor = (-0.2,-0.6),
           loc = 'lower right')
plt.grid(False)
picname = "FigureS1_dissolved_species_1D.png"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = "FigureS1_dissolved_species_1D.pdf"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#Figure S2: 1D profile biomass
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
dashedline = mlines.Line2D([], [], linestyle = '-.', color='grey', markersize=15, label='Homogeneous')
solidline = mlines.Line2D([], [], linestyle = 'solid', color='grey', markersize=15, label='Heterogeneous')
blue_patch = mpatches.Patch(color="blue", label= 'Ammonia oxidizers', alpha = 0.5)
black_patch = mpatches.Patch(color="black", label= 'Aerobes', alpha = 0.5)
green_patch = mpatches.Patch(color="darkgreen", label='Nitrate reducers', alpha = 0.5)
patchlist = [blue_patch, dashedline, black_patch, solidline, green_patch]
Regimes = ["Slow", "Equal", "Fast"]
trialist = proc.masterscenarios()
Trial = ["50", "73", "63"]
species = proc.speciesdict("Saturated")
gvarnames = list(g for g in species.keys() if (species[g]["State"] == "Active") and (species[g]["Location"] == "Immobile"))
gvarnames.remove('Immobile active sulphate reducers')
cvars = list(species[g]['TecIndex'] for g in gvarnames)
velindex = 2
colors = ["black", "darkgreen", "blue"]
columntitles = ["Slow flow", "Medium flow", "Fast flow"]
pad = 230
figbig, axes = plt.subplots(3,3, figsize=(13, 10), sharey = True, sharex = True)
for t in Trial:
    for r in Regimes:
        if r == "Equal":
            rtitle = "Medium"
        else:
            rtitle = r
        fileh = os.path.join(raw_directory, r + "AR_0_NS-AH_df.npy")
        datah = np.load(fileh)
        i = Trial.index(t)*len(Regimes) + Regimes.index(r)
        host = axes.flat[i]
        file = os.path.join(raw_directory, r + "AR_0_NS-A" + str(t) + "_df.npy")
        data = np.load(file)
        masstime, conctime = sta.biomasstimefunc(data,0,50,0,30, 51, gvarnames,"Saturated")
        masstime, conctimeh = sta.biomasstimefunc(datah,0,50,0,30, 51, gvarnames,"Saturated")
        yindex = list(range(51))
        #fig, host = axe.subplots()
        host.plot(conctimeh[-1, :, 0],yindex,label=gvarnames[0],color=colors[0],linestyle="-.")
        host.plot(conctime[-1, :, 0],yindex,label=gvarnames[0],color=colors[0],linestyle="-")
        par1 = host.twiny()
        par2 = host.twiny()

        # Offset the top spine of par2.  The ticks and label have already been
        # placed on the top by twiny above.
        par2.spines["top"].set_position(("axes", 1.2))
        # Having been created by twinx, par2 has its frame off, so the line of its
        # detached spine is invisible.  First, activate the frame but make the patch
        # and spines invisible.
        make_patch_spines_invisible(par2)
        # Second, show the right spine.

        par1.plot(conctimeh[-1, :, 1],yindex,label=gvarnames[1],color=colors[1],linestyle="-.")
        par1.plot(conctime[-1, :, 1],yindex,label=gvarnames[1],color=colors[1],linestyle="-")
        par2.plot(conctimeh[-1, :, 2],yindex,label=gvarnames[2],color=colors[2],linestyle="-.")
        par2.plot(conctime[-1, :, 2],yindex,label=gvarnames[2],color=colors[2],linestyle="-")

        host.set_ylim(0, 51)
        host.set_xlim(0, 500)
        par1.set_xlim(0, 200)
        par2.set_xlim(0, 30)
        host.xaxis.label.set_color("black")
        tkw = dict(size=4, width=1.5, labelsize=14)
        host.tick_params(axis="x", colors="black", **tkw)
        host.tick_params(axis="y", **tkw)
        if i < 3:
            host.set_title (rtitle + " flow", **titlekw)
            par2.spines["top"].set_visible(True)
            par1.xaxis.label.set_color("blue")
            par2.xaxis.label.set_color("darkgreen")
            par1.tick_params(axis="x", colors="blue", **tkw)
            par2.tick_params(axis="x", colors="darkgreen", **tkw)
            par1.set_xlabel(species[gvarnames[1]]["Graphname"] + " ($\mu$M)", **legendkw)
            par2.set_xlabel(species[gvarnames[2]]["Graphname"] + " ($\mu$M)", **legendkw)
        elif i > 5:
            host.set_xlabel(species[gvarnames[0]]["Graphname"], **legendkw)
            par1.set_xticks([])
            par2.set_xticks([])
        else:
            par1.set_xticks([])
            par2.set_xticks([])
figbig.gca().invert_yaxis()
figbig.subplots_adjust(top=1.0, hspace = 0.2, wspace = 0.2)
for t,a in zip(Trial[::-1],range(3)):
    plt.annotate("Variance: " + str(trialist[t]["Het"])+ " &\nAnisotropy: " + str(trialist[t]["Anis"]),
                 xy=(0.1, 0.17), xytext=(-50, 0.7 + pad*a),xycoords='figure fraction',textcoords='offset points',rotation = "vertical",size='large', ha='center', va='baseline',fontsize = 16)
    axes.flat[3*a].set_ylabel("Y (cm)", **titlekw)
plt.legend(handles = patchlist, ncol = 3, bbox_to_anchor = (-0.6,-0.6),loc = 'lower center', **legendkw)
plt.grid(False)
picname = "FigureS2_biomass_1D.png"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = "FigureS2_biomass_1D.pdf"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#------ Figure S3: Distribution of dissolved species heatmap
import matplotlib.gridspec as gridspec
Regimes = ["Slow", "Equal", "Fast"]
trialist = proc.masterscenarios()
Trial = ["50", "73", "63"]
species = proc.speciesdict("Saturated")
gvarnames = ["DO", "DOC", "Ammonium", "Nitrate"]
velindex = 2
colorscheme = 'YlGnBu'
columntitles = ["Velocity\ndistribution pattern", "Slow\nflow", "Medium\nflow", "Fast\nflow"]
fig = plt.figure(figsize=(14, 14))
outer = gridspec.GridSpec(3, 4, wspace=0.2, hspace=0.2)
pad = 210
for t in Trial:
    file = os.path.join(raw_directory, "EqualAR_0_NS-A"+str(t)+"_df.npy")
    data = np.load(file)
    left = gridspec.GridSpecFromSubplotSpec(1, 1,
                subplot_spec=outer[4*Trial.index(t)], wspace=0.3, hspace=0.1)
    axe = plt.Subplot(fig, left[0])
    velocity = abs(data[velindex, -1, :, :])
    sns.heatmap(velocity, cmap = colorscheme, ax = axe, cbar = False)
    axe.set_ylabel ("Variance: " + str(trialist[t]["Het"])+ " &\nAnisotropy: " + str(trialist[t]["Anis"]),
                   rotation = "vertical", ha = "center", **titlekw)
    axe.set_xticks([])
    axe.set_yticks([])
    fig.add_subplot(axe)

    for r in Regimes:
        i = Trial.index(t)*len(Regimes) + Regimes.index(r) + Trial.index(t) + 1
        if i%4 != 0:
            inner = gridspec.GridSpecFromSubplotSpec(2, 2,
                                                     subplot_spec=outer[i], wspace=0.4, hspace=0.15)
            file = os.path.join(raw_directory, r + "AR_0_NS-A"+str(t)+"_df.npy")
            data = np.load(file)
            for g in gvarnames:
                axe = plt.Subplot(fig, inner[gvarnames.index(g)])
                sns.heatmap (data[species[g]["TecIndex"], -1, :, :], cmap = colorscheme, ax= axe)
                axe.set_title(g, ha = "center", **legendkw)
                axe.set_xticks([])
                axe.set_yticks([])
                fig.add_subplot(axe)
for a in range(4):
    plt.annotate(columntitles[a], xy=(0.15, 0.92), xytext=(0.0 + pad*a, 10),
                xycoords='figure fraction', textcoords='offset points',
                size='large', ha='center', va='baseline',
                **titlekw)
fig.show()
picname = "FigureS3_dissolved_species_heatmaps.png"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = "FigureS3_dissolved_species_heatmaps.pdf"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#Figure S4: Distribution of biomass heatmap
import matplotlib.gridspec as gridspec
Regimes = ["Slow", "Equal", "Fast"]
trialist = proc.masterscenarios()
Trial = ["50", "73", "63"]
species = proc.speciesdict("Saturated")
iaspecies = list(g for g in species if ((species[g]["State"]=="Active") and (species[g]["Location"] == "Immobile")))
gvarnames = list(g for g in iaspecies if (g != "Immobile active sulphate reducers"))
sptitles = ["Aerobic\ndegraders", "Nitrate\nreducers", "Ammonia\noxidizers"]
velindex = 2
colorscheme = 'YlGnBu'
columntitles = ["Velocity\ndistribution pattern", "Slow flow", "Medium flow", "Fast flow"]
fig = plt.figure(figsize=(24, 8))
outer = gridspec.GridSpec(3, 4, width_ratios = [0.2,1, 1, 1],wspace=0.15, hspace=0.3)
pad = 0.28
for t in Trial:
    file = os.path.join(raw_directory, "EqualAR_0_NS-A"+str(t)+"_df.npy")
    data = np.load(file)
    left = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outer[4*Trial.index(t)],wspace=0.3,hspace=0.1)
    axe=plt.Subplot(fig, left[0])
    velocity=abs(data[velindex,-1,:,:])
    sns.heatmap(velocity,cmap=colorscheme,ax=axe,cbar=False)
    axe.set_ylabel ("Variance: "+str(trialist[t]["Het"])+" &\nAnisotropy: "+str(trialist[t]["Anis"]),
                   rotation="vertical",**suptitlekw,ha="center")
    axe.set_xticks([])
    axe.set_yticks([])
    fig.add_subplot(axe)
    for r in Regimes:
        i=Trial.index(t)*len(Regimes)+Regimes.index(r)+Trial.index(t)+1
        if i%4!=0:
            inner = gridspec.GridSpecFromSubplotSpec(1,3,subplot_spec=outer[i],wspace=0.3,hspace=0.1)
            file = os.path.join(raw_directory, r+"AR_0_NS-A"+str(t)+"_df.npy")
            data = np.load(file)
            for g in gvarnames:
                axe=plt.Subplot(fig,inner[gvarnames.index(g)])
                sns.heatmap (data[species[g]["TecIndex"],-1,:,:],cmap=colorscheme,ax= axe)
                axe.set_title(sptitles[gvarnames.index(g)],**titlekw,ha="center")
                axe.set_xticks([])
                axe.set_yticks([])
                fig.add_subplot(axe)
for a in range(1,4,1):
    plt.annotate(columntitles[a],xy=(0.05, 0.05),xytext=(0.28 + (a-1)*pad, 0.95),
                xycoords='figure fraction',textcoords='figure fraction',
                size='large',ha='center',va='baseline',**titlekw)
plt.annotate(columntitles[0],xy=(0.1, 0.92),xytext=(0.1, 0.913),xycoords='figure fraction',textcoords='figure fraction',size='large',ha='center',va='baseline',**suptitlekw)
fig.show()
picname = "FigureS4_immobile_biomass_heatmaps.png"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = "FigureS4_immobile_biomass_heatmaps.pdf"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#Figure S5: DO profile 1D

import matplotlib as mpl

FlowRegimes = ["Slow", "Medium", "Fast"]

Redscmap1 = mpl.cm.Reds(np.linspace(0, 1, 30))
Greenscmap1 = mpl.cm.Greens(np.linspace(0, 1, 30))
Bluescmap1 = mpl.cm.Blues(np.linspace(0, 1, 30))
Redscmap = mpl.colors.ListedColormap(Redscmap1[10:, :-1])
Greenscmap = mpl.colors.ListedColormap(Greenscmap1[10:, :-1])
Bluescmap = mpl.colors.ListedColormap(Bluescmap1[10:, :-1])
colseries = ["Reds", "Greens", "Blues"]
Regimes = ["Slow", "Equal", "Fast"]
trialist = proc.masterscenarios()
Trial = ["H", "37", "68", "79", "52", "77", "84", "74", "63"]
species = proc.speciesdict("Saturated")
gvarnames = ["DO"]
cvars = list(species[g]['TecIndex'] for g in gvarnames)
figbig, axes = plt.subplots(1,3, figsize=(16, 5), sharey = True, sharex = True)
for r in Regimes:
    lines = []
    i = Regimes.index(r)
    host = axes.flat[i]
    host.set_ylabel ("Y (cm)", **legendkw)
    colors = sns.color_palette(palette = colseries[i], n_colors = len(Trial))
    for t in Trial:
        file = os.path.join(raw_directory, r+"AR_0_NS-A"+str(t)+"_df.npy")
        data = np.load(file)
        conctime, TotalFlow, Headinlettime = sta.conc_time (data,0,50,0,30, 51, gvarnames,"Saturated")
        p1, = host.plot(
                conctime[np.shape(conctime)[0] - 1, :, -1],
                yindex,
                label=str(trialist[t]["Het"]) + ":" + str(trialist[t]["Anis"]),
                color=colors[Trial.index(t)],
            )
        lines.append(p1)
    host.legend(lines, [l.get_label() for l in lines], **legendkw)
    if r == "Equal":
        host.set_title("Medium flow", **titlekw)
    else:
        axes.flat[Regimes.index(r)].set_title(r + " flow", **titlekw)

    host.set_xlim(0, 250)
    host.set_ylabel("Y (cm)", **titlekw)
    host.set_xlabel("DO ($\mu$M)", **titlekw)
    host.tick_params(axis="y", **labelkw)
    host.tick_params(axis="x", **labelkw)
    host.invert_yaxis()
plt.grid(False)
picname = "FigureS5_DOwithhet_1D.png"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = "FigureS5_DOwithhet_1D.pdf"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#Figure S6: Distribution of concentration derived Damkohler number

path_da_data = os.path.join(results_dir, "Da_BG.csv")
da = pd.read_csv(path_da_data)
da.columns
da.shape
gvarnames = ["DO", "Nitrogen", "TOC"]
finaldata = da[da['Chem'].isin (gvarnames)]
finaldata["logDa"] = np.log10(finaldata.Da)
#Show distribution of Da numbers and then ratio of Da and Pe

plt.title('Distribution of Damk$\ddot{o}$hler number', **titlekw)
sns.distplot(finaldata[da["Regime"] == "Slow"]["logDa"], color = "indianred", label = "Slow flow", kde = False, bins = 20)
sns.distplot(finaldata[da["Regime"] == "Medium"]["logDa"], color = "g", label = "Medium flow", kde = False, bins = 20)
sns.distplot(finaldata[da["Regime"] == "Fast"]["logDa"], color = "steelblue", label = "Fast flow", kde = False, bins = 20)
plt.tick_params(**labelkw)
plt.xlabel ("log$_{10}$Da", **titlekw)
plt.ylabel ("Number of scenarios", **titlekw)
plt.legend(**legendkw)
plt.grid(False)
picname = "FigureS6_Da_distribution.png"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = "FigureS6_Da_distribution.pdf"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#Figure S7: Impact on removal of reactive species- comparison of Dat and chemical-flow regime specific plots
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
grey_dot = mlines.Line2D([], [], linestyle = '', marker = "o", markerfacecolor = "grey", markeredgecolor = "grey", markersize=10, label='DO', alpha = 0.5)
grey_triangle = mlines.Line2D([], [], linestyle = '', marker = "^", markerfacecolor = "grey", markeredgecolor = "grey",markersize=10, label='Nitrogen', alpha = 0.5)
grey_square = mlines.Line2D([], [], linestyle = '', marker = "s", markerfacecolor = "grey", markeredgecolor = "grey",markersize=10, label='TOC', alpha = 0.5)
my_pal = {3:"indianred", 2: "g", 0:"steelblue", 1 :"orange"}
blue_patch = mpatches.Patch(color="steelblue", label= "Fast flow", alpha = 0.5)
green_patch = mpatches.Patch(color="g", label="Medium flow", alpha = 0.5)
red_patch = mpatches.Patch(color="indianred", label="Slow flow", alpha = 0.5)
patchlist = [blue_patch, green_patch, red_patch, grey_square, grey_dot, grey_triangle]

path_da_data = os.path.join(parent_dir, "Results", "Da_BG.csv")
da = pd.read_csv(path_da_data)#, sep = "\t")
da.columns
da.shape
gvarnames = ["DO", "Nitrogen", "TOC"]
regimes = ["Slow", "Medium", "Fast"]
finaldata = da[da['Chem'].isin (gvarnames)]
mymarklist = ["o", "^", "s", "d"]
pal = ["indianred", "g", "steelblue"]

for r in regimes:
    for g in gvarnames:
        data = finaldata[(finaldata["Chem"] == g) & (finaldata["Regime"] == r)]
        plt.scatter(data["fraction"]*100, 100*data['reldelmassflux_spatial_fraction'], c = pal[regimes.index(r)], marker = mymarklist[gvarnames.index(g)], alpha = 0.5, label = g)
plt.yscale("log")
plt.yticks((40,100,300,600), (40,100,300,600))
plt.tick_params(**labelkw)
plt.legend(handles = patchlist, **legendkw)
plt.xlabel ("Residence time of solutes (%)", **titlekw)
plt.ylabel("Impact on removal of\nreactive species (%)", **titlekw)
plt.subplots_adjust (top = 0.92)
plt.grid(False)
picname = "FigureS7_removal_impact.png"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
picname = "FigureS7_removal_impact.pdf"
plt.savefig(picname, dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#Check if distributions are significantly different:

#Figure S8_analytical distribution
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
orangeline = mlines.Line2D([], [], linestyle = 'solid', color='orange', markersize=15, label='First order: -0.5')
greenline = mlines.Line2D([], [], linestyle = 'solid', color='g', markersize=15, label='First order: 0.5')
greyline = mlines.Line2D([], [], linestyle = 'dashed', color='grey', markersize=15, label='Zero order')
blue_dot = mlines.Line2D([], [], linestyle = '', marker = 'o', color='steelblue', alpha = 0.5, markersize=15, label='Simulation result')
patchlist = [orangeline, greenline, greyline, blue_dot]

path_da_data = os.path.join(results_dir, "Da_BG.csv")
da = pd.read_csv(path_da_data)#, sep = "\t")
da.columns
da.shape
da['%fraction'] = da['fraction']*100
gvarnames = ["DO", "Nitrogen", "TOC"]

da["logDa"] = np.log10(da.Da)
da.loc[da["logDa"] < -1, "PeDamark"] = "0"
da.loc[(da["logDa"] > -1) & (da["logDa"] < 0), "PeDamark"] = "1"
da.loc[(da["logDa"] > 0) & (da["logDa"] <0.5), "PeDamark"] = "2"
da.loc[(da["logDa"] > 0.5), "PeDamark"] = "3"
labels = {"3" : 'log$_{10}$Da > 0.5',
          "2" : r'0 < log$_{10}$Da < 0.5',
          "1" : r'-1 < log$_{10}$Da < 0',
         "0" : r'log$_{10}$Da < -1'}

finaldata = da[da['Chem'].isin (gvarnames)]
mymarklist = ["o", "^", "s", "d"]

Datseries = [-0.5,0.5]
Daseries = list(10**x for x in Datseries)
damult = [1000, 100, 10]
fraction = np.arange(0.1, 1.1, step = 0.01)
styledict = {0: 'orange', 1 : 'g'}
lines = {0: '--', 1: 'solid', 2: ':'}

fig, ax = plt.subplots (nrows = 1, ncols = 2, figsize = (10,4), sharey = True, sharex = True)
idx = 0
for i in ["1", "2"]:
    axe = ax.flat[idx]
    subset = finaldata[finaldata["PeDamark"]==i]
    subset_sorted = subset.sort_values(by = ['fraction'])
    y = np.array(subset_sorted[['reldelmassflux_spatial_fraction']])*100
    datafrac = np.array(subset_sorted[["fraction"]])
    print(len(datafrac))
    axe.set_title (labels[i], **titlekw)
    axe.scatter(datafrac*100, y, alpha = 0.5, label = "Simulation result")
    dat = Datseries[idx]
    #for da in Daseries:
    da = Daseries[idx]
    k = da
    y_sol1 = []
    for tf in datafrac:
        cfrac = 100*(1 - np.exp (-da * tf))/(1 - np.exp(-da))
        y_sol1.append(cfrac)
    RMSE_sol = np.sqrt(((y - y_sol1)**2).mean())
    axe.plot(datafrac*100, y_sol1, c = styledict[Daseries.index(da)],
                label = "First order: "+ str(np.round(Datseries[Daseries.index(da)],2)))
    print (int(RMSE_sol))
    y_sol = []
    for tf in datafrac:
        cfrac = np.array(tf*100)
        y_sol.append(cfrac)
    RMSE_zero = np.sqrt(((y - y_sol)**2).mean())
    axe.plot(datafrac*100, y_sol, linestyle = '--', c = 'grey', label = "Zero order")
    print (int(RMSE_sol), int(RMSE_zero))
    axe.text(60,40, 'RMSE: ' + str(int(RMSE_zero)), {'color': 'grey', 'fontsize': 14})
    axe.text(20,100, 'RMSE: ' + str(int(RMSE_sol)), {'color': styledict[Daseries.index(da)], 'fontsize': 14})
    axe.set_ylim((0,120))
    idx += 1
ax.flat[0].set_ylabel("Nutrient removal (%)", **titlekw)
plt.annotate("Residence time of solutes (%)",xy=(-0.5, 0),xytext=(0, -30),xycoords="axes fraction",textcoords="offset points",
       size="large",ha="left",va="center",**titlekw)
plt.legend(handles = patchlist, title = r'log$_{10}$Da', **legendkw,
           bbox_to_anchor = (-0.1,-0.2), loc = 'upper center', ncol = 2)
plt.grid(False)
plt.savefig("Figure_S8_comparison_timeeffect.png", dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)
plt.savefig("Figure_S8_comparison_timeeffect.pdf", dpi = 300, bbox_inches = 'tight', pad_inches = 0.1)

#Figure S9: Microbial biomass fractions with spatial heterogeneity
#Load standard values required to navigate through the datasets
filename = "biomass_comparison_steadystate_BG.csv"
Regimes = ["Slow", "Medium", "Fast"]
vels = [0.00038,0.0038,0.038]
gw = 1

scdict = proc.masterscenarios() #master dictionary of all spatially heterogeneous scenarios that were run
ratenames = proc.masterrates("saturated")

#Domains
Trial = list(t for t,values in scdict.items())

#Reactive species of concerns
States = ["Active", "Inactive"]
Locations = ["Mobile", "Immobile"]
allspecies = proc.speciesdict("Saturated")
microbialspecies = list(t for t in allspecies.keys() if allspecies[t]["State"] in States)
biomass_data_path = os.path.join(results_dir, filename)
allbiomassdata = pd.read_csv(biomass_data_path)
print(allbiomassdata.columns)
allbiomassdata['Regime'] = allbiomassdata['Regime'].replace({'Equal':'Medium'})
uniquespecies = allbiomassdata.Chem.unique().tolist()
for s in uniquespecies:
    if s == "Total":
        allbiomassdata.loc[allbiomassdata.Chem == s, 'State'] = 'NA'
        allbiomassdata.loc[allbiomassdata.Chem == s, 'Location'] = 'NA'
    else:
        allbiomassdata.loc[allbiomassdata.Chem == s, 'State'] = allspecies[s]["State"]
        allbiomassdata.loc[allbiomassdata.Chem == s, 'Location'] = allspecies[s]['Location']
col_dit = {'Slow':'indianred', 'Medium':'g', 'Fast':'steelblue'}
uniquespecies.remove('Total')
uniquespecies.remove('Immobile active sulphate reducers')
uniquespecies.remove('Immobile inactive sulphate reducers')
uniquespecies.remove('Mobile active sulphate reducers')
uniquespecies.remove('Mobile inactive sulphate reducers')
print(uniquespecies)
x = allbiomassdata.groupby(['Regime','Trial','State', 'Location'], as_index=False)['Contribution'].sum()
x = pd.merge(x, allbiomassdata[['Regime', 'Trial', 'Time', 'fraction']], on = ['Regime', 'Trial'])
fig, axeses = plt.subplots(2,2, sharex = True, sharey = True, figsize = (8,8))
i = 0
for a in ["Active", "Inactive"]:
    for l in ["Immobile", "Mobile"]:
        subdata = x[(x.State == a) & (x.Location == l)]
        ax = axeses.flat[i]
        ax.set_title(a + " " + l + " biomass", **titlekw)
        for r in ["Slow", "Medium", "Fast"]:
            data = subdata[subdata.Regime == r]
            ax.scatter(100*data.fraction, 100*data.Contribution, c = col_dit[r], label = r + " flow")
        i += 1
ax.legend(**legendkw)
axeses.flat[2].set_xlabel ("Residence time of solutes (%)", **titlekw)
axeses.flat[3].set_xlabel ("Residence time of solutes (%)", **titlekw)
axeses.flat[0].set_ylabel ("Contribution to total biomass (%)", **titlekw)
axeses.flat[2].set_ylabel ("Contribution to total biomass (%)", **titlekw)
plt.grid(False)
plt.savefig("FigureS9_contribution_state_location_biomass.png", dpi = 300)
plt.savefig("FigureS9_contribution_state_location_biomass.pdf")