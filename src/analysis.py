import yaml
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr
import matplotlib.ticker as ticker
from scipy.optimize import curve_fit
import pandas
import os
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
import sys
from matplotlib.colors import LogNorm
from scipy.optimize import least_squares
import mdtraj as md
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.set_printoptions(suppress=True)  # Cancel scientific counting display
np.set_printoptions(threshold=np.inf)
import matplotlib as mpl

def calcRg(cwd,dataset, df,record,cycle):
    t = md.load_dcd(f"{cwd}/{dataset}/{record}/{cycle}/{record}.dcd",f"{cwd}/{dataset}/{record}/{cycle}/{record}.pdb")  # nm
    residues = [res.name for res in t.top.atoms]
    masses = df.loc[residues,'MW'].values
    masses[0] += 2
    masses[-1] += 16
    # calculate the center of mass
    cm = np.sum(t.xyz*masses[np.newaxis,:,np.newaxis],axis=1)/masses.sum()
    # calculate residue-cm distances
    si = np.linalg.norm(t.xyz - cm[:,np.newaxis,:],axis=2)
    # calculate rg
    rgarray = np.sqrt(np.sum(si**2*masses,axis=1)/masses.sum())
    np.save(f"{cwd}/{dataset}/{record}/{cycle}/Rg_traj.npy", rgarray)
    rg = rgarray.mean()
    return rgarray, rg
def calcRs(traj):
    pairs = traj.top.select_pairs('all','all')
    d = md.compute_distances(traj,pairs)
    nres = traj.n_atoms
    ij = np.arange(2,nres,1)
    diff = [x[1]-x[0] for x in pairs]
    dij = np.empty(0)
    for i in ij:
        dij = np.append(dij, np.sqrt((d[:,diff==i]**2).mean().mean()))  # mean()? giulio
    return ij,dij,np.mean(1/d,axis=1)

def correlationVaryingsigmaandlambda():
    cwd = "/projects/prism/people/ckv176/stickiness/src"
    latex_residues = pd.read_pickle(f"{cwd}/latex_residues.pkl")
    ave_sigma = 0.5609338492721576
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)
    latex_residues["sigma_diff"] = ave_sigma-latex_residues["sigma_ori"]
    latex_residues["lambda_diff"] = latex_residues["lambda_ave"]-latex_residues["lambda_ori"]
    ax1.scatter(latex_residues["sigma_diff"], latex_residues["lambda_diff"])
    ax1.hlines(0, xmin=-0.15, xmax=0.15, colors="red")
    ax1.vlines(0, ymin=-0.1, ymax=0.1, colors="red")
    ax1.set_xlabel("$\\langle\\rm{σ}\\rangle-\\rm{σ}_{AA}$")
    ax1.set_ylabel("$λ(\\langle\\rm{σ}\\rangle)-λ(\\rm{σ}_{AA})$")
    plt.tight_layout()
    plt.show()
    latex_residues.to_csv("latex_residues_plot.csv")
    print("\n", latex_residues)

def plotAminoAcidFreq():
    cwd = "/projects/prism/people/ckv176/stickiness/src"
    dataset = "IDPsCAC3_based_2.2_0.08_1"
    allproteins = pd.read_pickle(f'{cwd}/{dataset}/allproteins.pkl')
    residues = pd.read_csv(f'{cwd}/residues_pub.csv').set_index('one')
    pool = []
    for record in allproteins.index:
        pool += allproteins.loc[record, "fasta"]
    sum = 0
    for oneletter in residues.index:
        freq = ''.join(pool).count(oneletter)/len(pool)
        sum += freq
        print(f"{oneletter}: {freq}")
    print(f"sum: {sum}")

def generate_PROvsNu():
    cwd = "/projects/prism/people/ckv176/stickiness/src"
    dataset = "IDPsCAC2_based_2.4_0.03_0.5_1"
    cycle = 8
    proteinsRgs = pd.read_pickle(f"{cwd}/{dataset}/proteinsRgs.pkl")
    proteinsRgs_Nu = proteinsRgs.copy().loc[proteinsRgs.index]
    nu_values, nu_errors, proline_percent = [], [], []

    f = lambda x, R0, v: R0 * np.power(x, v)
    for record in proteinsRgs.index:
        print(record)
        if os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/dij.npy') and os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/ij.npy'):
            dij = np.load(f'{cwd}/{dataset}/{record}/{cycle}/dij.npy')
            ij = np.load(f'{cwd}/{dataset}/{record}/{cycle}/ij.npy')
        else:
            traj = md.load_dcd(f"{cwd}/{dataset}/{record}/{cycle}/{record}.dcd",
                               f"{cwd}/{dataset}/{record}/{cycle}/{record}.pdb")  # nm
            ij, dij, invrij = calcRs(traj)
            np.save(f'{cwd}/{dataset}/{record}/{cycle}/dij.npy', dij)
            np.save(f'{cwd}/{dataset}/{record}/{cycle}/ij.npy', ij)
        popt, pcov = curve_fit(f, ij[ij > 5], dij[ij > 5], p0=[.4, .5])
        # scaling exponent
        nu_values.append(popt[1])
        nu_errors.append(pcov[1, 1] ** 0.5)  # std
        proline_percent.append(proteinsRgs_Nu.loc[record, "fasta"].count("P") / len(proteinsRgs_Nu.loc[record, "fasta"]))
    proteinsRgs_Nu["nu_values"] = nu_values
    proteinsRgs_Nu["nu_errors"] = nu_errors
    proteinsRgs_Nu["proline_percent"] = proline_percent
    proteinsRgs_Nu.to_pickle(f"{cwd}/{dataset}/proteinsRgs_Nu.pkl")

def plot_PROvsNu():
    cwd = "/projects/prism/people/ckv176/stickiness/src"
    dataset = "IDPsCAC2_based_2.4_0.03_0.5_1"
    proteinsRgs_Nu = pd.read_pickle(f"{cwd}/{dataset}/proteinsRgs_Nu.pkl")
    fig = plt.figure(figsize=(5, 5), dpi=300)
    ax1 = plt.subplot2grid((1, 1), (0, 0), rowspan=1, colspan=1)

    ax1.scatter(proteinsRgs_Nu["proline_percent"], proteinsRgs_Nu["nu_values"])

    ax1.set_xlabel("proline_percent")
    ax1.set_ylabel("nu_values")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # plotAminoAcidFreq()
    # correlationVaryingsigmaandlambda()
    # generate_PROvsNu()
    plot_PROvsNu()
    pass