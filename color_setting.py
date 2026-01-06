cyan = "#33BBEE"
blue = "#0077BB"
orange = "#EE7733"
yellow = "#f0e442"
magenta = "#EE3377"
green = "#009988"
red = "#CC3311"
grey = "#BBBBBB"
CALVADOS2_CA = "$\\rm{CALVADOS2}_{C\\mathit{\\alpha}}$"
CALVADOS2_COM = "$\\rm{CALVADOS2}_{COM}$"
CALVADOS2_SCCOM = "$\\rm{CALVADOS2}_{SCCOM}$"
CA = '$\\rm{C}\\mathit{\\alpha}$'
CALVADOS3_COM = "$\\rm{CALVADOS3}_{COM}$"
import matplotlib.gridspec as gridspec
import pandas as pd
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

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
np.set_printoptions(suppress=True)  # Cancel scientific counting display
np.set_printoptions(threshold=np.inf)
import matplotlib as mpl
import random
import numpy as np

def get_predictions(cwd, dataset, cycle, validate, PRE_seq):
    multidomain_names = []  # no MDPs in this project
    allproteins = pd.read_pickle(f"{cwd}/{dataset}/allproteins.pkl")
    if validate:
        multidomain_names_validate = list(pd.read_pickle(f"{cwd}/{dataset}/MultiDomainsRgs_validate.pkl").index)
        multidomain_names = np.setdiff1d(multidomain_names_validate, multidomain_names).tolist()
        allproteins_validate = pd.read_pickle(f"{cwd}/{dataset}/allproteins_validate.pkl")
        allproteins = allproteins_validate.loc[np.setdiff1d(list(allproteins_validate.index), list(allproteins.index))]
    allproteins = allproteins.loc[np.setdiff1d(list(allproteins.index), PRE_seq)]  # might exclude aSyn, but because aSyn has different names in both dataframe, it still can be included anyway.
    IDP_names = list(np.setdiff1d(list(allproteins.index), multidomain_names))
    cal = []
    # bsheet_per = []

    for name in allproteins.index:
        if name in ['FP_Ash1_FP', 'FP_E1A_FP', 'FP_F1_GS16_FP', 'FP_F1_GS30_FP', 'FP_F1_GS8_FP', 'FP_F3_GS16_FP', 'FP_F3_GS30_FP',
            'FP_F3_GS8_FP', 'FP_FRQ_N_CRA_WF81AA_FP', 'FP_FRQ_N_CRA_native_FP', 'FP_FRQ_N_CRA_phos_full_FP', 'FP_FRQ_N_DIS_FP',
            'FP_FRQ_N_TET_FP', 'FP_FRQ_T_REE_FP', 'FP_FUS_FP', 'FP_PUMA_S1_FP', 'FP_PUMA_S2_FP', 'FP_PUMA_S3_FP', 'FP_PUMA_WT_FP', 'FP_p53_FP',
        'SNAP_FUS_PLDY2F_RBDR2K', 'SNAP_FUS_PLDY2F', 'SNAP_FUS', 'GFP_FUS', 'FUS_PLDY2F_RBDR2K', 'FL_FUS']:  # , 'hSUMO_hnRNPA1S', 'hnRNPA1S'
            if name in multidomain_names:
                multidomain_names.remove(name)
            else:
                IDP_names.remove(name)
    allproteins = allproteins.loc[IDP_names+multidomain_names]
    for name in allproteins.index:
        if not os.path.isfile(f"{cwd}/{dataset}/{name}/{cycle + 1}/Rg_traj.npy"):
            df = pd.read_csv(f'{cwd}/{dataset}/residues_{cycle}.csv').set_index('three')
            t = md.load_dcd(f"{cwd}/{dataset}/{name}/{cycle + 1}/{name}.dcd",
                            f"{cwd}/{dataset}/{name}/{cycle + 1}/{name}.pdb")
            residues = [res.name for res in t.top.atoms]
            masses = df.loc[residues, 'MW'].values
            masses[0] += 2
            masses[-1] += 16
            # calculate the center of mass
            cm = np.sum(t.xyz * masses[np.newaxis, :, np.newaxis], axis=1) / masses.sum()
            # calculate residue-cm distances
            si = np.linalg.norm(t.xyz - cm[:, np.newaxis, :], axis=2)
            # calculate rg
            rgarray = np.sqrt(np.sum(si ** 2 * masses, axis=1) / masses.sum())
            np.save(f"{cwd}/{dataset}/{name}/{cycle + 1}/Rg_traj.npy", rgarray)

        else:
            rgarray = np.load(f"{cwd}/{dataset}/{name}/{cycle + 1}/Rg_traj.npy")
        cal_value = np.mean(rgarray)

        cal.append(cal_value)
        """if name in multidomain_names:
            ssdomains = get_ssdomains(name, f"{cwd}/domains.yaml")
            len_domain = 0
            for ssdomain in ssdomains:
                len_domain += len(ssdomain)
            bsheet_per.append(np.load(f"{cwd}/relax_stride/{name}_SScount.npy", allow_pickle=True).item()["E"]/len_domain)"""

    allproteins["cal"] = cal
    allproteins['N'] = allproteins['fasta'].apply(lambda x: len(x))
    allproteins = allproteins.sort_values('N', ascending=False)
    allproteins.to_pickle(f"{cwd}/{dataset}/predictions.pkl")

    return allproteins, IDP_names, multidomain_names

def bootstrap_pearsonr(sim, exp, times=1000):
    indices = np.arange(sim.size)
    bootstrap_pr = []
    for _ in range(times):
        picked = random.choices(indices, k=indices.size)
        bootstrap_pr.append(pearsonr(exp[picked], sim[picked])[0])
    pr_err = np.std(bootstrap_pr)
    return pr_err


def bootstrap(sim, times=1000):
    indices = np.arange(sim.size)
    res_list = []
    for _ in range(times):
        picked = random.choices(indices, k=indices.size)
        res_list.append(sim[picked].mean())
    res_err = np.std(res_list)
    return res_err

def determineSigniDigit(value, error):
    print(f"{value}\u00B1{error}")
    # not perfect
    real_decimal_list = str(error).split(".")
    real_int = int(real_decimal_list[0])
    if real_int != 0:
        exponent = len(str(real_int))-1
        if exponent==0:
            Digit = 0
            scd = False
        else:
            Digit = exponent
            scd = True
    else:
        decimal_str = real_decimal_list[1]
        Digit = 1
        scd = False  # scientific counting display
        for i in decimal_str:
            if i=="0":
                Digit += 1
            else:
                break
    # print(Digit, scd)
    if scd:
        value = value/(10**Digit)
        error = error/(10**Digit)
        first_part = int(np.round(value+1E-8, 0))  # +1E-8 used for rounding
        second_part = int(np.round(error+1E-8, 0))
        science = "$\\times{10}^%d$" % (Digit)
        output = f"({first_part}\u00B1{second_part}){science}"
    else:
        if Digit==0:
            first_part = int(np.round(value+1E-8, Digit))
            second_part = int(np.round(error+1E-8, Digit))
            output = f"{first_part}\u00B1{second_part}"
        else:
            first_part = np.round(value, Digit)
            tmp_size = len(str(first_part).split(".")[1])
            second_part = np.round(error, Digit)

            output = f"{first_part}{''.join(['0']*(Digit-tmp_size))}\u00B1{second_part}"

    print(output)
    return output
