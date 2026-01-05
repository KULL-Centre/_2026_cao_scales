########################################################################################################################
#                                   import modules                                                                     #
########################################################################################################################
from utils import *
import subprocess
from rawdata import *
import os
import MDAnalysis
from argparse import ArgumentParser
pd.set_option('display.max_columns', None)
parser = ArgumentParser()
parser.add_argument('--batch_sys', nargs='?', type=str)
parser.add_argument('--CoarseGrained', nargs='?', type=str)
parser.add_argument('--gpu_str', nargs='?', type=str)
parser.add_argument('--theta', nargs='?', type=float)
parser.add_argument('--cycle', nargs='?', type=str)
parser.add_argument('--dataset_replica', nargs='?', type=int)
parser.add_argument('--cutoff', nargs='?', type=float)
parser.add_argument('--validate_indicator', nargs='?', type=str)
parser.add_argument('--dataset_prefix', nargs='?', type=str)
parser.add_argument('--max_cycle', nargs='?', type=int)
args = parser.parse_args()
########################################################################################################################
#                                    general simulation details                                                        #
########################################################################################################################
cwd_dict = {"Computerome": "/home/people/fancao/stickiness/src", "ROBUST": "/home/fancao/stickiness/src", "Deic": "/groups/sbinlab/fancao/stickiness/src"}
use_newdata_dict = {"IDPsCA": True,
                    "IDPsWeiAveSigmaCA": True,
                    "IDPsallow_negativeCA": True,
                    "IDPsWeiAveSigmaallow_negativeCA": True,
                    "IDPsfractPriorCA": True,
                    "IDPsWeiAveSigmafractPriorCA": True,
                    "CALVADOS2CA": True,
                    "CALVADOS2WeiAveSigmaCA": True,
                    "IDPsresetErrCA": True,
                    "IDPsresetErrNominmaxCA": True,
                    "IDPsresetErrC2_trainingCA": False,
                    "IDPsresetErrC2_trainingNominmaxCA": False,
                    "IDPsresetErrC2_priorCA": True,
                    "IDPsresetErrC2_priorfractPriorCA": True,
                    "IDPsresetErrC2_priorminmaxCA": True,
                    "IDPsresetErrC2_priorNominmaxCA": True,
                    "IDPsWeiAveresetErrC2_priorNominmaxCA": True,
                    }
normal_dict = {"IDPsCA": True,
               "IDPsWeiAveSigmaCA": True,
               "IDPsallow_negativeCA": True,
               "IDPsWeiAveSigmaallow_negativeCA": True,
               "IDPsfractPriorCA": True,
               "IDPsWeiAveSigmafractPriorCA": True,
               "CALVADOS2CA": True,
               "CALVADOS2WeiAveSigmaCA": True,
               "IDPsresetErrCA": True,
               "IDPsresetErrNominmaxCA": True,
               "IDPsresetErrC2_trainingCA": True,
               "IDPsresetErrC2_trainingNominmaxCA": True,
               "IDPsresetErrC2_priorCA": True,
               "IDPsresetErrC2_priorfractPriorCA": True,
               "IDPsresetErrC2_priorminmaxCA": True,
               "IDPsresetErrC2_priorNominmaxCA": True,
               "IDPsWeiAveresetErrC2_priorNominmaxCA": True,
               }
include_PREloss_dict = {"IDPsCA": False,
                        "IDPsWeiAveSigmaCA": False,
                        "IDPsallow_negativeCA": False,
                        "IDPsWeiAveSigmaallow_negativeCA": False,
                        "IDPsfractPriorCA": False,
                        "IDPsWeiAveSigmafractPriorCA": False,
                        "CALVADOS2CA": False,
                        "CALVADOS2WeiAveSigmaCA": False,
                        "IDPsresetErrCA": False,
                        "IDPsresetErrNominmaxCA": False,
                        "IDPsresetErrC2_trainingCA": False,
                        "IDPsresetErrC2_trainingNominmaxCA": False,
                        "IDPsresetErrC2_priorCA": False,
                        "IDPsresetErrC2_priorfractPriorCA": False,
                        "IDPsresetErrC2_priorminmaxCA": False,
                        "IDPsresetErrC2_priorNominmaxCA": False,
                        "IDPsWeiAveresetErrC2_priorNominmaxCA": False,
                        }
initial_types_dict = {"IDPsCA": "C2_based",
                      "IDPsWeiAveSigmaCA": "C2_based",
                      "IDPsallow_negativeCA": "C2_based",
                      "IDPsWeiAveSigmaallow_negativeCA": "C2_based",
                      "IDPsfractPriorCA": "C2_based",
                      "IDPsWeiAveSigmafractPriorCA": "C2_based",
                      "CALVADOS2CA": "C2",
                      "CALVADOS2WeiAveSigmaCA": "C2",
                      "IDPsresetErrCA": "C2_based",
                      "IDPsresetErrNominmaxCA": "C2_based",
                      "IDPsresetErrC2_trainingCA": "C2_based",
                      "IDPsresetErrC2_trainingNominmaxCA": "C2_based",
                      "IDPsresetErrC2_priorCA": "C2_based",
                      "IDPsresetErrC2_priorfractPriorCA": "C2_based",
                      "IDPsresetErrC2_priorminmaxCA": "C2_based",
                      "IDPsresetErrC2_priorNominmaxCA": "C2_based",
                      "IDPsWeiAveresetErrC2_priorNominmaxCA": "C2_based",
                      }
wei_sigma_dict = {"IDPsCA": False,
                  "IDPsWeiAveSigmaCA": True,
                  "IDPsallow_negativeCA": False,
                  "IDPsWeiAveSigmaallow_negativeCA": True,
                  "IDPsfractPriorCA": False,
                  "IDPsWeiAveSigmafractPriorCA": True,
                  "CALVADOS2CA": False,
                  "CALVADOS2WeiAveSigmaCA": True,
                  "IDPsresetErrCA": False,
                  "IDPsresetErrNominmaxCA": False,
                  "IDPsresetErrC2_trainingCA": False,
                  "IDPsresetErrC2_trainingNominmaxCA": False,
                  "IDPsresetErrC2_priorCA": False,
                  "IDPsresetErrC2_priorfractPriorCA": False,
                  "IDPsresetErrC2_priorminmaxCA": False,
                  "IDPsresetErrC2_priorNominmaxCA": False,
                  "IDPsWeiAveresetErrC2_priorNominmaxCA": True,
                  }
giulio_dict = {"IDPsCA": True,
               "IDPsWeiAveSigmaCA": True,
               "IDPsallow_negativeCA": True,
               "IDPsWeiAveSigmaallow_negativeCA": True,
               "IDPsfractPriorCA": True,
               "IDPsWeiAveSigmafractPriorCA": True,
               "CALVADOS2CA": True,
               "CALVADOS2WeiAveSigmaCA": True,
               "IDPsresetErrCA": True,
               "IDPsresetErrNominmaxCA": True,
               "IDPsresetErrC2_trainingCA": False,
               "IDPsresetErrC2_trainingNominmaxCA": False,
               "IDPsresetErrC2_priorCA": True,
               "IDPsresetErrC2_priorfractPriorCA": True,
               "IDPsresetErrC2_priorminmaxCA": True,
               "IDPsresetErrC2_priorNominmaxCA": True,
               "IDPsWeiAveresetErrC2_priorNominmaxCA": True,
               }
allow_negative_dict = {
   "IDPsCA": False,
   "IDPsWeiAveSigmaCA": False,
   "IDPsallow_negativeCA": True,
   "IDPsWeiAveSigmaallow_negativeCA": True,
   "IDPsfractPriorCA": False,
   "IDPsWeiAveSigmafractPriorCA": False,
   "CALVADOS2CA": False,
   "CALVADOS2WeiAveSigmaCA": False,
   "IDPsresetErrCA": False,
    "IDPsresetErrNominmaxCA": False,
    "IDPsresetErrC2_trainingCA": False,
    "IDPsresetErrC2_trainingNominmaxCA": False,
    "IDPsresetErrC2_priorCA": False,
    "IDPsresetErrC2_priorfractPriorCA": False,
    "IDPsresetErrC2_priorminmaxCA": False,
    "IDPsresetErrC2_priorNominmaxCA": False,
    "IDPsWeiAveresetErrC2_priorNominmaxCA": False,
}

fractPrior_dict = {
   "IDPsfractPriorCA": True,
   "IDPsWeiAveSigmafractPriorCA": True,
   "IDPsallow_negativeCA": False,
   "IDPsCA": False,
   "IDPsWeiAveSigmaCA": False,
   "CALVADOS2CA": False,
   "CALVADOS2WeiAveSigmaCA": False,
   "IDPsresetErrCA": False,
    "IDPsresetErrNominmaxCA": False,
    "IDPsresetErrC2_trainingCA": False,
    "IDPsresetErrC2_trainingNominmaxCA": False,
    "IDPsresetErrC2_priorCA": False,
    "IDPsresetErrC2_priorfractPriorCA": True,
    "IDPsresetErrC2_priorminmaxCA": False,
    "IDPsresetErrC2_priorNominmaxCA": False,
    "IDPsWeiAveresetErrC2_priorNominmaxCA": False,
}

resetErr_dict = {
    "IDPsresetErrCA": True,
    "IDPsresetErrNominmaxCA": True,
    "IDPsresetErrC2_trainingCA": True,
    "IDPsresetErrC2_trainingNominmaxCA": True,
    "IDPsresetErrC2_priorCA": True,
    "IDPsresetErrC2_priorfractPriorCA": True,
    "IDPsresetErrC2_priorminmaxCA": True,
    "IDPsresetErrC2_priorNominmaxCA": True,
    "IDPsCA": False,
    "CALVADOS2CA": False,
    "IDPsWeiAveresetErrC2_priorNominmaxCA": True,
}

minmax_TF_dict = {
    "IDPsresetErrCA": True,
    "IDPsresetErrC2_trainingCA": True,
    "IDPsresetErrC2_trainingNominmaxCA": False,
    "IDPsresetErrNominmaxCA": False,
    "IDPsresetErrC2_priorCA": False,
    "IDPsresetErrC2_priorfractPriorCA": False,
    "IDPsresetErrC2_priorminmaxCA": True,
    "IDPsresetErrC2_priorNominmaxCA": False,
    "CALVADOS2CA": False,
    "IDPsWeiAveresetErrC2_priorNominmaxCA": False,
}

C2_prior_dict = {
    "IDPsresetErrC2_priorCA": True,
    "IDPsresetErrC2_priorfractPriorCA": True,
    "IDPsresetErrC2_priorminmaxCA": True,
    "IDPsresetErrC2_priorNominmaxCA": True,
    "CALVADOS2CA": False,
    "IDPsWeiAveresetErrC2_priorNominmaxCA": True
}

pathToconvert_cg2all_dict ={"Deic": "/groups/sbinlab/fancao/miniconda3/envs/Calvados_MDP/bin/convert_cg2all",
            "ROBUST": "/home/fancao/miniconda3/envs/Calvados_MDP/bin/convert_cg2all"}
path2gmx_dict = {"Deic": "/groups/sbinlab/fancao/gmx_20202/bin/gmx",
            "ROBUST": "/home/fancao/gmx_20225/bin/gmx"}

########################################################################################################################
#          you are only allowed to change this block during other optimizations are running                            #
########################################################################################################################
validate_indicator = 'off' if args.validate_indicator == None else args.validate_indicator
if validate_indicator == "on":
    validate = True
elif validate_indicator == "off":
    validate = False
else:
    raise Exception("What do you think you are doing with validate?")
batch_sys = 'ROBUST' if args.batch_sys == None else args.batch_sys  # schedule system, Deic, Computerome, ROBUST
cwd = cwd_dict[batch_sys]  # current working directory
CoarseGrained = "CA" if args.CoarseGrained == None else args.CoarseGrained  # CoarseGrained strategy, COM means only residues of domains have COM representation, IDRs still have CA representation;
gpu_str = "on" if args.gpu_str == None else args.gpu_str
theta = float(1950 if args.theta == None else args.theta)  # prior default value: 0.05
cycles = list(range(1)) if args.cycle == None else [int(c) for c in args.cycle.split(",")]  # range(6, 11)
# cycles = list(range(6, 11)) if args.cycle == None else [int(c) for c in args.cycle.split(",")]  # range(6, 11)
dataset_replica = int(2 if args.dataset_replica == None else args.dataset_replica)
cutoff = float(2.4 if args.cutoff == None else args.cutoff)  # cutoff for the nonionic interactions, nm
# CALVADOS2WeiAveSigma, CALVADOS2, IDPsWeiAveSigma, IDPs, IDPsfractPrior, IDPsWeiAveSigmafractPrior, IDPsWeiAveSigmaallow_negative, IDPsallow_negative,
dataset_prefix = 'IDPsWeiAveresetErrC2_priorNominmax' if args.dataset_prefix == None else args.dataset_prefix  # IDPsresetErrC2_priorNominmax, IDPsWeiAveresetErrC2_priorNominmax
max_cycle = int(max(cycles) if args.max_cycle == None else args.max_cycle)

# don't forget to change final dataset
partition = "sbinlab_ib2"  # opt on deic, sbinlab_ib2 or sbinlab
dataset = f"{dataset_prefix}{CoarseGrained}"
cpu_max = 20  # 40 for Computerome thinnode, 20 for Deic and ROBUST(GPU)
if gpu_str == "on":
    gpu = True
elif gpu_str == "off":
    gpu = False
else:
    raise Exception("What do you think you are doing with gpu?")
if gpu and (batch_sys != "ROBUST"):
    raise Exception(f"Are you using {batch_sys} to submit gpu jobs?")
if gpu and cpu_max != 20:
    raise Exception(f"Please make sure 20 is used for 'cpu_max'")
path2giulio = f"{cwd}/rg_filtered_data_Giulio.csv"
multidomain_names = []
k_restraint = 700  # unit:KJ/(mol*nm^2); prior default value: 700, not used in this project
rebalancechi2_rg = False
lambda_oneMax = False
replicas = 20  # nums of replica for each sequence
xi_0 = .5  # increase from .1 to 1.
eps_factor = 0.2
gpu_id = 0
effective_frac = 0.4
resetErr_ratio = 0.01
discard_first_nframes = 10  # the first ${discard_first_nframes} will be discarded when merging replicas
runtime = 20  # hours (overwrites steps if runtime is > 0)
nframes = 200  # total number of frames to keep for each replica (exclude discarded frames)
########################################################################################################################
#                              multidomain simulation details                                                    #
########################################################################################################################
# kb = 8.31451E-3  # unit:KJ/(mol*K);
slab = False  # slab simulation parameters
# protein and parameter list
Usecheckpoint = False
########################################################################################################################
#                                             submit simulations                                                       #
########################################################################################################################
validate_str = "_validate" if validate else ""
allow_negative = allow_negative_dict[dataset]
wei_sigma = wei_sigma_dict[dataset]
use_newdata = use_newdata_dict[dataset]
normal = normal_dict[dataset]
path2gmx = path2gmx_dict[batch_sys]
pathToconvert_cg2all = pathToconvert_cg2all_dict[batch_sys]
fractPrior = fractPrior_dict[dataset]
include_PREloss = include_PREloss_dict[dataset]
initial_type = initial_types_dict[dataset]
giulio = giulio_dict[dataset]
minmax_TF = minmax_TF_dict[dataset]
C2_prior = C2_prior_dict[dataset]
resetErr = resetErr_dict[dataset]
dataset = f"{dataset}{initial_type}_{cutoff}_{theta}_{xi_0}_{dataset_replica}{validate_str}"
validateRg_pro = ["dChMINUS_Rg", "sNhPLUS_Rg", 'ERNTD_S118D', 'ERNTD_S118A', 'ERNTD_WT', 'Eralpha_NTD', 'HMPVP_NTD', 'SMAD4_linker', 'SMAD2_linker', 'Nup153_82', 'NLS_Rg', 'IBB_Rg', 'NUL_Rg', 'NUS_Rg', 'Nsp1_Rg', 'Nup49_Rg', 'N98_Rg']
validateSAXS_pro = ["V2", "V3", "V4", "V5", "ER_NTD", "CTR_XRCC4", "Nlp441_543", "CTir", "TRPV4", "Red1"]
validatePRE_pro = ["aSyn", "OPN", "FUS", "FUS12E"]
more_fret = pd.read_pickle(f'{cwd}/proteins_fret.pkl')
more_fret["fasta"] = more_fret['fasta'].apply(lambda x: list(x))
validateFRET_pro = ["NLS", "NUS", "IBB", "NUL", "Nup49", "Sic1"] + list(more_fret.index)
validate_pro = validateRg_pro + validateSAXS_pro + validatePRE_pro + validateFRET_pro
validate_pro = list(set(validate_pro))
if validate:
    cutoff = 2.4  # 2.0 nm for validation

fdomains = f'{cwd}/{dataset}/domains.yaml'
if not os.path.isdir(f"{cwd}/{dataset}"):
    os.system(f"mkdir -p {cwd}/{dataset}")

proteinsPRE = initProteinsPRE(normal=normal, include_PREloss=include_PREloss)
proteinsRgs = initIDPsRgs(normal=normal, use_newdata=use_newdata, validate=validate, giulio=giulio, path2giulio=path2giulio)
if resetErr:
    print("Resetting Error....")
    proteinsRgs = resetexpRgErr(proteinsRgs, resetErr_ratio)
allproteins = pd.concat((proteinsPRE, proteinsRgs), sort=True)
proteinsPRE.to_pickle(f'{cwd}/{dataset}/proteinsPRE.pkl')
proteinsRgs.to_pickle(f'{cwd}/{dataset}/proteinsRgs.pkl')
allproteins.to_pickle(f'{cwd}/{dataset}/allproteins.pkl')

for cycle in cycles:
    if not validate:
        if cycle > max_cycle:  # for gpu
            break
        if gpu and cycle != cycles[0]:
            break
    os.system(f"cp {cwd}/domains.yaml {fdomains}")

    if cycle == 0:
        if initial_type in ["C1", "C2", "C3"]:
            os.system(f"cp {cwd}/residues_pub.csv {cwd}/{dataset}")  # specify lambda initial values
        if initial_type in ["0.5", "Ran", "C2_based", "C3_based"]:
            os.system(f"cp {cwd}/residues_-1.csv {cwd}/{dataset}")  # use 0.5 or random or C3
        create_parameters(cwd, dataset, cycle, initial_type, wei_sigma=wei_sigma)
    if validate:
        if initial_type in ["C1", "C2", "C3"]:
            os.system(f"cp {cwd}/residues_pub.csv {cwd}/{dataset}")  # specify lambda initial values
        if initial_type in ["0.5", "Ran", "C2_based", "C3_based"]:
            os.system(f"cp {cwd}/{dataset[:-len(validate_str)]}/residues_{cycle-1}.csv {cwd}/{dataset}")  # use 0.5 or random

        proteinsPRE = initProteinsPRE(normal=normal, include_PREloss=True)
        proteinsRgs = initIDPsRgs(normal=normal, use_newdata=use_newdata, validate=True, giulio=giulio,
                                  path2giulio=path2giulio, francesco=True)
        if resetErr:
            print("Resetting Error....")
            proteinsRgs = resetexpRgErr(proteinsRgs, resetErr_ratio)
        proteinsFRET = initIDPsFRET(validate=True)
        proteinsFRET = pd.concat((proteinsFRET, more_fret))
        allproteins = pd.concat((proteinsPRE, proteinsRgs, proteinsFRET), sort=True)
        proteinsPRE.to_pickle(f'{cwd}/{dataset}/proteinsPRE_validate.pkl')
        proteinsRgs.to_pickle(f'{cwd}/{dataset}/proteinsRgs_validate.pkl')
        proteinsFRET.to_pickle(f'{cwd}/{dataset}/proteinsFRET_validate.pkl')
        allproteins.to_pickle(f'{cwd}/{dataset}/allproteins_validate.pkl')

    allproteins['N'] = allproteins['fasta'].apply(lambda x: len(x))
    N_res = []
    for record in allproteins.index:
        name = record.split("@")[0]
        if record in multidomain_names:
            domain_len = 0
            for domain in get_ssdomains(name, fdomains, output=False):
                domain_len += len(domain)
            N_res.append(allproteins.loc[record].N - domain_len)
        else:
            N_res.append(allproteins.loc[record].N)

    allproteins["N_res"] = N_res
    # allproteins.to_pickle(f'{cwd}/{dataset}/test.pkl')
    # print(allproteins.N_res)
    allproteins = allproteins.sort_values('N_res', ascending=False)
    # simulate
    jobid_2 = []
    if gpu:
        command_lines_dict = defaultdict(list)
        num_proteins = len(allproteins.index)
    pro_idx = 0
    parallel_gpus = len(list(allproteins.index))  #
    for record, prot in allproteins.iterrows():
        if validate:
            if record not in validate_pro:
                continue
        config_merge_filename = f'config_merge.yaml'
        jobid_1 = []
        name = record.split("@")[0]
        do_merge = False
        name_dcd_readable = True
        try:
            MDAnalysis.coordinates.DCD.DCDReader(f"{cwd}/{dataset}/{record}/{cycle}/{record}.dcd")
        except Exception:
            do_merge = True
            name_dcd_readable = False
        # os.system(f"rm {cwd}/{dataset}/{record}/{cycle}/fret-data-1-90.pkl")
        # os.system(f"rm {cwd}/{dataset}/{record}/{cycle}/fret-data-{allproteins.loc[record].N - 1:d}-2.pkl")
        # os.system(f"rm {cwd}/{dataset}/{record}/{cycle}/proteinsPRE_validate_{record}.pkl")
        # os.system(f"rm {cwd}/{dataset}/{record}/{cycle}/{record}_SAXS.pkl")
        if name_dcd_readable and len(MDAnalysis.coordinates.DCD.DCDReader(f"{cwd}/{dataset}/{record}/{cycle}/{record}.dcd"))!=int(replicas * nframes):
            do_merge = True
        if validate:
            if (record in validateSAXS_pro) and (not os.path.exists(f"{cwd}/{dataset}/{record}/{cycle}/{record}_SAXS.pkl")):
                do_merge = True
            if (record in validatePRE_pro) and (not os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/proteinsPRE_validate_{record}.pkl')):
                do_merge = True
            if (record in ['Nup49', 'NLS', 'NUS', 'IBB', 'NUL']) and (not os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/fret-data-{allproteins.loc[record].N - 1:d}-2.pkl')):
                do_merge = True
            if (record in ['Sic1']) and (not os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/fret-data-1-90.pkl')):
                do_merge = True
            if (record in ['Nup49', 'NLS', 'NUS', 'IBB', 'NUL']) and (not os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/fret30-data-{allproteins.loc[record].N - 1:d}-2.pkl')):
                do_merge = True
            if (record in ['Sic1']) and (not os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/fret30-data-1-90.pkl')):
                do_merge = True
            if (record in list(more_fret.index)) and (not os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/fret10-data-{more_fret.loc[record].labels[0]}-{more_fret.loc[record].labels[-1]}.pkl')):
                do_merge = True

        if do_merge:
            print(record)
            if not os.path.isdir(f"{cwd}/{dataset}/{record}/{cycle}"):
                os.system(f"mkdir -p {cwd}/{dataset}/{record}/{cycle}")
            merge_commandline = f"python3 -u {cwd}/merge_replicas.py --config {cwd}/{dataset}/{record}/{cycle}/{config_merge_filename}"
            L = int(np.ceil((prot.N - 1) * 0.38 + 4))
            if record in multidomain_names:
                isIDP = False
                path2fasta = f'{cwd}/multidomain_fasta/{name}.fasta'  # no fasta needed if pdb provided
                input_pae = ""  # decide_best_pae(cwd, name)
                path2pdb = f'{cwd}/extract_relax/{name}_rank0_relax.pdb'  # af2 predicted structure
                use_pdb = True
                use_hnetwork = True
                use_ssdomains = True
                domain_len = 0
                for domain in get_ssdomains(name, fdomains, output=False):
                    domain_len += len(domain)
                N_save = 3000 if prot.N_res < 100 else int(np.ceil(3e-4 * prot.N_res ** 2) * 1000)  # interval
                cost = (allproteins.loc[record].N ** 2 + domain_len * 15) * N_save / 1000
            else:
                isIDP = True
                path2fasta = ""  # no fasta needed if pdb provided
                input_pae = None
                path2pdb = ""  # af2 predicted structure
                use_pdb = False
                use_hnetwork = False
                use_ssdomains = False
                N_save = 3000 if prot.N_res < 100 else int(np.ceil(3e-4 * prot.N_res ** 2) * 1000)  # interval
                cost = allproteins.loc[record].N ** 2 * N_save / 1000
            N_steps = (nframes + discard_first_nframes) * int(N_save)
            Threads, node, exclude_node = determineThreadsnode(use_newdata, batch_sys, cost)  # Threads for each replica
            collected_replicas = []
            replicas_list4MD = []
            for replica in range(replicas):
                dcd_readable = True
                do_MD = False
                try:
                    MDAnalysis.coordinates.DCD.DCDReader(f"{cwd}/{dataset}/{record}/{cycle}/{replica}.dcd")
                except Exception:
                    dcd_readable = False
                    do_MD = True
                if dcd_readable and len(MDAnalysis.coordinates.DCD.DCDReader(f"{cwd}/{dataset}/{record}/{cycle}/{replica}.dcd")) != int(nframes + discard_first_nframes):
                    do_MD = True
                if do_MD:
                    if len(replicas_list4MD)+1 <= cpu_max//Threads:
                        replicas_list4MD.append(replica)
                    else:
                        collected_replicas.append(replicas_list4MD)
                        replicas_list4MD = []
                        replicas_list4MD.append(replica)
            collected_replicas.append(replicas_list4MD)
            for replicas_list4MD_idx, replicas_list4MD in enumerate(collected_replicas):
                if len(replicas_list4MD)!=0:
                    config_sim_filename = f'config_sim{replicas_list4MD_idx}.yaml'
                    config_sim_data = dict(cwd=cwd, name=name, dataset=dataset,
                       path2fasta=path2fasta, temp=float(prot.temp), ionic=float(prot.ionic), cycle=cycle, pH=float(prot.pH),
                       replicas_list4MD=replicas_list4MD, cutoff=cutoff, L=L, wfreq=int(N_save), slab=slab,
                       use_pdb=use_pdb, path2pdb=path2pdb, use_hnetwork=use_hnetwork, fdomains=fdomains,
                       use_ssdomains=use_ssdomains, input_pae=input_pae, k_restraint=k_restraint, record=record,
                       runtime=runtime, gpu_id=gpu_id%4, Threads=Threads, overwrite=True, N_res=prot.N_res,
                       CoarseGrained=CoarseGrained, isIDP=isIDP, Usecheckpoint=Usecheckpoint, eps_factor=eps_factor,
                       initial_type=initial_type, seq=prot.fasta, steps=N_steps, gpu=gpu, replicas=replicas,
                       discard_first_nframes=discard_first_nframes,validate=validate, nframes=nframes)
                    write_config(cwd, dataset, record, cycle, config_sim_data, config_filename=config_sim_filename)
                    if (not gpu) and len(cycles) != 1 and cycle != cycles[0]:
                        sim_dependency = f"#SBATCH --dependency=afterok:{optimize_jobid}" if batch_sys in ['Deic', 'ROBUST'] else f"#PBS -W depend=afterok:{optimize_jobid}"
                    else:
                        sim_dependency = ""
                    requested_resource = f"1:ppn={len(replicas_list4MD)*Threads}"
                    simrender_dict = dict(cwd=cwd, dataset=dataset, record=record, cycle=f'{cycle}',
                        requested_resource=requested_resource, node=node, sim_dependency=sim_dependency,
                        config_sim_filename=config_sim_filename, Threads=Threads, mem="153600" if gpu else f"{len(replicas_list4MD) * 1000}",
                    requested_cpunum=1 if gpu else len(replicas_list4MD) * Threads, replicas_list4MD_idx=replicas_list4MD_idx,
                                          exclude_node=exclude_node)
                    if gpu:
                        command_lines_dict[pro_idx % parallel_gpus].append(f"python3 -u {cwd}/simulate.py --config {cwd}/{dataset}/{record}/{cycle}/{config_sim_filename} --CUDA_VISIBLE_DEVICES $CUDA_VISIBLE_DEVICES")
                    else:
                        with open(f"{cwd}/{dataset}/{record}/{cycle}/{record}_sim{replicas_list4MD_idx}.{'sh' if batch_sys=='Deic' else 'pbs'}", 'w') as submit:
                            if batch_sys == "Deic":
                                submit.write(submission_1.render(simrender_dict))
                            else:
                                submit.write(submission_4.render(simrender_dict))
                        proc = subprocess.run([f"{'sbatch' if batch_sys in ['Deic', 'ROBUST'] else 'qsub'}", f"{cwd}/{dataset}/{record}/{cycle}/{record}_sim{replicas_list4MD_idx}.{'sh' if batch_sys in ['Deic', 'ROBUST'] else 'pbs'}"],capture_output=True)
                        os.system("sleep 1")
                        print(proc)
                        jobid_1.append(int(proc.stdout.split(b' ')[-1].split(b'\\')[0]))
            # merge
            if len(jobid_1) != 0:
                jobs = ""
                for job in jobid_1:
                    jobs += f":{job}"
                merge_dependency = f"#SBATCH --dependency=afterok{jobs}" if batch_sys in ['Deic', 'ROBUST'] else f"#PBS -W depend=afterok{jobs}"
            else:
                merge_dependency = ""
            config_mer_filename = f'config_merge.yaml'
            # copy of config_sim_data
            num_cpus = 2  # 2 for fret calculation
            config_merge_data = dict(cwd=cwd, name=name, dataset=dataset, path2fasta=path2fasta, temp=float(prot.temp),
                                     ionic=float(prot.ionic), cycle=cycle, pH=float(prot.pH),
                                     replicas_list4MD=replicas_list4MD, cutoff=cutoff, L=L, wfreq=int(N_save),
                                     slab=slab, use_pdb=use_pdb, path2pdb=path2pdb, use_hnetwork=use_hnetwork,
                                     fdomains=fdomains, use_ssdomains=use_ssdomains, input_pae=input_pae,
                                     k_restraint=k_restraint, record=record, runtime=runtime, gpu_id=gpu_id % 4,
                                     Threads=Threads, overwrite=True, N_res=prot.N_res, CoarseGrained=CoarseGrained,
                                     isIDP=isIDP, Usecheckpoint=Usecheckpoint, eps_factor=eps_factor,
                                     initial_type=initial_type, seq=prot.fasta, steps=N_steps, gpu=gpu,
                                     replicas=replicas, discard_first_nframes=discard_first_nframes, validate=validate,
                                     nframes=nframes, path2gmx=path2gmx, pathToconvert_cg2all=pathToconvert_cg2all,
                                     validate_pro=validate_pro, validateFRET_pro=validateFRET_pro,
                                     validatePRE_pro=validatePRE_pro, num_cpus=num_cpus,
                                     validateSAXS_pro=validateSAXS_pro, validateRg_pro=validateRg_pro)
            write_config(cwd, dataset, record, cycle, config_merge_data, config_filename=config_mer_filename)
            mergerender_dict = dict(cwd=cwd, dataset=dataset, record=record, cycle=f'{cycle}',
                requested_resource="1:ppn=1", node=node, merge_dependency=merge_dependency, num_cpus=num_cpus,
                config_mer_filename=config_mer_filename, mem=10, merge_commandline=merge_commandline)
            with open(f"{cwd}/{dataset}/{record}/{cycle}/{record}.{'sh' if batch_sys in ['Deic', 'ROBUST'] else 'pbs'}", 'w') as submit:
                if batch_sys == "Deic":
                    submit.write(submission_2.render(mergerender_dict))
                elif batch_sys == "ROBUST":
                    submit.write(submission_2_rb.render(mergerender_dict))
                else:
                    submit.write(submission_6.render(mergerender_dict))
            if gpu:
                command_lines_dict[pro_idx % parallel_gpus].append(f"{'sbatch' if batch_sys in ['Deic', 'ROBUST'] else 'qsub'} {cwd}/{dataset}/{record}/{cycle}/{record}.{'sh' if batch_sys in ['Deic', 'ROBUST'] else 'pbs'}")
                pro_idx += 1  # more dynamic
            else:
                proc = subprocess.run([f"{'sbatch' if batch_sys in ['Deic', 'ROBUST'] else 'qsub'}",
                    f"{cwd}/{dataset}/{record}/{cycle}/{record}.{'sh' if batch_sys in ['Deic', 'ROBUST'] else 'pbs'}"], capture_output=True)
                print(proc)
                jobid_2.append(int(proc.stdout.split(b' ')[-1].split(b'\\')[0]))

    if not gpu:
        print(f'Simulating: {jobid_1}')
    if gpu:
        for parallel_gpu in range(parallel_gpus):
            if len(command_lines_dict[parallel_gpu]) == 0:
                continue
            commands = "\n".join(command_lines_dict[parallel_gpu])
            simrender_dict = dict(cwd=cwd, dataset=dataset, cycle=f'{cycle}', commands=commands,
                                  gpu_job_name=f"cycle{cycle}sim{parallel_gpu}", mem="20480", requested_cpunum=1)

            with open(f"{cwd}/{dataset}/gpu_cycle{cycle}sim{parallel_gpu}.sh", 'w') as submit:
                submit.write(submission_1_rb.render(simrender_dict))
            # if cycle == cycles[0]:
            proc = subprocess.run([f"{'sbatch' if batch_sys in ['Deic', 'ROBUST'] else 'qsub'}", f"{cwd}/{dataset}/gpu_cycle{cycle}sim{parallel_gpu}.sh"], capture_output=True)
            print(proc)
            jobid_2.append(int(proc.stdout.split(b' ')[-1].split(b'\\')[0]))

    if not validate:
        # optimize
        if len(jobid_2) != 0:
            jobs = ""
            for job in jobid_2:
                jobs += f":{job}"
            opt_dependency = f"#SBATCH --dependency=afterok{jobs}" if batch_sys in ['Deic', 'ROBUST'] else f"#PBS -W depend=afterok{jobs}"
        else:
            opt_dependency = ""
        config_filename = f'config_opt{cycle}_{theta}.yaml'

        if gpu:
            num_cpus = 14
            mem = 80
            submit_next = f"python3 -u {cwd}/submit.py --batch_sys {batch_sys} --CoarseGrained {CoarseGrained} --gpu_str {gpu_str} --theta {theta} --cycle {cycle + 1} --dataset_replica {dataset_replica} --cutoff {cutoff} --validate_indicator {validate_indicator} --dataset_prefix {dataset_prefix} --max_cycle {max_cycle}"
        else:
            submit_next = ""
            if partition == "sbinlab_ib2":
                num_cpus = 20
                mem = 80
            else:
                num_cpus = 20
                mem = 300  # 300
        config_data = dict(cwd=cwd, log_path="LOG", dataset=dataset, cycle=cycle, num_cpus=num_cpus if batch_sys in ['Deic', "ROBUST"] else 38, cutoff=cutoff,
                           use_newdata=use_newdata, theta=theta, include_PREloss=include_PREloss, fdomains=fdomains,
                            initial_type=initial_type, rebalancechi2_rg=rebalancechi2_rg, lambda_oneMax=lambda_oneMax,
                           allow_negative=allow_negative, fractPrior=fractPrior, xi_0=xi_0, effective_frac=effective_frac,
                           resetErr=resetErr, resetErr_ratio=resetErr_ratio, minmax_TF=minmax_TF, C2_prior=C2_prior)
        yaml.dump(config_data, open(f'{cwd}/{dataset}/{config_filename}', 'w'))
        optrender_dict = dict(cwd=cwd, dataset=dataset, opt_dependency=opt_dependency,proteins=' '.join(proteinsPRE.index),
                              cycle=f'{cycle}', path2config=f"{cwd}/{dataset}/{config_filename}", num_cpus=num_cpus, partition=partition,
                                mem=mem, include_PREloss=str(include_PREloss), submit_next=submit_next)
        with open(f"{cwd}/{dataset}/opt_{cycle}.{'sh' if batch_sys in ['Deic', 'ROBUST'] else 'pbs'}", 'w') as submit:
            if batch_sys == "Deic":
                submit.write(submission_3.render(optrender_dict))
            elif batch_sys == "ROBUST":
                submit.write(submission_3_rb.render(optrender_dict))
            else:
                submit.write(submission_7.render(optrender_dict))
        proc = subprocess.run([f"{'sbatch' if batch_sys in ['Deic', 'ROBUST'] else 'qsub'}",f"{cwd}/{dataset}/opt_{cycle}.{'sh' if batch_sys in ['Deic', 'ROBUST'] else 'pbs'}"],capture_output=True)
        print(proc)
        optimize_jobid = int(proc.stdout.split(b' ')[-1].split(b'\\')[0])
