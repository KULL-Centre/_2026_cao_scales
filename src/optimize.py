import warnings

import numpy as np
import pandas as pd
from utils import evaluatePRE, calcDistSums, calcAHenergy, reweight, load_parameters, balance_chi2_rg, minmax, checknecessaryfiles
warnings.filterwarnings('ignore')
from rawdata import *
import time
import os
from argparse import ArgumentParser
from sklearn.neighbors import KernelDensity
import ray
import logging
import shutil
import yaml

def optimize(config):
    checknecessaryfiles(config)
    Time = lambda: time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    time_total = time.time()
    cwd, log_path, dataset, cycle, num_cpus, cutoff = config["cwd"], config["log_path"],config["dataset"],config["cycle"],config["num_cpus"], config["cutoff"]
    theta = config["theta"]
    include_PREloss = config["include_PREloss"]
    fdomains = config["fdomains"]
    fractPrior = config["fractPrior"]
    initial_type = config["initial_type"]
    rebalancechi2_rg = config["rebalancechi2_rg"]
    allow_negative = config["allow_negative"]
    lambda_oneMax = config["lambda_oneMax"]
    xi_0 = config["xi_0"]
    effective_frac = config["effective_frac"]
    resetErr = config["resetErr"]
    resetErr_ratio = config["resetErr_ratio"]
    minmax_TF = config["minmax_TF"]
    C2_prior = config["C2_prior"]

    ray.init(num_cpus=num_cpus)  # will ignore the requested num of cpus in submiting script, so num_cpus should be declared again here;
    dp = 0.05
    eta = .1
    # xi_0 = .5  # increase from .1 to .5
    rc = cutoff
    # os.environ["NUMEXPR_MAX_THREADS"] = "1"

    if not os.path.isdir(f"{cwd}/{dataset}/{log_path}"):
        os.system(f"mkdir -p {cwd}/{dataset}/{log_path}")

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s|%(message)s",
                        datefmt = '%Y-%m-%d|%H:%M:%S-%a',
                        filename=f'{cwd}/{dataset}/{log_path}/log'
                        )
    IDPsRgs = pd.read_pickle(f'{cwd}/{dataset}/proteinsRgs.pkl').astype(object)
    MultiDomainsRgs = pd.DataFrame()  # no MDPs used here
    proteinsRgs = IDPsRgs
    proteinsPRE = pd.read_pickle(f'{cwd}/{dataset}/proteinsPRE.pkl').astype(object)  # can read an empty dataframe
    multidomain_names = list(MultiDomainsRgs.index)
    proteinsPRE["weights"] = False  # pandas needs to be 1.5.2 or above!!!!!!!!
    allproteins = pd.concat((proteinsPRE, IDPsRgs, MultiDomainsRgs), sort=True)

    for record in list(proteinsPRE.index):  # an empty dataframe will not enter this loop
        if not os.path.isdir(f'{cwd}/{dataset}/{record}/{cycle}'):
            os.mkdir(f'{cwd}/{dataset}/{record}/{cycle}')
        if not os.path.isdir(f'{cwd}/{dataset}/{record}/{cycle}/calcPREs'):
            os.mkdir(f'{cwd}/{dataset}/{record}/{cycle}/calcPREs')

    proc_PRE = [(label,record) for record,prot in proteinsPRE.iterrows() for label in prot.labels]  # an empty dataframe will return an empty list
    # [(24, 'aSyn'), (42, 'aSyn'), (62, 'aSyn'), (87, 'aSyn'), (103, 'aSyn'), (10, 'OPN'), (33, 'OPN'), (64, 'OPN'), (88, 'OPN'), (117, 'OPN'), (130, 'OPN'), (144, 'OPN'), (162, 'OPN'), (184, 'OPN'), (203, 'OPN'), (16, 'FUS'), (86, 'FUS'), (142, 'FUS'), (16, 'FUS12E'), (86, 'FUS12E'), (142, 'FUS12E')]

    df = load_parameters(cwd, dataset, cycle, initial_type)
    df = df.set_index("three")
    if C2_prior:
        prior_lambdas = pd.read_csv(f"{cwd}/residues_pub.csv").set_index("one", drop=False)
        prior_lambdas["lambdas"] = prior_lambdas.CALVADOS2
        if minmax_TF:
            prior_lambdas = minmax(prior_lambdas).set_index("one", drop=False)
        else:
            prior_lambdas = prior_lambdas.set_index("one", drop=False)

    logging.info(f"################cycle{cycle} optimization starts################")
    logging.info(f"fractPrior: {fractPrior}")
    logging.info(f"lambda_oneMax: {lambda_oneMax}")
    logging.info(f"minmax_TF: {minmax_TF}")
    logging.info(f"C2_prior: {C2_prior},\n {prior_lambdas if C2_prior else ''}")
    logging.info(f"resetErr: {resetErr}, {f'resetErr_ratio: {resetErr_ratio}' if resetErr else ''}")
    logging.info(f"effective_frac: {effective_frac}")
    logging.info(f"PRE sequences({len(proteinsPRE.index)}) used for optimization: {list(proteinsPRE.index)}")
    logging.info(f"Rgs sequences({len(proteinsRgs.index)}) used for optimization: {list(proteinsRgs.index)}")
    logging.info(f"cycle: {cycle}, before optimization:\n {df.lambdas}")

    for record in proteinsPRE.index:
        # type(proteinsPRE.loc["aSyn"]["expPREs"])-> <class 'pandas.core.frame.DataFrame'>
        proteinsPRE.at[record,'expPREs'] = loadExpPREs(cwd,dataset,record,proteinsPRE.loc[record])
    time0 = time.time()
    # proteinsPRE["weights"]
    # aSyn      False
    # OPN       False
    # FUS       False
    # FUS12E    False

    # !!!!!!!!!!!!
    ray.get([evaluatePRE.remote(cwd,dataset,label,record,cycle,proteinsPRE.loc[record], log_path) for n,(label,record) in enumerate(proc_PRE)])  # empty dataframe will not perform evaluatePRE calculations
    logging.info(f'Timing evaluatePRE {np.round(time.time()-time0, 3)}s')


    time0 = time.time()
    prot4calcDistSums = pd.concat((proteinsPRE, proteinsRgs), sort=True)
    prot4calcDistSums['N'] = prot4calcDistSums['fasta'].apply(lambda x: len(x))
    prot4calcDistSums = prot4calcDistSums.sort_values('N', ascending=False)
    ray.get([calcDistSums.remote(cwd, dataset, df, record.split("@")[0], record, cycle, prot, multidomain_names, rc, fdomains) for record, prot in prot4calcDistSums.iterrows()])
    logging.info(f'Timing calcDistSums {np.round(time.time()-time0, 3)}s')


    #!!!!!!!!!!!
    for record in proteinsPRE.index:
        np.save(f'{cwd}/{dataset}/{record}/{cycle}/{record}_AHenergy.npy',
            calcAHenergy(cwd,dataset, df, record, cycle))
        tau_c, chi2_pre = optTauC(cwd, dataset, record, cycle, proteinsPRE.loc[record])
        proteinsPRE.at[record,'tau_c'] = tau_c
        proteinsPRE.at[record,'chi2_pre'] = chi2_pre
        proteinsPRE.at[record,'initPREs'] = loadInitPREs(cwd, dataset, record, cycle, proteinsPRE.loc[record])
        if os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/initPREs'):
            shutil.rmtree(f'{cwd}/{dataset}/{record}/{cycle}/initPREs')  # delete files recursively
        shutil.copytree(f'{cwd}/{dataset}/{record}/{cycle}/calcPREs',f'{cwd}/{dataset}/{record}/{cycle}/initPREs')  # copy dir
    if include_PREloss:
        proteinsPRE.to_pickle(f'{cwd}/{dataset}/{str(cycle)}_init_proteinsPRE.pkl')
    # !!!!!!!!!!
    # proteinsPRE = pd.read_pickle(f'{cwd}/{dataset}/{str(cycle)}_init_proteinsPRE.pkl').astype(object)
    for record in proteinsRgs.index:
        np.save(f'{cwd}/{dataset}/{record}/{cycle}/{record}_AHenergy.npy',
            calcAHenergy(cwd,dataset, df,record, cycle))
        rgarray, rg, chi2_rg = calcRg(cwd,dataset, df,record,cycle,proteinsRgs.loc[record])
        proteinsRgs.at[record,'rgarray'] = rgarray
        proteinsRgs.at[record,'Rg'] = rg
        proteinsRgs.at[record,'chi2_rg'] = chi2_rg
    proteinsRgs.to_pickle(f'{cwd}/{dataset}/{str(cycle)}_init_proteinsRgs.pkl')

    if include_PREloss:
        logging.info(f'Initial Chi2 PRE {np.round(proteinsPRE.chi2_pre.mean(),3)} +/- {np.round(proteinsPRE.chi2_pre.std(),3)}')
    logging.info(f'Initial Chi2 Gyration Radius {np.round(balance_chi2_rg(IDPsRgs, MultiDomainsRgs, proteinsRgs) if rebalancechi2_rg else proteinsRgs.chi2_rg.mean(),3)} +/- {np.round(proteinsRgs.chi2_rg.std(),3)}')

    selHPS = pd.read_csv(f'{cwd}/selHPS.csv',index_col=0)
    # kde = KernelDensity(kernel='gaussian',bandwidth=0.05).fit(selHPS.T.values)
    if minmax_TF:
        trial_df_minmax = minmax(df).set_index("one", drop=False)
    else:
        trial_df_minmax = df.set_index("one", drop=False)
    if not fractPrior:
        if not C2_prior:
            kde = KernelDensity(kernel='gaussian', bandwidth=.05).fit(selHPS.loc[trial_df_minmax.one].T.values)
            theta_prior = theta * kde.score_samples(trial_df_minmax.lambdas.values.reshape(1, -1))[0]
        else:
            theta_prior = -theta * np.mean(np.power(trial_df_minmax.lambdas.to_numpy() - prior_lambdas.loc[trial_df_minmax.index].lambdas.to_numpy(), 2))
    else:
        frac = {}
        for aa in trial_df_minmax.one:
            n_aa = allproteins.fasta.sum().count(aa)
            n_tot = len(allproteins.fasta.sum())
            frac[aa] = n_aa / n_tot
        if not C2_prior:
            kde = {}
            for aa in trial_df_minmax.one:
                kde[aa] = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(selHPS.loc[aa].values.reshape(-1, 1))
            tmp = [kde[aa].score_samples(trial_df_minmax.loc[aa].lambdas.reshape(1, -1))[0] / frac[aa] for aa in trial_df_minmax.one]
            logging.info(f"{list(zip(np.round(tmp, 4), list(trial_df_minmax.one)))}")
            theta_prior = theta * np.sum(tmp)
        else:
            tmp = [np.power(trial_df_minmax.loc[aa].lambdas - prior_lambdas.loc[aa].lambdas, 2) / frac[aa] for aa in trial_df_minmax.index]
            theta_prior = -theta * np.sum(tmp)

    xi = xi_0
    logging.info(f'Initial theta*prior {np.round(theta_prior,2)}')
    logging.info(f'theta {np.round(theta,4)}')
    logging.info('xi {:g}'.format(xi))

    dfchi2 = pd.DataFrame(columns=['chi2_pre','chi2_rg','theta_prior','lambdas','xi','lambdas_minmax'])
    dfchi2.loc[0] = [proteinsPRE.chi2_pre.mean(),balance_chi2_rg(IDPsRgs, MultiDomainsRgs, proteinsRgs) if rebalancechi2_rg else proteinsRgs.chi2_rg.mean(),theta_prior,df.lambdas,xi,trial_df_minmax.lambdas]
    time0 = time.time()
    micro_cycle = 0
    df[f'lambdas_{cycle-1}'] = df.lambdas
    Rgloss_multidomain = pd.DataFrame(columns=[multidomain_names])
    Rgloss_IDP = pd.DataFrame(columns=np.setdiff1d(list(proteinsRgs.index), multidomain_names).tolist())
    cnt_tot = 0
    cnt_passed = 0
    cnt_accepted = 0
    for k in range(2,200000):  # 200000
        cnt_tot += 1
        if (xi<1e-8):
            xi = xi_0
            micro_cycle += 1
            if (micro_cycle==10):
                logging.info('xi {:g}'.format(xi))
                break

        xi = xi * .99
        passed, trial_df, trial_proteinsPRE, trial_proteinsRgs, Rgloss4multidomain, Rgloss4IDPs = reweight(cwd, dataset,cycle, dp,df,proteinsPRE,proteinsRgs,multidomain_names, proc_PRE, log_path, lambda_oneMax, allow_negative, effective_frac)
        if passed:
            cnt_passed += 1
            if minmax_TF:
                trial_df_minmax = minmax(trial_df).set_index("one", drop=False)
            else:
                trial_df_minmax = trial_df.set_index("one", drop=False)
            if not fractPrior:
                if not C2_prior:
                    kde = KernelDensity(kernel='gaussian', bandwidth=.05).fit(selHPS.loc[trial_df_minmax.one].T.values)
                    theta_prior = theta * kde.score_samples(trial_df_minmax.lambdas.values.reshape(1, -1))[0]
                else:
                    theta_prior = -theta * np.mean(np.power(trial_df_minmax.lambdas.to_numpy() - prior_lambdas.loc[trial_df_minmax.index].lambdas.to_numpy(), 2))
            else:
                frac = {}
                for aa in trial_df_minmax.one:
                    n_aa = allproteins.fasta.sum().count(aa)
                    n_tot = len(allproteins.fasta.sum())
                    frac[aa] = n_aa / n_tot
                if not C2_prior:
                    kde = {}
                    for aa in trial_df_minmax.one:
                        kde[aa] = KernelDensity(kernel='gaussian', bandwidth=0.05).fit(
                            selHPS.loc[aa].values.reshape(-1, 1))
                    tmp = [kde[aa].score_samples(trial_df_minmax.loc[aa].lambdas.reshape(1, -1))[0] / frac[aa] for aa in trial_df_minmax.one]
                    logging.info(f"{list(zip(np.round(tmp, 4), list(trial_df_minmax.one)))}")
                    theta_prior = theta * np.sum(tmp)
                else:
                    # print(trial_df_minmax.index, prior_lambdas.index)
                    tmp = [np.power(trial_df_minmax.loc[aa].lambdas - prior_lambdas.loc[aa].lambdas, 2) / frac[aa] for aa in trial_df_minmax.index]
                    theta_prior = -theta * np.sum(tmp)

            delta1 = [balance_chi2_rg(IDPsRgs, MultiDomainsRgs, trial_proteinsRgs) if rebalancechi2_rg else trial_proteinsRgs.chi2_rg.mean()][0] - theta_prior + [eta*trial_proteinsPRE.chi2_pre.mean() if include_PREloss else 0.][0]
            delta2 = [balance_chi2_rg(IDPsRgs, MultiDomainsRgs, proteinsRgs) if rebalancechi2_rg else proteinsRgs.chi2_rg.mean()][0] - dfchi2.iloc[-1]['theta_prior'] + [eta*proteinsPRE.chi2_pre.mean() if include_PREloss else 0.][0]
            delta = delta1 - delta2
            if ( np.exp(-delta/xi) > np.random.rand() ):
                cnt_accepted += 1
                proteinsPRE = trial_proteinsPRE.copy()
                proteinsRgs = trial_proteinsRgs.copy()
                if len(multidomain_names) != 0:
                    Rgloss_multidomain.loc[k-1] = Rgloss4multidomain
                Rgloss_IDP.loc[k-1] = Rgloss4IDPs
                df = trial_df.copy()
                dfchi2.loc[k-1] = [trial_proteinsPRE.chi2_pre.mean(),balance_chi2_rg(IDPsRgs, MultiDomainsRgs, trial_proteinsRgs) if rebalancechi2_rg else trial_proteinsRgs.chi2_rg.mean(),theta_prior,df.lambdas,xi,trial_df_minmax.lambdas]
                logging.info('Iter {:d}, micro cycle {:d}, xi {:g}, Chi2 PRE {:.2f}, Chi2 Rg {:.2f}, theta*prior {:.2f}'.format(k-1,micro_cycle,xi,trial_proteinsPRE.chi2_pre.mean(),balance_chi2_rg(IDPsRgs, MultiDomainsRgs, trial_proteinsRgs) if rebalancechi2_rg else trial_proteinsRgs.chi2_rg.mean(),theta_prior))
                logging.info(f'trial_df: {trial_df}\ntrial_df_minmax: {trial_df_minmax}')
                logging.info('percentage passed {:.3f}'.format(cnt_passed / cnt_tot * 100))
                logging.info('percentage accepted {:.3f}'.format(cnt_accepted / cnt_tot * 100))

    logging.info(f'Timing Reweighting {np.round(time.time()-time0,3)}')
    logging.info(f'Theta {np.round(theta,4)}')

    dfchi2['cost'] = dfchi2.chi2_rg - dfchi2.theta_prior + [eta*dfchi2.chi2_pre if include_PREloss else 0.][0]
    dfchi2.to_pickle(f'{cwd}/{dataset}/{str(cycle)}_chi2.pkl')
    df.lambdas = dfchi2.loc[pd.to_numeric(dfchi2['cost']).idxmin()].lambdas  # that is why the last theta_prior from previous cycle is not necessarily equal to printed one in the log
    logging.info(f"cycle: {cycle}, after optimization:\n {df.lambdas}")
    df[f'lambdas_{cycle}'] = df.lambdas
    if include_PREloss:
        proteinsPRE.to_pickle(f'{cwd}/{dataset}/proteinsPRE.pkl')
    proteinsRgs.to_pickle(f'{cwd}/{dataset}/proteinsRgs.pkl')
    Rgloss_multidomain.to_pickle(f'{cwd}/{dataset}/Rgloss_multidomain_{cycle}.pkl')
    Rgloss_IDP.to_pickle(f'{cwd}/{dataset}/Rgloss_IDP_{cycle}.pkl')
    df.to_csv(f'{cwd}/{dataset}/residues_{cycle}.csv')
    logging.info(f'Cost at 0: {np.round(dfchi2.loc[0].cost,2)}')
    logging.info(f"Min Cost at {pd.to_numeric(dfchi2['cost']).idxmin()}: {np.round(dfchi2.cost.min(),2)}")

    time_end = time.time()
    target_seconds = time_end - time_total  # total used time
    logging.info(f"{Time()}|total optimization used time: {target_seconds // 3600}h {(target_seconds // 60) % 60}min {np.round(target_seconds % 60, 2)}s")
    logging.info(f"################cycle{cycle} optimization ends################")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path2config', dest='path2config', type=str, required=True)
    args = parser.parse_args()
    with open(f'{args.path2config}', 'r') as stream:
        config = yaml.safe_load(stream)
    optimize(config)

    pass
