import logging
from FRETpredict import FRETpredict
from ast import literal_eval
import MDAnalysis
import os
import yaml
from utils import visualize_traj, energy_details
from rawdata import *
from argparse import ArgumentParser
from utils import load_parameters, evaluatePRE, SAXSout_MPI, extractSAXS
import ray
from DEERPREdict.PRE import PREpredict

def centerDCD(config):
    record = config["record"]
    cwd = config["cwd"]
    nframes = config["nframes"]
    dataset = config["dataset"]
    cycle = config["cycle"]
    replicas = config["replicas"]
    pathToconvert_cg2all = config["pathToconvert_cg2all"]
    path2gmx = config["path2gmx"]
    discard_first_nframes = config["discard_first_nframes"]
    validate = config["validate"]
    initial_type = config["initial_type"]
    validate_pro = config["validate_pro"]
    validatePRE_pro = config["validatePRE_pro"]
    validateSAXS_pro = config["validateSAXS_pro"]
    validateRg_pro = config["validateRg_pro"]
    num_cpus = config["num_cpus"]

    more_fret = pd.read_pickle(f'{cwd}/proteins_fret.pkl')
    print("validate:", validate)
    residues = load_parameters(cwd, dataset, cycle, initial_type)
    if not validate:
        prot = pd.read_pickle(f'{cwd}/{dataset}/allproteins.pkl').loc[record]
    else:
        prot = pd.read_pickle(f'{cwd}/{dataset}/allproteins_validate.pkl').loc[record]

    incomplete = True
    while incomplete:
        try:
            top = md.Topology()
            chain = top.add_chain()
            for resname in prot.fasta:
                residue = top.add_residue(residues.loc[resname, 'three'], chain)
                top.add_atom(residues.loc[resname, 'three'], element=md.element.carbon, residue=residue)
            for i in range(len(prot.fasta) - 1):
                top.add_bond(top.atom(i), top.atom(i + 1))
            traj = md.load_dcd(f"{cwd}/{dataset}/{record}/{cycle}/0.dcd", top=top)[discard_first_nframes:]
            for i in range(1, replicas):
                t = md.load_dcd(f"{cwd}/{dataset}/{record}/{cycle}/{i}.dcd", top=top)[discard_first_nframes:]
                traj = md.join([traj, t])
            traj = traj.image_molecules(inplace=False, anchor_molecules=[set(traj.top.chain(0).atoms)], make_whole=True)
            assert len(traj) == int(nframes * replicas)
            traj.center_coordinates()
            traj.xyz += traj.unitcell_lengths[0, 0] / 2
            print(f'Number of frames: {traj.n_frames}')
            traj.save_dcd(f'{cwd}/{dataset}/{record}/{cycle}/{record}.dcd')
            traj[0].save_pdb(f'{cwd}/{dataset}/{record}/{cycle}/{record}.pdb')
        except Exception:
            os.system("rm /home/people/fancao/core*")
            os.system("sleep 1")
        else:
            incomplete = False

    visualize_traj(cwd, dataset, record, cycle)
    if validate:
        if record in validate_pro:
            if record not in validateRg_pro:
                ray.init(num_cpus=num_cpus)
                # reconstruction speed is determined by the requested cpu num in submitting script
                if not os.path.exists(f"{cwd}/{dataset}/{record}/{cycle}/allatom.pdb"):
                    os.system(f"{pathToconvert_cg2all} -p {cwd}/{dataset}/{record}/{cycle}/{record}_first.pdb -d {cwd}/{dataset}/{record}/{cycle}/{record}.dcd -o {cwd}/{dataset}/{record}/{cycle}/allatom.dcd -opdb {cwd}/{dataset}/{record}/{cycle}/{record}_topAA.pdb --cg CalphaBasedModel --device cpu")
                    os.system(f"{path2gmx} editconf -resnr 1 -f {cwd}/{dataset}/{record}/{cycle}/{record}_topAA.pdb -o {cwd}/{dataset}/{record}/{cycle}/allatom.pdb")

                # SAXS calculation
                if record in validateSAXS_pro:
                    if not os.path.isfile(f"{cwd}/experimentalSAXS/bift_{record}.dat"):
                        raise Exception(f"{cwd}/experimentalSAXS/bift_{record}.dat does not exist!")
                    if not os.path.isdir(f"{cwd}/{dataset}/{record}/{cycle}/allatom"):
                        os.system(f"mkdir -p {cwd}/{dataset}/{record}/{cycle}/allatom")
                    AAdcd = md.load_dcd(f"{cwd}/{dataset}/{record}/{cycle}/allatom.dcd",f"{cwd}/{dataset}/{record}/{cycle}/allatom.pdb")
                    total_frames = len(AAdcd)
                    assert total_frames % num_cpus == 0
                    subframes = int(total_frames/num_cpus)
                    ray.get([SAXSout_MPI.remote(AAdcd[subframes*n:subframes*(n+1)], n, subframes, config) for n in range(num_cpus)])
                    extractSAXS(cwd, dataset, record, cycle, total_frames)
                    os.system(f"rm -rf {cwd}/{dataset}/{record}/{cycle}/allatom")


                # PRE calculation
                if record in validatePRE_pro:
                    proteinsPRE = pd.read_pickle(f'{cwd}/{dataset}/proteinsPRE_validate.pkl').astype(object)  # can read an empty dataframe
                    proteinsPRE["weights"] = False  # pandas needs to be 1.5.2 or above!!!!!!!!
                    proteinsPRE = proteinsPRE.loc[record]
                    proc_PRE = [(label, record) for label in proteinsPRE.labels]  # an empty dataframe will return an empty list
                    if not os.path.isdir(f'{cwd}/{dataset}/{record}/{cycle}/calcPREs'):
                        os.mkdir(f'{cwd}/{dataset}/{record}/{cycle}/calcPREs')
                    os.system(f"cp -r {cwd}/expPREs/{record}/expPREs {cwd}/{dataset}/{record}")
                    proteinsPRE['expPREs'] = loadExpPREs(cwd, dataset, record, proteinsPRE)
                    ray.get([evaluatePRE.remote(cwd, dataset, label, record, cycle, proteinsPRE, "LOG") for n,(label,record) in enumerate(proc_PRE)])  # empty dataframe will not perform evaluatePRE calculations
                    tau_c, chi2_pre = optTauC(cwd, dataset, record, cycle, proteinsPRE)
                    proteinsPRE['tau_c'] = tau_c
                    proteinsPRE['chi2_pre'] = chi2_pre
                    proteinsPRE['initPREs'] = loadInitPREs(cwd, dataset, record, cycle, proteinsPRE)
                    proteinsPRE.to_pickle(f'{cwd}/{dataset}/{record}/{cycle}/proteinsPRE_validate_{record}.pkl')

                # FRET calculation
                u = MDAnalysis.Universe(f"{cwd}/{dataset}/{record}/{cycle}/allatom.pdb", f"{cwd}/{dataset}/{record}/{cycle}/allatom.dcd")
                df_rg = pd.read_csv(f'{cwd}/rg_test_data.csv', index_col=0)
                df_rg = df_rg.sort_values('N', ascending=False)
                df_rg.rel_rg_err_replicas = df_rg.rel_rg_err_replicas.apply(lambda x: literal_eval(x))
                df_ped = pd.read_csv(f'{cwd}/PED_data.csv', index_col=0)
                df_ped.loc[df_ped.drop('p27Kip1').index, 'exp_rg'] = df_rg.loc[df_ped.drop('p27Kip1').index, 'exp_rg']
                df_ped.loc[df_ped.drop('p27Kip1').index, 'exp_rg_err'] = df_rg.loc[df_ped.drop('p27Kip1').index, 'exp_rg_err']
                # FRET cutoff20
                if (record in ['Sic1']) and (not os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/fret-data-1-90.pkl')):
                    FRET = FRETpredict(protein=u, residues=[1, 90], chains=['A', 'A'], temperature=293,
                                       donor='AlexaFluor 488', acceptor='AlexaFluor 647', electrostatic=True,
                                       libname_1=f'AlexaFluor 488 C1R cutoff20',
                                       libname_2=f'AlexaFluor 647 C2R cutoff20',
                                       output_prefix=f'{cwd}/{dataset}/{record}/{cycle}/fret')
                    FRET.run()
                if (record in ['NLS', 'NUS', 'IBB', 'NUL', 'Nup49']) and (not os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/fret-data-{len(prot.fasta) - 1:d}-2.pkl')):
                    FRET = FRETpredict(protein=u, residues=[df_ped.loc[record].N - 1, 2], chains=['A', 'A'],
                                       temperature=296,
                                       donor='AlexaFluor 488', acceptor='AlexaFluor 594', electrostatic=True,
                                       libname_1=f'AlexaFluor 488 C1R cutoff20',
                                       libname_2=f'AlexaFluor 594 C1R cutoff20',
                                       output_prefix=f'{cwd}/{dataset}/{record}/{cycle}/fret')
                    FRET.run()

                # FRET cutoff30
                if (record in ['Sic1']) and (not os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/fret30-data-1-90.pkl')):
                    FRET = FRETpredict(protein=u, residues=[1, 90], chains=['A', 'A'], temperature=293,
                                       donor='AlexaFluor 488', acceptor='AlexaFluor 647', electrostatic=True,
                                       libname_1=f'AlexaFluor 488 C1R cutoff30',
                                       libname_2=f'AlexaFluor 647 C2R cutoff30',
                                       output_prefix=f'{cwd}/{dataset}/{record}/{cycle}/fret30')
                    FRET.run()
                if (record in ['NLS', 'NUS', 'IBB', 'NUL', 'Nup49']) and (not os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/fret30-data-{len(prot.fasta) - 1:d}-2.pkl')):
                    FRET = FRETpredict(protein=u, residues=[df_ped.loc[record].N - 1, 2], chains=['A', 'A'],
                                       temperature=296,
                                       donor='AlexaFluor 488', acceptor='AlexaFluor 594', electrostatic=True,
                                       libname_1=f'AlexaFluor 488 C1R cutoff30',
                                       libname_2=f'AlexaFluor 594 C1R cutoff30',
                                       output_prefix=f'{cwd}/{dataset}/{record}/{cycle}/fret30')
                    FRET.run()
                # new FRET proteins
                if (record in list(more_fret.index)) and (not os.path.exists(f'{cwd}/{dataset}/{record}/{cycle}/fret10-data-{more_fret.loc[record].labels[0]}-{more_fret.loc[record].labels[-1]}.pkl')):
                    FRET = FRETpredict(u, log_file=f'{cwd}/{dataset}/{record}/{cycle}/fret_log', residues=more_fret.loc[record].labels,
                                       chains=['A', 'A'], temperature=more_fret.loc[record].temp, fixed_R0=True, r0=6.0,
                                       donor='Lumiprobe Cy3b C2R', acceptor='CF 660R C2R', electrostatic=False,
                                       libname_1='Lumiprobe Cy3b C2R cutoff10', libname_2='CF 660R C2R cutoff10',
                                       output_prefix=f'{cwd}/{dataset}/{record}/{cycle}/fret10',
                                       verbose=False, calc_distr=False)
                    FRET.run()
        else:
            raise


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--config', nargs='?', default='config.yaml', const='config.yaml', type=str)
    args = parser.parse_args()
    with open(f'{args.config}', 'r') as stream:
        config = yaml.safe_load(stream)
    centerDCD(config)
    # energy_details(args.cwd, args.dataset, args.name, args.cycle, args.fdomains, rc=args.rc)
