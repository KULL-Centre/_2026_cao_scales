import pandas as pd
import numpy as np
import mdtraj as md
import itertools
from DEERPREdict.utils import Operations

def loadExpPREs(cwd,dataset, record,prot):
    value = {}
    error = {}
    resnums = np.arange(1,len(prot.fasta)+1)
    for label in prot.labels:
        value[label], error[label] = np.loadtxt(f'{cwd}/{dataset}/{record}/expPREs/exp-{label}.dat',unpack=True)
    v = pd.DataFrame(value,index=resnums)
    v.rename_axis('residue', axis='index', inplace=True)
    v.rename_axis('label', axis='columns',inplace=True)
    e = pd.DataFrame(error,index=resnums)
    e.rename_axis('residue', axis='index', inplace=True)
    e.rename_axis('label', axis='columns',inplace=True)
    return pd.concat(dict(value=v,error=e),axis=1)

def loadInitPREs(cwd, dataset, record, cycle, prot):
    obs = 1 if prot.obs=='ratio' else 2
    value = {}
    resnums = np.arange(1,len(prot.fasta)+1)
    for label in prot.labels:
        value[label] = np.loadtxt(f'{cwd}/{dataset}/{record}/{cycle}/calcPREs/res-{label}.dat')[:,obs]
    v = pd.DataFrame(value,index=resnums)
    v.rename_axis('residue', inplace=True)
    v.rename_axis('label', axis='columns',inplace=True)
    return v

def calcChi2(cwd, dataset, record, cycle, prot):
    obs = 1 if prot.obs=='ratio' else 2
    chi2 = 0
    for label in prot.labels:
        y = np.loadtxt(f'{cwd}/{dataset}/{record}/{cycle}/calcPREs/res-{label}.dat')[:,obs]
        chi = (prot.expPREs.value[label].values - y) / prot.expPREs.error[label].values
        chi = chi[~np.isnan(chi)]
        chi2 += np.nansum( np.power( chi, 2) ) / chi.size
    return chi2 / len(prot.labels)

def optTauC(cwd, dataset, record, cycle, prot):
    obs = 1 if prot.obs == 'ratio' else 2
    chi2list = []
    tau_c = np.arange(2,10.05,1)
    for tc in tau_c:
        chi2 = 0
        for label in prot.labels:
            # the first two columns in res-{label}.dat
            x,y = np.loadtxt(f'{cwd}/{dataset}/{record}/{cycle}/calcPREs/res-{label}.dat',usecols=(0,1),unpack=True)
            # x: [  1.   2.   3. ... 218. 219. 220.] (number of residue)
            # y: [nan 0.14382832 0.28971916 ... 0.99804923 0.99705705 0.99578599] (number of residue)

            # index of residues with a real value (start with 0)
            measured_resnums = np.where(~np.isnan(y))[0]
            data = pd.read_pickle(f'{cwd}/{dataset}/{record}/{cycle}/calcPREs/res-{label}.pkl', compression='gzip')
            # the reason why nan shows up is that the corresponding steric partition function is under cutoff.
            # So it is discarded
            # print(data)
            # r3[[nan, ... nan], ... [1.47580169e-04, ... 3.25930228e-06]]  (traj_len, measured_resnums_len)
            # r6 similar to r3, but distributions of nan are not necessarily the same
            # angular the similar as the above
            # print(data.index)  Index(['r3', 'r6', 'angular'], dtype='object')
            gamma_2_av = np.full(y.size, fill_value=np.NaN)
            s_pre = np.power(data['r3'], 2)/data['r6']*data['angular']  # calculate following the formula
            gamma_2 = Operations.calc_gamma_2(data['r6'], s_pre, tau_c = tc * 1e-9, tau_t = 1e-10, wh = prot.wh, k = 1.23e16)  # calculate following the formula
            # print(gamma_2.shape)  (traj_len, measured_resnums_len)
            gamma_2 = np.ma.MaskedArray(gamma_2, mask = np.isnan(gamma_2))
            gamma_2_av[measured_resnums] = np.ma.average(gamma_2, axis=0).data  # averaged over traj
            # For samples with particularly high PRE rates it can be infeasible to obtain Γ2 from 174 multiple time-point measurements,
            # https://doi.org/10.1101/2020.08.09.243030
            if prot.obs == 'ratio':
                y = 10 * np.exp(-gamma_2_av * 0.01) / ( 10 + gamma_2_av )
            else:
                y = gamma_2_av

            # calculate chi
            chi = (prot.expPREs.value[label].values - y) / prot.expPREs.error[label].values
            chi = chi[~np.isnan(chi)]
            chi2 += np.nansum( np.power( chi, 2) ) / chi.size
        chi2list.append(chi2 / len(prot.labels))

    tc_min = tau_c[np.argmin(chi2list)]  # pick up the smallest value

    for label in prot.labels:
        x,y = np.loadtxt(f'{cwd}/{dataset}/{record}/{cycle}/calcPREs/res-{label}.dat',usecols=(0,1),unpack=True)
        measured_resnums = np.where(~np.isnan(y))[0]
        data = pd.read_pickle(f'{cwd}/{dataset}/{record}/{cycle}/calcPREs/res-{label}.pkl', compression='gzip')
        gamma_2_av = np.full(y.size, fill_value=np.NaN)
        s_pre = np.power(data['r3'], 2)/data['r6']*data['angular']
        gamma_2 = Operations.calc_gamma_2(data['r6'], s_pre, tau_c = tc_min * 1e-9, tau_t = 1e-10, wh = prot.wh, k = 1.23e16)
        gamma_2 = np.ma.MaskedArray(gamma_2, mask = np.isnan(gamma_2))
        gamma_2_av[measured_resnums] = np.ma.average(gamma_2, axis=0).data
        i_ratio = 10 * np.exp(-gamma_2_av * 0.01) / ( 10 + gamma_2_av )
        np.savetxt(f'{cwd}/{dataset}/{record}/{cycle}/calcPREs/res-{label}.dat',np.c_[x,i_ratio,gamma_2_av])

    return tc_min, calcChi2(cwd, dataset, record, cycle, prot)

def reweightRg(df,record,prot):
    #rg = np.sqrt( np.dot(np.power(prot.rgarray,2), prot.weights) )
    rg = np.dot(prot.rgarray, prot.weights)
    chi2_rg = np.power((prot.expRg-rg)/prot.expRgErr,2)
    #chi2_rg = np.power((prot.expRg-rg)/(prot.expRg*0.03),2)
    return rg, chi2_rg

def calcRg(cwd,dataset, df,record,cycle,prot):
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
    chi2_rg = np.power((prot.expRg-rg)/prot.expRgErr,2)
    return rgarray, rg, chi2_rg

def initProteinsPRE(normal=True, include_PREloss=True):
    proteins = pd.DataFrame(columns=['labels','wh','tau_c','temp','obs','pH','ionic','expPREs','initPREs','eff','chi2_pre','fasta','weights'],dtype=object)
    fasta_OPN = """MHQDHVDSQSQEHLQQTQNDLASLQQTHYSSEENADVPEQPDFPDVPSKSQETVDDDDDDDNDSNDTDESDEVFTDFPTEAPVAPFNRGDNAGRGDSVAYGFRAKAHVVKASKIRKAARKLIEDDATTEDGDSQPAGLWWPKESREQNSRELPQHQSVENDSRPKFDSREVDGGDSKASAGVDSRESQGSVPAVDASNQTLESAEDAEDRHSIENNEVTR""".replace('\n', '')
    fasta_FUS = """MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQSSYSSYGQSQNTGYGTQSTPQGYGSTGGYGSSQSSQSSYGQQSSYPGYGQQPAPSSTSGSYGSSSQSSSYGQPQSGSYSQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS""".replace('\n', '')  # PrLD
    fasta_FUS12E = """GMASNDYEQQAEQSYGAYPEQPGQGYEQQSEQPYGQQSYSGYEQSTDTSGYGQSSYSSYGQEQNTGYGEQSTPQGYGSTGGYGSEQSEQSSYGQQSSYPGYGQQPAPSSTSGSYGSSEQSSSYGQPQSGSYEQQPSYGGQQQSYGQQQSYNPPQGYGQQNQYNS""".replace('\n', '')
    fasta_aSyn = """MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA""".replace('\n', '')
    fasta_A2 = """GHMGRGGNFGFGDSRGGGGNFGPGPGSNFRGGSDGYGSGRGFGDGYNGYGGGPGGGNFGGSPGYGGGRGGYGGGGPGYGNQGGGYGGGYDNYGGGNYGSGNYNDFGNYNQQPSNYGPMKSGNFGGSRNMGGPYGGGNYGPGGSGGSGGYGGRSRY""".replace('\n', '')
    if normal:
        if include_PREloss:
            # proteins.loc['A2'] = dict(labels=[99, 143], tau_c=10.0, wh=850, temp=298, obs='ratio', pH=5.5, fasta=list(fasta_A2), ionic=0.005, weights=False)  # salt concentration is too low
            proteins.loc['aSyn'] = dict(labels=[24, 42, 62, 87, 103], tau_c=1.0, wh=700,temp=283,obs='ratio',pH=7.4,fasta=list(fasta_aSyn),ionic=0.2,weights=False)
            proteins.loc['OPN'] = dict(labels=[10, 33, 64, 88, 117, 130, 144, 162, 184, 203], tau_c=3.0, wh=800,temp=298,obs='rate',pH=6.5,fasta=list(fasta_OPN),ionic=0.15,weights=False)
            proteins.loc['FUS'] = dict(labels=[16, 86, 142], tau_c=10.0, wh=850,temp=298,obs='rate',pH=5.5,fasta=list(fasta_FUS),ionic=0.15,weights=False)
            proteins.loc['FUS12E'] = dict(labels=[16, 86, 142], tau_c=10.0, wh=850,temp=298,obs='rate',pH=5.5,fasta=list(fasta_FUS12E),ionic=0.15,weights=False)
    return proteins

def initIDPsRgs(normal=True, use_newdata=False, validate=False, giulio=True, path2giulio="", francesco=False):
    # copied from rawdata_saxs.py, after this project is published, those new dataset can be used for CALVADOS_SAXS model
    proteins = pd.DataFrame(columns=['temp','expRg','expRgErr','Rg','rgarray','eff','chi2_rg','weights','pH','ionic','fasta'],dtype=object)
    fasta_GHRICD = """SKQQRIKMLILPPVPVPKIKGIDPDLLKEGKLEEVNTILAIHDSYKPEFHSDDSWVEFIELDIDEPDEKTEESDTDRLLSSDHEKSHSNLGVKDGDSGRTSCCEPDILETDFNANDIHEGTSEVAQPQRLKGEADLLCLDQKNQNNSPYHDACPATQQPSVIQAEKNKPQPLPTEGAESTHQAAHIQLSNPSSLSNIDFYAQVSDITPAGSVVLSPGQKNKAGMSQCDMHPEMVSLCQENFLMDNAYFCEADAKKCIPVAPHIKVESHIQPSLNQEDIYITTESLTTAAGRPGTGEHVPGSEMPVPDYTSIHIVQSPQGLILNATALPLPDKEFLSSCGYVSTDQLNKIMP""".replace('\n', '')
    fasta_Ash1 = """SASSSPSPSTPTKSGKMRSRSSSPVRPKAYTPSPRSPNYHRFALDSPPQSPRRSSNSSITKKGSRRSSGSSPTRHTTRVCV"""
    fasta_CTD2 = """FAGSGSNIYSPGNAYSPSSSNYSPNSPSYSPTSPSYSPSSPSYSPTSPCYSPTSPSYSPTSPNYTPVTPSYSPTSPNYSASPQ"""
    fasta_Hst52 = """DSHAKRHHGYKRKFHEKHHSHRGYDSHAKRHHGYKRKFHEKHHSHRGY""".replace('\n', '') # DOI: 10.1021/acs.jpcb.0c09635
    fasta_Hst5 = """DSHAKRHHGYKRKFHEKHHSHRGY""".replace('\n', '')
    fasta_aSyn140 = """MDVFMKGLSKAKEGVVAAAEKTKQGVAEAAGKTKEGVLYVGSKTKEGVVHGVATVAEKTKEQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA""".replace('\n', '')
    fasta_PNt = """DWNNQSIVKTGERQHGIHIQGSDPGGVRTASGTTIKVSGRQAQGILLENPAAELQFRNGSVTSSGQLSDDGIRRFLGTVTVKAGKLVADHATLANVGDTWDDDGIALYVAGEQAQASIADSTLQGAGGVQIERGANVTVQRSAIVDGGLHIGALQSLQPEDLPPSRVVLRDTNVTAVPASGAPAAVSVLGASELTLDGGHITGGRAAGVAAMQGAVVHLQRATIRRGEALAGGAVPGGAVPGGAVPGGFGPGGFGPVLDGWYGVDVSGSSVELAQSIVEAPELGAAIRVGRGARVTVPGGSLSAPHGNVIETGGARRFAPQAAPLSITLQAGAH""".replace('\n','')
    fasta_PNtS1 = """DWNNQSIVKTGERQHGIHIQGSDPGGVRTASGTTIKVSGRQAQGILLENPAAELQFRNGSVTSSGQKSDDGIRRFLGTVTVLAGKLVADHATLANVGDTWDDDGIALYVAGEQAQASIADSTLQGAGGVQIERGANVTVQRSAIVLGGLHIGALQSLQPEDDPPSRVVLRDTNVTAVPASGAPAAVSVLGASLLTLDGGHITGGRAAGVAAMQGAVVHEQRATIRRGEALAGGAVPGGAVPGGAVPGGFGPGGFGPVLDGWYGVDVSGSSVELAQSIVEAPELGAAIRVGRGARVTVPGGSLSAPHGNVIETGGARRFAPQAAPLSITLQAGAH""".replace('\n','')
    fasta_PNtS4 = """DWNNQSIVKTGERQHGIHIQGSDPGGVRTASGTTIKVSGRQAQGILLENPAAELQFRNGSVTSSGQLSFVGITRDLGRDTVKAGKLVADHATLANVGDTWDDDGIALYVAGEQAQASIADSTLQGAGGVQIERGADVRVQREAIVDGGLHNGALQSLQPSILPPSTVVLRDTNVTAVPASGAPAAVLVSGASGLRLDGGHIHEGRAAGVAAMQGAVVTLQTATIRRGEALAGGAVPGGAVPGGAVPGGFGPGGFGPVLDGWYGVDVSGSSVELAQSIVEAPELGAAIRVGRGARVTVPGGSLSAPHGNVIETGGARRFAPQAAPLSITLQAGAH""".replace('\n','')
    fasta_PNtS5 = """DWNNQSIVKTGERQHGIHIQGSDPGGVRTASGTTIKVSGRQAQGILLENPAAELQFRNGSVTSSGQLSDDGIEDFLGTVTVDAGELVADHATLANVGDTWDDDGIALYVAGEQAQASIADSTLQGAGGVQIEDGANVTVQESAIVDGGLHIGALQSLQPRRLPPSRVVLRKTNVTAVPASGAPAAVSVLGASKLTLRGGHITGGRAAGVAAMQGAVVHLQRATIRRGRALAGGAVPGGAVPGGAVPGGFGPGGFGPVLDGWYGVDVSGSSVELAQSIVEAPELGAAIRVGRGARVTVPGGSLSAPHGNVIETGGARRFAPQAAPLSITLQAGAH""".replace('\n','')
    fasta_PNtS6 = """DWNNQSIVKTGERQHGIHIQGSDPGGVRTASGTTIKVSGRQAQGILLENPAAELQFRNGSVTSSGQLSDRGIDRFLGTVTVEAGKLVADHATLANVGDTWDKDGIALYVAGRQAQASIADSTLQGAGGVQIREGANVTVQRSAIVDGGLHIGALQSLQPERLPPSDVVLRDTNVTAVPASGAPAAVSVLGASRLTLDGGHITGGDAAGVAAMQGAVVHLQRATIERGEALAGGAVPGGAVPGGAVPGGFGPGGFGPVLDGWYGVDVSGSSVELAQSIVEAPELGAAIRVGRGARVTVPGGSLSAPHGNVIETGGARRFAPQAAPLSITLQAGAH""".replace('\n','')
    fasta_ACTR = """GTQNRPLLRNSLDDLVGPPSNLEGQSDERALLDQLHTLLSNTDATGLEEIDRALGIPELVNQGQALEPKQD""".replace('\n', '') # DOI: 10.1073/pnas.1322611111
    fasta_RNaseA = """KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV""".replace('\n', '')
    fasta_p15PAF = """MVRTKADSVPGTYRKVVAARAPRKVLGSSTSATNSTSVSSRKAENKYAGGNPVCVRPTPKWQKGIGEFFRLSPKDSEKENQIPEEAGSSGLGKAKRKACPLQPDHTNDEKE""".replace('\n', '') # DOI: 10.1016/j.bpj.2013.12.046
    fasta_CoRNID = """GPHMQVPRTHRLITLADHICQIITQDFARNQVPSQASTSTFQTSPSALSSTPVRTKTSSRYSPESQSQTVLHPRPGPRVSPENLVDKSRGSRPGKSPERSHIPSEPYEPISPPQGPAVHEKQDSMLLLSQRGVDPAEQRSDSRSPGSISYLPSFFTKLESTSPMVKSKKQEIFRKLNSSGGGDSDMAAAQPGTEIFNLPAVTTSGAVSSRSHSFADPASNLGLEDIIRKALMGSFDDKVEDHGVVMSHPVGIMPGSASTSVVTSSEARRDE""".replace('\n', '') # SASDF34
    fasta_Sic1 = """GSMTPSTPPRSRGTRYLAQPSGNTSSSALMQGQKTPQKPSQNLVPVTPSTTKSFKNAPLLAPPNSNMGMTSPFNGLTSPQRSPFPKSSVKRT""".replace('\n', '')
    fasta_FhuA = """SESAWGPAATIAARQSATGTKTDTPIQKVPQSISVVTAEEMALHQPKSVKEALSYTPGVSVGTRGASNTYDHLIIRGFAAEGQSQNNYLNGLKLQGNFYNDAVIDPYMLERAEIMRGPVSVLYGKSSPGGLLNMVSKRPTTEPL""".replace('\n', '')
    fasta_K10 = """MQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL""".replace('\n', '')
    fasta_K27 = """MSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVY""".replace('\n', '')
    fasta_K32 = """MSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIINKKLDLSNVQSKCGSKDNIKHVPGGGSVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVY""".replace('\n', '')
    fasta_K44 = """MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKAEEAGIGDTPSLEDEAAGHVTQARMVSKSKDGTGSDDKKAKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIE""".replace('\n', '')
    fasta_K25 = """MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKAEEAGIGDTPSLEDEAAGHVTQARMVSKSKDGTGSDDKKAKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRL""".replace('\n', '')
    fasta_K23 = """MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKAEEAGIGDTPSLEDEAAGHVTQARMVSKSKDGTGSDDKKAKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLTHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL""".replace('\n', '')
    fasta_A1 = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')  # no deletion
    fasta_M12FP12Y = """GSMASASSSQRGRSGSGNYGGGRGGGYGGNDNYGRGGNYSGRGGYGGSRGGGGYGGSGDGYNGYGNDGSNYGGGGSYNDYGNYNNQSSNYGPMKGGNYGGRSSGGSGGGGQYYAKPRNQGGYGGSSSSSSYGSGRRY""".replace('\n', '')
    fasta_P7FM7Y = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGFGGSGDGFNGFGNDGSNFGGGGSFNDFGNFNNQSSNFGPMKGGNFGGRSSGGSGGGGQFFAKPRNQGGFGGSSSSSSFGSGRRF""".replace('\n', '')
    fasta_M9FP6Y = """GSMASASSSQRGRSGSGNFGGGRGGGYGGNDNYGRGGNYSGRGGFGGSRGGGGYGGSGDGYNGGGNDGSNYGGGGSYNDSGNYNNQSSNFGPMKGGNYGGRSSGGSGGGGQYGAKPRNQGGYGGSSSSSSYGSGRRY""".replace('\n', '')
    fasta_M8FP4Y = """GSMASASSSQRGRSGSGNFGGGRGGGYGGNDNGGRGGNYSGRGGFGGSRGGGGYGGSGDGYNGGGNDGSNYGGGGSYNDSGNYNNQSSNFGPMKGGNYGGRSSGGSGGGGQYGAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_M9FP3Y = """GSMASASSSQRGRSGSGNFGGGRGGGYGGNDNGGRGGNYSGRGGFGGSRGGGGYGGSGDGYNGGGNDGSNYGGGGSYNDSGNGNNQSSNFGPMKGGNYGGRSSGGSGGGGQYGAKPRNQGGYGGSSSSSSYGSGRRS""".replace('\n', '')
    fasta_M10R = """GSMASASSSQGGSSGSGNFGGGGGGGFGGNDNFGGGGNFSGSGGFGGSGGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGSSSGPYGGGGQYFAKPGNQGGYGGSSSSSSYGSGGGF""".replace('\n', '')
    fasta_M6R = """GSMASASSSQGGRSGSGNFGGGRGGGFGGNDNFGGGGNFSGSGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGSSSGPYGGGGQYFAKPGNQGGYGGSSSSSSYGSGGRF""".replace('\n', '')
    fasta_P2R = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFRNDGSNFGGGGRYNDFGNYNNQSSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_P7R = """GSMASASSSQRGRSGRGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGRYGGSGDRYNGFGNDGRNFGGGGSYNDFGNYNNQSSNFGPMKGGNFRGRSSGPYGRGGQYFAKPRNQGGYGGSSSSRSYGSGRRF""".replace('\n', '')
    fasta_M3RP3K = """GSMASASSSQRGKSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSKGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRKF""".replace('\n', '')
    fasta_M6RP6K = """GSMASASSSQKGKSGSGNFGGGRGGGFGGNDNFGKGGNFSGRGGFGGSKGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGKSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRKF""".replace('\n', '')
    fasta_M10RP10K = """GSMASASSSQKGKSGSGNFGGGKGGGFGGNDNFGKGGNFSGKGGFGGSKGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGKSSGGSGGGGQYFAKPKNQGGYGGSSSSSSYGSGKKF""".replace('\n', '')
    fasta_M4D = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNGNFGRGGNFSGRGGFGGSRGGGGYGGSGGGYNGFGNSGSNFGGGGSYNGFGNYNNQSSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')
    fasta_P4D = """GSMASASSSQRDRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGDFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSDPYGGGGQYFAKPRNQGGYGGSSSSSSYDSGRRF""".replace('\n', '')
    fasta_P8D = """GSMASASSSQRDRSGSGNFGGGRDGGFGGNDNFGRGDNFSGRGDFGGSRDGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFGPMKGGNFGGRSSDPYGGGGQYFAKPRNQDGYGGSSSSSSYDSGRRF""".replace('\n', '')
    fasta_P12D = """GSMASADSSQRDRDDSGNFGDGRGGGFGGNDNFGRGGNFSDRGGFGGSRGDGGYGGDGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFDPMKGGNFGDRSSGPYDGGGQYFAKPRNQGGYGGSSSSSSYGSDRRF""".replace('\n', '')
    fasta_P12E = """GSMASAESSQREREESGNFGEGRGGGFGGNDNFGRGGNFSERGGFGGSRGEGGYGGEGDGYNGFGNDGSNFGGGGSYNDFGNYNNQSSNFEPMKGGNFGERSSGPYEGGGQYFAKPRNQGGYGGSSSSSSYGSERRF""".replace('\n', '')
    fasta_P7KP12D = """GSMASADSSQRDRDDKGNFGDGRGGGFGGNDNFGRGGNFSDRGGFGGSRGDGKYGGDGDKYNGFGNDGKNFGGGGSYNDFGNYNNQSSNFDPMKGGNFKDRSSGPYDKGGQYFAKPRNQGGYGGSSSSKSYGSDRRF""".replace('\n', '')
    fasta_P7KP12Db = """GSMASAKSSQRDRDDDGNFGKGRGGGFGGNKNFGRGGNFSKRGGFGGSRGKGKYGGKGDDYNGFGNDGDNFGGGGSYNDFGNYNNQSSNFDPMDGGNFDDRSSGPYDDGGQYFADPRNQGGYGGSSSSKSYGSKRRF""".replace('\n', '')
    fasta_M12FP12YM10R = """GSMASASSSQGGSSGSGNYGGGGGGGYGGNDNYGGGGNYSGSGGYGGSGGGGGYGGSGDGYNGYGNDGSNYGGGGSYNDYGNYNNQSSNYGPMKGGNYGGSSSGPYGGGGQYYAKPGNQGGYGGSSSSSSYGSGGGY""".replace('\n', '')
    fasta_M10FP7RP12D = """GSMASADSSQRDRDDRGNFGDGRGGGGGGNDNFGRGGNGSDRGGGGGSRGDGRYGGDGDRYNGGGNDGRNGGGGGSYNDGGNYNNQSSNGDPMKGGNGRDRSSGPYDRGGQYGAKPRNQGGYGGSSSSRSYGSDRRG""".replace('\n', '')
    fasta_SH4UD = """MGSNKSKPKDASQRRRSLEPAENVHGAGGGAFPASQTPSKPASADGHRGPSAAFAPAAAEPKLFGGFNSSDTVTSPQRAGPLAGGSAWSHPQFEK""".replace('\n', '') # DOI:
    fasta_hNL3cyt = """MYRKDKRRQEPLRQPSPQRGAWAPELGAAPEEELAALQLGPTHHECEAGPPHDTLRLTALPDYTLTLRRSPDDIPLMTPNTITMIPNSLVGLQTLHPYNTFAAGFNSTGLPHSHSTTRV""".replace('\n', '') # DOI: 10.1529/biophysj.107.126995
    fasta_ColNT = """MGSNGADNAHNNAFGGGKNPGIGNTSGAGSNGSASSNRGNSNGWSWSNKPHKNDGFHSDGSYHITFHGDNNSKPKPGGNSGNRGNNGDGASSHHHHHH""".replace('\n', '') # SASDC53
    fasta_tau35 = """EPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIINKKLDLSNVQSKCGSKDNIKHVPGGGSVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL""".replace('\n', '')
    fasta_CAHSD = """MSGRNVESHMERNEKVVVNNSGHADVKKQQQQVEHTEFTHTEVKAPLIHPAPPIISTGAAGLAEEIVGQGFTASAARISGGTAEVHLQPSAAMTEEARRDQERYRQEQESIAKQQEREMEKKTEAYRKTAEAEAEKIRKELEKQHARDVEFRKDLIESTIDRQKREVDLEAKMAKRELDREGQLAKEALERSRLATNVEVNFDSAAGHTVSGGTTVSTSDKMEIKRN""".replace('\n','')
    fasta_p532070 = """GPGSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPGPDEAPRMPEAALEHHHHHH""" # DOI: 10.1038/s41467-021-21258-5

    if normal:
        proteins.loc['tau35'] = dict(temp=293.2,expRg=4.64,expRgErr=0.1,pH=7.4,fasta=list(fasta_tau35),ionic=0.15)  # checked, 6/12/2022
        # proteins.loc['CAHSD'] = dict(temp=293,expRg=4.84,expRgErr=0.2,pH=7.0,fasta=list(fasta_CAHSD),ionic=0.07)  # salt concentration is too low
        proteins.loc['GHRICD'] = dict(temp=298,expRg=6.0,expRgErr=0.5,pH=7.3,fasta=list(fasta_GHRICD),ionic=0.35)
        proteins.loc['p532070'] = dict(eps_factor=0.2,temp=277,expRg=2.39,expRgErr=0.05,pH=7,fasta=list(fasta_p532070),ionic=0.1)
        proteins.loc['Ash1'] = dict(temp=293,expRg=2.9,expRgErr=0.05,pH=7.5,fasta=list(fasta_Ash1),ionic=0.15)
        proteins.loc['CTD2'] = dict(temp=293,expRg=2.614,expRgErr=0.05,pH=7.5,fasta=list(fasta_CTD2),ionic=0.12)
        proteins.loc['ColNT'] = dict(temp=277,expRg=2.8,expRgErr=0.033,pH=7.6,fasta=list(fasta_ColNT),ionic=0.433)
        proteins.loc['hNL3cyt'] = dict(temp=293,expRg=3.15,expRgErr=0.2,pH=8.5,fasta=list(fasta_hNL3cyt),ionic=0.3)
        proteins.loc['SH4UD'] = dict(temp=293.15,expRg=2.71,expRgErr=0.04,pH=8.0,fasta=list(fasta_SH4UD),ionic=0.216)
        proteins.loc['Sic1'] = dict(temp=293,expRg=3.0,expRgErr=0.4,pH=7.5,fasta=list(fasta_Sic1),ionic=0.2)
        proteins.loc['FhuA'] = dict(temp=298,expRg=3.34,expRgErr=0.1,pH=7.5,fasta=list(fasta_FhuA),ionic=0.15)
        proteins.loc['K10'] = dict(temp=288,expRg=4.0,expRgErr=0.1,pH=7.4,fasta=list(fasta_K10),ionic=0.15)
        proteins.loc['K27'] = dict(temp=288,expRg=3.7,expRgErr=0.2,pH=7.4,fasta=list(fasta_K27),ionic=0.15)
        proteins.loc['K25'] = dict(temp=288,expRg=4.1,expRgErr=0.2,pH=7.4,fasta=list(fasta_K25),ionic=0.15)
        proteins.loc['K32'] = dict(temp=288,expRg=4.2,expRgErr=0.3,pH=7.4,fasta=list(fasta_K32),ionic=0.15)
        proteins.loc['K23'] = dict(temp=288,expRg=4.9,expRgErr=0.2,pH=7.4,fasta=list(fasta_K23),ionic=0.15)
        proteins.loc['K44'] = dict(temp=288,expRg=5.2,expRgErr=0.2,pH=7.4,fasta=list(fasta_K44),ionic=0.15)
        proteins.loc['A1'] = dict(temp=298,expRg=2.76,expRgErr=0.02,pH=7.0,fasta=list(fasta_A1),ionic=0.15)
        proteins.loc['M12FP12Y'] = dict(temp=298,expRg=2.60,expRgErr=0.02,pH=7.0,fasta=list(fasta_M12FP12Y),ionic=0.15)
        proteins.loc['P7FM7Y'] = dict(temp=298,expRg=2.72,expRgErr=0.01,pH=7.0,fasta=list(fasta_P7FM7Y),ionic=0.15)
        proteins.loc['M9FP6Y'] = dict(temp=298,expRg=2.66,expRgErr=0.01,pH=7.0,fasta=list(fasta_M9FP6Y),ionic=0.15)
        proteins.loc['M8FP4Y'] = dict(temp=298,expRg=2.71,expRgErr=0.01,pH=7.0,fasta=list(fasta_M8FP4Y),ionic=0.15)
        proteins.loc['M9FP3Y'] = dict(temp=298,expRg=2.68,expRgErr=0.01,pH=7.0,fasta=list(fasta_M9FP3Y),ionic=0.15)
        proteins.loc['M10R'] = dict(temp=298,expRg=2.67,expRgErr=0.01,pH=7.0,fasta=list(fasta_M10R),ionic=0.15)
        proteins.loc['M6R'] = dict(temp=298,expRg=2.57,expRgErr=0.01,pH=7.0,fasta=list(fasta_M6R),ionic=0.15)
        proteins.loc['P2R'] = dict(temp=298,expRg=2.62,expRgErr=0.02,pH=7.0,fasta=list(fasta_P2R),ionic=0.15)
        proteins.loc['P7R'] = dict(temp=298,expRg=2.71,expRgErr=0.01,pH=7.0,fasta=list(fasta_P7R),ionic=0.15)
        proteins.loc['M3RP3K'] = dict(temp=298,expRg=2.63,expRgErr=0.02,pH=7.0,fasta=list(fasta_M3RP3K),ionic=0.15)
        proteins.loc['M6RP6K'] = dict(temp=298,expRg=2.79,expRgErr=0.01,pH=7.0,fasta=list(fasta_M6RP6K),ionic=0.15)
        proteins.loc['M10RP10K'] = dict(temp=298,expRg=2.85,expRgErr=0.01,pH=7.0,fasta=list(fasta_M10RP10K),ionic=0.15)
        proteins.loc['M4D'] = dict(temp=298,expRg=2.64,expRgErr=0.01,pH=7.0,fasta=list(fasta_M4D),ionic=0.15)
        proteins.loc['P4D'] = dict(temp=298,expRg=2.72,expRgErr=0.03,pH=7.0,fasta=list(fasta_P4D),ionic=0.15)
        proteins.loc['P8D'] = dict(temp=298,expRg=2.69,expRgErr=0.01,pH=7.0,fasta=list(fasta_P8D),ionic=0.15)
        proteins.loc['P12D'] = dict(temp=298,expRg=2.80,expRgErr=0.01,pH=7.0,fasta=list(fasta_P12D),ionic=0.15)
        proteins.loc['P12E'] = dict(temp=298,expRg=2.85,expRgErr=0.01,pH=7.0,fasta=list(fasta_P12E),ionic=0.15)
        proteins.loc['P7KP12D'] = dict(temp=298,expRg=2.92,expRgErr=0.01,pH=7.0,fasta=list(fasta_P7KP12D),ionic=0.15)
        proteins.loc['P7KP12Db'] = dict(temp=298,expRg=2.56,expRgErr=0.01,pH=7.0,fasta=list(fasta_P7KP12Db),ionic=0.15)
        proteins.loc['M12FP12YM10R'] = dict(temp=298,expRg=2.61,expRgErr=0.02,pH=7.0,fasta=list(fasta_M12FP12YM10R),ionic=0.15)
        proteins.loc['M10FP7RP12D'] = dict(temp=298,expRg=2.86,expRgErr=0.01,pH=7.0,fasta=list(fasta_M10FP7RP12D),ionic=0.15)
        proteins.loc['Hst5'] = dict(temp=298,expRg=1.38,expRgErr=0.05,pH=7.0,fasta=list(fasta_Hst5),ionic=0.168)
        proteins.loc['Hst52'] = dict(temp=298,expRg=1.9,expRgErr=0.05,pH=7.0,fasta=list(fasta_Hst52),ionic=0.168)
        proteins.loc['aSyn140'] = dict(temp=293,expRg=3.55,expRgErr=0.1,pH=7.4,fasta=list(fasta_aSyn140),ionic=0.2)
        proteins.loc['ACTR'] = dict(temp=278,expRg=2.63,expRgErr=0.1,pH=7.4,fasta=list(fasta_ACTR),ionic=0.2)
        proteins.loc['RNaseA'] = dict(temp=298,expRg=3.36,expRgErr=0.1,pH=7.5,fasta=list(fasta_RNaseA),ionic=0.15)
        proteins.loc['p15PAF'] = dict(temp=298,expRg=2.81,expRgErr=0.1,pH=7.0,fasta=list(fasta_p15PAF),ionic=0.15)
        proteins.loc['CoRNID'] = dict(temp=293.15,expRg=4.7,expRgErr=0.2,pH=7.5,fasta=list(fasta_CoRNID),ionic=0.192)
        proteins.loc['PNt'] = dict(temp=298,expRg=5.11,expRgErr=0.1,pH=7.5,fasta=list(fasta_PNt),ionic=0.15)
        proteins.loc['PNtS1'] = dict(temp=298,expRg=4.92,expRgErr=0.1,pH=7.5,fasta=list(fasta_PNtS1),ionic=0.15)
        proteins.loc['PNtS4'] = dict(temp=298,expRg=5.34,expRgErr=0.1,pH=7.5,fasta=list(fasta_PNtS4),ionic=0.15)
        proteins.loc['PNtS5'] = dict(temp=298,expRg=4.87,expRgErr=0.1,pH=7.5,fasta=list(fasta_PNtS5),ionic=0.15)
        proteins.loc['PNtS6'] = dict(temp=298,expRg=5.26,expRgErr=0.1,pH=7.5,fasta=list(fasta_PNtS6),ionic=0.15)
    if use_newdata:

        # 10.1002/pro.4986
        # SAXS data already included in experimentalSAXS
        fasta_sfAFP = """CKGADGAHGVNGCPGTAGAAGSVGGPGCDGGHGGNGGNGNPGCAGGVGGAGGASGGTGVGGWGGKGGSGTPKGADGAPGAPGSHHWHHHHHH""".replace('\n', '')
        # proteins.loc['sfAFP@50'] = dict(temp=293.15, expRg=None, expRgErr=None, pH=7.5, fasta=list(fasta_sfAFP), ionic=0.101)
        proteins.loc['sfAFP'] = dict(temp=293.15, expRg=2.3, expRgErr=0.11, pH=7.5, fasta=list(fasta_sfAFP), ionic=0.351)

        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0162950#sec023
        # SASDBY5
        fasta_MetC = """GSGGIEGRHAGRQKVQEMKEKFSTIIKAEMPTQSSSPDLPASQAPQQLERIVLYLIENLQKSVDSAETVGGQGMESLMDDGYSSPANTLTLEELAPSPTPALALVPPAPSSVKSSISKSVSVVNVTAARKFQQEHQKQRERDREQLKERTNSTQGVIRQLSSCLSEAETASCILSPASSLSASEAPDTPDPHSNTSPPPSLHTRPSVLHRTLTSTLR""".replace('\n', '')
        proteins.loc['MetC'] = dict(temp=288.15, expRg=5.16, expRgErr=0.56, pH=7.5, fasta=list(fasta_MetC), ionic=0.167)

        # https://academic.oup.com/nar/article/46/1/387/4607802#107883072
        # SASDCY4
        fasta_RNaseE603_850 = """ERQQDRRKPRQNNRRDRNERRDTRSERTEGSDNREENRRNRRQAQQQTAETRESRQQAEVTEKARTADEQQAPRRERSRRRNDDKRQAQQEAKALNVEEQSVQETEQEERVRPVQPRRKQRQLNQKVRYEQSVAEEAVVAPVVEETVAAEPIVQEAPAPRTELVKVPLPVVAQTAPEQQEENNADNRDNGGMPRRSRRSPRHLRVSGQRRRRYRDERYPTQSPMPLTVACASPELASGKVWIRYPIVRHHHHHH""".replace('\n', '')
        proteins.loc['RNaseE603_850'] = dict(temp=288, expRg=5.26, expRgErr=0.031, pH=7.5, fasta=list(fasta_RNaseE603_850), ionic=0.274)  # recalculated Rg and err

        # https://journals.asm.org/doi/10.1128/mBio.00810-20
        # duplicated sequence compared with Giulio's data, but SAXS data can be used later on
        # SASDEX4
        # fasta_UL11StII = """MGLSFSGTRPCCCRNNVLITDDGEVVSLTAHDFDVVDIESEEEGNFYVPPDMRGVTRAPGRQRLRSSDPPSRHTHRRTPGGACPATQFPPPMSDSEWSHPQFEK""".replace('\n', '')
        # proteins.loc['UL11StII'] = dict(temp=277, expRg=2.429, expRgErr=0.2, pH=7.5, fasta=list(fasta_UL11StII), ionic=0.122)

        # https://www.frontiersin.org/articles/10.3389/fmolb.2021.779240/full
        # duplicated sequence compared with Giulio's data, but SAXS data can be used later on
        # SASDLT4
        # SASDLU4
        # fasta_Tau2N3R = """MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKESPLQTPTEDGSEEPGSETSDAKSTPTAEDVTAPLVDEGAPGKQAAAQPHTEIPEGTTAEEAGIGDTPSLEDEAAGHVTQARMVSKSKDGTGSDDKKAKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL""".replace('\n', '')
        # fasta_Tau2N4R = """MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKESPLQTPTEDGSEEPGSETSDAKSTPTAEDVTAPLVDEGAPGKQAAAQPHTEIPEGTTAEEAGIGDTPSLEDEAAGHVTQARMVSKSKDGTGSDDKKAKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIINKKLDLSNVQSKCGSKDNIKHVPGGGSVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL""".replace('\n', '')
        # proteins.loc['Tau2N4R'] = dict(temp=293.2, expRg=6.72, expRgErr=0.28, pH=7.4, fasta=list(fasta_Tau2N4R), ionic=0.15)
        # proteins.loc['Tau2N3R'] = dict(temp=293.2, expRg=6.33, expRgErr=0.28, pH=7.4, fasta=list(fasta_Tau2N3R), ionic=0.15)

        # https://doi.org/10.1016/j.str.2018.10.026
        # duplicated sequence compared with Giulio's data, but SAXS data can be used later on
        # SASDEE2, removed because the salt concentration is not consistent between the reported values in the article and SASBDB website.
        # fasta_ERa_NTD = """SNAMTMTLHTKASGMALLHQIQGNELEPLNRPQLKIPLERPLGEVYLDSSKPAVYNYPEGAAYEFNAAAAANAQVYGQTGLPYGPGSEAAAFGSNGLGGFPPLNSVSPSPLMLLHPPPQLSPFLQPHGQQVPYYLENEPSGYTVREAGPPAFYRPNSDNRRQGGRERLASTNDKGSMAMESAKETRY""".replace('\n', '')
        # proteins.loc['ERa_NTD'] = dict(temp=283.15, expRg=3, expRgErr=0.12, pH=7.4, fasta=list(fasta_ERa_NTD), ionic=0.305)

        # https://www.sciencedirect.com/science/article/pii/S002192582031036X?via%3Dihub
        # duplicated sequence compared with Giulio's data, but SAXS data can be used later on
        # also has PRE data, but not included yet
        # SASDD92
        # fasta_MAP2c = """MADERKDEGKAPHWTSASLTEAAAHPHSPEMKDQGGSGEGLSRSANGFPYREEEEGAFGEHGSQGTYSDTKENGINGELTSADRETAEEVSARIVQVVTAEAVAVLKGEQEKEAQHKDQPAALPLAAEETVNLPPSPPPSPASEQTAALEEATSGESAQAPSAFKQAKDKVTDGITKSPEKRSSLPRPSSILPPRRGVSGDREENSFSLNSSISSARRTTRSEPIRRAGKSGTSTPTTPGSTAITPGTPPSYSSRTPGTPGTPSYPRTPGTPKSGILVPSEKKVAIIRTPPKSPATPKQLRLINQPLPDLKNVKSKIGSTDNIKYQPKGGQVQIVTKKIDLSHVTSKCGSLKNIRHRPGGGRVKIESVKLDFKEKAQAKVGSLDNAHHVPGGGNVKIDSQKLNFREHAKARVDHGAEIITQSPSRSSVASPRRLSNVSSSGSINLLESPQLATLAEDVTAALAKQGL""".replace('\n', '')
        # proteins.loc['MAP2c'] = dict(temp=293.27, expRg=6.59, expRgErr=1.3, pH=6.9, fasta=list(fasta_MAP2c), ionic=0.163)

        # https://academic.oup.com/nar/article/45/21/12170/4129030
        # duplicated sequence compared with Giulio's data, but SAXS data can be used later on
        # SASDC62
        # fasta_Bdomain = """GPPGSMAGGGGSSDGSGRAAGRRASRSSGRARRGRHEPGLGGPAERGAG""".replace('\n', '')
        # proteins.loc['Bdomain'] = dict(temp=277.15, expRg=1.648, expRgErr=0.05, pH=7, fasta=list(fasta_Bdomain), ionic=0.154)

        # https://www.sciencedirect.com/science/article/pii/S2001037021003949?via%3Dihub
        # SASDKD9
        fasta_S4L = """GAMIDLSGLTLQSNAPSSMMVKDEYVHDFEGQPSLSTEGHSIQTIQHPPSNRASTETYSTPALLAPSESNATSTANFPNIPVASTSQPASILGGSHSEGLLQIASGPQPGQQQNGFTGQPATYHHNSTTTWTGS""".replace('\n', '')
        proteins.loc['S4L'] = dict(temp=283.15, expRg=3.6, expRgErr=0.31, pH=7.2, fasta=list(fasta_S4L), ionic=0.169)  # recalculated Rg and err

        # https://www.sciencedirect.com/science/article/pii/S2590028521000259?via%3Dihub
        # SASDL99
        # SASDL89
        # SASDL79
        # SASDL69
        fasta_ED1 = """GSSHHHHHHSSGLVPRGSHMQIVATNLPPEDQDGSGDDSDNFSGSGAGALQDITLSQQTPSTWKDTQLLTAIPTSPEPTGLEATAASTSTLPAGEGPKEGEAVVLPEVEPGLTAREQEATPRPRETTQLPTTHLASTTTATTAQEPATSHPHRDMQPGHHETSTPAGPSQADLHTPHTEDGGPSATERAAEDGASSQLPAAEGSGEQDFTFETSGENTAVVAVEPDRRNQSPVDQGATGASQGLLDRKEVLGDYKDDDDK""".replace('\n', '')
        fasta_ED2 = """GSSHHHHHHSSGLVPRGSHMESRAELTSDKDMYLDNSSIEEASGVYPIDDDDYASASGSGADEDVESPELTTSRPLPKILLTSAAPKVETTTLNIQNKIPAQTKSPEETDKEKVHLSDSERKMDPAEEDTNVYTEKHSDSLFKRTEDYKDDDDK""".replace('\n', '')
        proteins.loc['ED1'] = dict(temp=293.15, expRg=5.42, expRgErr=0.066, pH=7.4, fasta=list(fasta_ED1), ionic=0.153)  # recalculated Rg and err
        proteins.loc['ED2'] = dict(temp=293.15, expRg=4.13, expRgErr=0.15, pH=7.4, fasta=list(fasta_ED2), ionic=0.153)  # recalculated Rg and err


        # NFLt: https://arxiv.org/abs/2212.01894
        fasta_S26_45 = """SAYSGLQSSSYLMSARSFPA""".replace('\n', '')
        fasta_S45_64 = """YYTSHVQEEQTEVEETIEAT""".replace('\n', '')
        fasta_S67_86 = """KAEEAKDEPPSEGEAEEEEK""".replace('\n', '')
        fasta_S66_81 = """ATKAEEAKDEPPSEGEA""".replace('\n', '')
        fasta_S87_105 = """EKEEGEEEEGAEEEEAAKDE""".replace('\n', '')
        fasta_S82_96 = """AEEEEEKEKEEGEEEEGA""".replace('\n', '')
        fasta_S106_128 = """SEDTKEEEEGGEGEEEDTKE""".replace('\n', '')
        fasta_S110_125 = """TKEEEEGGEGEEEDTKES""".replace('\n', '')
        fasta_S129_146 = """SEEEEKKEESAGEEQVAKKKD""".replace('\n', '')
        fasta_S130_143 = """SEEEEKKEESAGEEQV""".replace('\n', '')
        proteins.loc['S26_45'] = dict(temp=298.15, expRg=1.094, expRgErr=0.05, pH=8.0, fasta=list(fasta_S26_45), ionic=0.161)
        proteins.loc['S45_64'] = dict(temp=298.15, expRg=1.221, expRgErr=0.05, pH=8.0, fasta=list(fasta_S45_64), ionic=0.161)
        proteins.loc['S67_86'] = dict(temp=298.15, expRg=1.31, expRgErr=0.05, pH=8.0, fasta=list(fasta_S67_86), ionic=0.161)
        proteins.loc['S66_81'] = dict(temp=298.15, expRg=1.168, expRgErr=0.05, pH=8.0, fasta=list(fasta_S66_81), ionic=0.161)
        proteins.loc['S87_105'] = dict(temp=298.15, expRg=1.309, expRgErr=0.05, pH=8.0, fasta=list(fasta_S87_105), ionic=0.161)
        proteins.loc['S82_96'] = dict(temp=298.15, expRg=1.168, expRgErr=0.05, pH=8.0, fasta=list(fasta_S82_96), ionic=0.161)
        proteins.loc['S106_128'] = dict(temp=298.15, expRg=1.248, expRgErr=0.05, pH=8.0, fasta=list(fasta_S106_128), ionic=0.161)
        proteins.loc['S110_125'] = dict(temp=298.15, expRg=1.164, expRgErr=0.05, pH=8.0, fasta=list(fasta_S110_125), ionic=0.161)
        proteins.loc['S129_146'] = dict(temp=298.15, expRg=1.267, expRgErr=0.05, pH=8.0, fasta=list(fasta_S129_146), ionic=0.161)
        proteins.loc['S130_143'] = dict(temp=298.15, expRg=1.093, expRgErr=0.05, pH=8.0, fasta=list(fasta_S130_143), ionic=0.161)
    if validate or giulio:
        fasta_DSS1 = """MSRAALPSLENLEDDDEFEDFATENWPMKDTELDTGDDTLWENNWDDEDIGDDDFSVQLQAELKKKGVAAC""".replace('\n', '')
        fasta_PTMA = """GPSDAAVDTSSEITTKDLKEKKEVVEEAENGRDAPANGNANEENGEQEADNEVDEEEEEGGEEEEEEEEGDGEEEDGDEDEEAESATGKRAAEDDEDDDVDTKKQKTDEDD""".replace('\n', '')
        fasta_NHE6cmdd = """GPPLTTTLPACCGPIARCLTSPQAYENQEQLKDDDSDLILNDGDISLTYGDSTVNTEPATSSAPRRFMGNSSEDALDRELAFGDHELVIRGTRLVLPMDDSEPPLNLLDNTRHGPA""".replace('\n', '')
        fasta_ANAC046 = """NAPSTTITTTKQLSRIDSLDNIDHLLDFSSLPPLIDPGFLGQPGPSFSGARQQHDLKPVLHHPTTAPVDNTYLPTQALNFPYHSVHNSGSDFGYGAGSGNNNKGMIKLEHSLVSVSQETGLSSDVNTTATPEISSYPMMMNPAMMDGSKSACDGLDDLIFWEDLYTS""".replace('\n', '')
        fasta_p27Cv14 = """GSHMKGACKSSSPPSNDQGRPGDPKQVIDKTEVERTQDTSNIQETQSANNSGPDKPSRCDLAVSGVAAAALPAPGHANSTARDLTRDEEAGSVEQTPKKPGLRRRQT""".replace('\n', '')
        fasta_p27Cv15 = """GSHMKGACIVANSPPDDVKSKEDVPQTDPRLTGGDRDNARASRTGNDPAGASTQSAEVACSNPILSTPDAQEKQAGTSNSKERPHEQLSAGSVEQTPKKPGLRRRQT""".replace('\n', '')
        fasta_p27Cv31 = """GSHMKGACKVPAQESQDVSGSRPAAPLIGAPANSEDTHLVDPKTDPSDSQTGLAEQCAGIRKRPATDDSSTQNKRANRTEENVSDGSPNAGSVEQTPKKPGLRRRQT""".replace('\n', '')
        fasta_p27Cv44 = """GSHMKGACRKPANAEADSSSCQNVPRGKSKQAPETPTGSPLGDATLNQVKPRRPSSASTNIGQLEDADEDDAEDHVGSAVTSQTIPNDRAGSVEQTPKKPGLRRRQT""".replace('\n', '')
        fasta_p27Cv56 = """GSHMKGACGSSVLGTGNPRNQAHVSDTSLEEDDDEQDDSTPDEVSQACTIVASALDINAATPRSPKASPKRKRKRQSTAPAQGNEPPGNAGSVEQTPKKPGLRRRQT""".replace('\n', '')
        fasta_p27Cv78 = """GSHMKGACALPSGVVPAEDDDDDEEEEDDQDPAQPQAVQGAAPSSGTNNSQPILPSIAVNSTTGPNSTAGKKKRKRRRTRHSNCATLSSAGSVEQTPKKPGLRRRQT""".replace('\n', '')
        fasta_A1S = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGNYNNQSSNFGPMKGGNFGGRSSGPYGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')  # after deletion
        # fasta_D91FATZ1 = """GPTVGGQLGTAGQGFSYSKSNGRGGSQAGGSGSAGQYGSDQQHHLGSGSGAGGTGGPAGQAGRGGAAGTAGVGETGSGDQAGGEGKHITVFKTYISPWERAMGVDPQQKMELGIDLLAYGAKAELPKYKSFNRTAMPYGGYEKASKRMTFQMPKFDLGPLLSEPLVLYNQNLSNRPSFNRTPIPWLSSGEPVDYNVDIGIPLDGETEEL""".replace('\n', '')
        # fasta_NFATZ1 = """MAHHHHHHVDDDDKIMPLSGTPAPNKKRKSSKLIMELTGGGQESSGLNLGKKISVPRDVMLEELSLLTNRGSKMFKLRQMRVEKFIYENHPDVFSDSSMDHFQKFLPTVGGQLGTAGQGFSYSKSNGRGGSQAGGSGSAGQYGSDQQHHLGSGSGAGGTGGPAGQAGRGGAAGTAGVGETGSGDQAGGEAE""".replace('\n', '')
        fasta_ChiZ164 = """SNAMTPVRPPHTPDPLNLRGPLDGPRWRRAEPAQSRRPGRSRPGGAPLRYHRTGVGMSRTGHGSRPV"""  # DOI: 10.3390/biom10060946
        fasta_cDAXX = """SPMSSLQISNEKNLEPGKQISRSSGEQQNKGRIVSPSLLSEEPLAPSSIDAESNGEQPEELTLEEESPVSQLFELEIEALPLDTPSSVETDISSSRKQSEEPFTTVLENGAGMVSSTSFNGGVSPHNWGDSGPPCKKSRKEKKQTGSGPLGNSYVERQRSVHEKNGKKICTLPSPPSPLASLAPVADSSTRVDSPSHGLVTSSLCIPSPARLSQTPHSQPPRPGTCKTSVATQCDPEEIIVLSDSD""".replace('\n', '')
        proteins.loc['cDAXX'] = dict(temp=293, expRg=4.75, expRgErr=0.05, pH=8.0, fasta=list(fasta_cDAXX), ionic=0.13)
        # proteins.loc['ChiZ164'] = dict(temp=293, expRg=2.42, expRgErr=0.01, pH=7.0, fasta=list(fasta_ChiZ164), ionic=0.065)  # salt concentration is too low
        # proteins.loc['D91FATZ1'] = dict(temp=293, expRg=4.0, expRgErr=0.1, pH=7.5, fasta=list(fasta_D91FATZ1), ionic=0.18)  # duplicated
        # proteins.loc['NFATZ1'] = dict(temp=293, expRg=3.6, expRgErr=0.1, pH=7.5, fasta=list(fasta_NFATZ1), ionic=0.18)
        # proteins.loc['A1S@S50'] = dict(temp=293, expRg=2.645, expRgErr=0.02, pH=7.5, fasta=list(fasta_A1S), ionic=0.05)
        proteins.loc['A1S@S150'] = dict(temp=293, expRg=2.65, expRgErr=0.02, pH=7.5, fasta=list(fasta_A1S), ionic=0.15)
        # proteins.loc['A1S@S300'] = dict(temp=293, expRg=2.62, expRgErr=0.02, pH=7.5, fasta=list(fasta_A1S), ionic=0.3)
        # proteins.loc['A1S@S500'] = dict(temp=293, expRg=2.528, expRgErr=0.02, pH=7.5, fasta=list(fasta_A1S), ionic=0.5)
        proteins.loc['p27Cv14'] = dict(temp=293, expRg=2.936, expRgErr=0.13, pH=7.2, fasta=list(fasta_p27Cv14), ionic=0.095)
        proteins.loc['p27Cv15'] = dict(temp=293, expRg=2.915, expRgErr=0.1, pH=7.2, fasta=list(fasta_p27Cv15), ionic=0.095)
        proteins.loc['p27Cv31'] = dict(temp=293, expRg=2.81, expRgErr=0.18, pH=7.2, fasta=list(fasta_p27Cv31), ionic=0.095)
        proteins.loc['p27Cv44'] = dict(temp=293, expRg=2.492, expRgErr=0.13, pH=7.2, fasta=list(fasta_p27Cv44), ionic=0.095)
        proteins.loc['p27Cv56'] = dict(temp=293, expRg=2.328, expRgErr=0.1, pH=7.2, fasta=list(fasta_p27Cv56), ionic=0.095)
        proteins.loc['p27Cv78'] = dict(temp=293, expRg=2.211, expRgErr=0.03, pH=7.2, fasta=list(fasta_p27Cv78), ionic=0.095)
        proteins.loc['DSS1'] = dict(temp=288, expRg=2.5, expRgErr=0.1, pH=7.4, fasta=list(fasta_DSS1), ionic=0.17)
        proteins.loc['PTMA'] = dict(temp=288, expRg=3.7, expRgErr=0.2, pH=7.4, fasta=list(fasta_PTMA), ionic=0.16)
        proteins.loc['NHE6cmdd'] = dict(temp=288, expRg=3.2, expRgErr=0.2, pH=7.4, fasta=list(fasta_NHE6cmdd), ionic=0.17)
        proteins.loc['ANAC046'] = dict(temp=298, expRg=3.6, expRgErr=0.3, pH=7.0, fasta=list(fasta_ANAC046), ionic=0.14)
        # https://europepmc.org/article/MED/30878202
        # PED00181
        fasta_DomainV = """GAMGTAGNKAALLDQIREGAQLKKVEQNSRPVSCSGRDALLDQIRQGIQLKSVADGQESTPPTPAPT""".replace('\n', '')
        proteins.loc['DomainV'] = dict(temp=288.15, expRg=2.43, expRgErr=0.024, pH=7.0, fasta=list(fasta_DomainV), ionic=0.1985)

        # https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0172507
        # PED00142
        fasta_hKISS1 = """GEPLEKVASVGNSRPTGQQLESLGLLAPGEQSLPCTERKPAATARLSRRGTSLSPPPESSGSPQQPGLSAPHSRQIPAPQGAVLVQREKDLPNYNWNSFGLRFGKREAAPGNHGRSAGRG""".replace('\n', '')
        proteins.loc['hKISS1'] = dict(temp=283.15, expRg=3.47, expRgErr=0.05, pH=7.0, fasta=list(fasta_hKISS1), ionic=0.159)

        # 10.1038/s41598-017-15299-4 or https://www.nature.com/articles/s41598-017-15299-4
        # PED00141
        fasta_HvASR1 = """GSPEFMAEEKHHHHLFHHKKEGEDFQPAADGGADVYGYSTETVVTGTGNEGEYERITKEEKHHKHKEHLGEMGAVAAGAFALYEKHEAKKDPEHAHKHKIEEELAAAAAVGAGGFVFHEHHEKKQDHKEAKEASGEKKHHLFG""".replace('\n', '')
        fasta_TtASR1 = """GSPEFMAEEKHHHLFHHKEGEDFQPAADGGVDTYGYSTETVVTATGNDGEYERITKEEKHHKHKEHLGEMGAAAAGAFALYEKHEAKKDPEHAHKHKIEEEVAAAAAVGAGGFVFHEHHEKKQDHKEAKEASGEKKHHLFG""".replace('\n', '')
        proteins.loc['HvASR1'] = dict(temp=293.15, expRg=3.51, expRgErr=0.09, pH=7.3, fasta=list(fasta_HvASR1), ionic=0.15)
        proteins.loc['TtASR1'] = dict(temp=293.15, expRg=3.31, expRgErr=0.08, pH=7.3, fasta=list(fasta_TtASR1), ionic=0.15)
        # https://doi.org/10.1016/j.jmb.2021.166954
        # SASDKT8, 3mg/ml
        fasta_VWF = """MEDREQAPNLVYMVTGNPASDEIKRLPGDIQVVPIGVGPNANVQELERIGWPNAPILIQDFETLPREAPDLVLQRGGWSHPQFEKGGGSGGGSGGWSHPQFEK""".replace('\n', '')
        proteins.loc['VWF'] = dict(temp=293, expRg=3.08, expRgErr=0.03, pH=7.4, fasta=list(fasta_VWF), ionic=0.153)

        # https://doi.org/10.1038/s41467-019-11951-x
        # SASDFK8
        fasta_GON7 = """MGHHHHHHENLYFQGELLGEYVGQEGKPQKLRVSCEAPGDGDPFQGLLSGVAQMKDMVTELFDPLVQGEVQHRVAAAPDEDLDGDDEDDAEDENNIDNRTNFDGPSAKRPKTPS""".replace('\n', '')
        proteins.loc['GON7'] = dict(temp=283, expRg=3.18, expRgErr=0.04, pH=6.5, fasta=list(fasta_GON7), ionic=0.211)

        # https://doi.org/10.1016/j.jmb.2021.166899
        # SASDK68
        fasta_TIF2NRID = """ERADGQSRLHDSKGQTKLLQLLTTKSDQMEPSPLASSLSDTNKDSTGSLPGSGSTHGTSLKEKHKILHRLLQDSSSPVDLAKLTAEATGKDLSQESSSTAPGSEVTIKQEPVSPKKKENALLRYLLDKDDTKDIGLPEITPKLERLDSKT""".replace('\n', '')
        proteins.loc['TIF2NRID'] = dict(temp=283.15, expRg=3.74, expRgErr=0.092, pH=6.8, fasta=list(fasta_TIF2NRID), ionic=0.175)

        # https://doi.org/10.1016/j.jbc.2022.102631
        # SASDJS7
        # it also has data under 6.5, but it's redundant
        fasta_PARCL = """GSMQYYENREKDYYEVAQGQRNGYGQSQSHNHEGYGQSQSRGGYGQIHNREGYNQNREGYSQSQSRPVYGLSPTLNHRSHGGFLDGLFKGQNGQKGQSGLGTFLGQHKSQEAKKSQGHGKLLGQHDQKKTHETNSGLNGLGMFINNGEKKHRRKSEHKKKNKDGHGSGNESGSSSGSDSD""".replace('\n', '')
        proteins.loc['PARCL'] = dict(temp=293.15, expRg=3.43, expRgErr=0.065, pH=7.5, fasta=list(fasta_PARCL), ionic=0.17)

        # https://www.jbc.org/article/S0021-9258(20)30509-3/fulltext
        # SASDF27
        fasta_BMAL1P624A = """GPDASSPGGKKILNGGTPDIPSTGLLPGQAQETPGYPYSDSSSILGENPHIGIDMIDNDQGSSSPSNDEAAMAVIMSLLEADAGLGGPVDFSDLPWAL""".replace('\n', '')
        proteins.loc['BMAL1P624A'] = dict(temp=283.25, expRg=2.77, expRgErr=0.09, pH=7.2, fasta=list(fasta_BMAL1P624A), ionic=0.154)

        # https://www.science.org/doi/10.1126/sciadv.abg7653
        # SASDJJ6
        # SASDJK6
        fasta_N_FATZ1 = """MAHHHHHHVDDDDKIMPLSGTPAPNKKRKSSKLIMELTGGGQESSGLNLGKKISVPRDVMLEELSLLTNRGSKMFKLRQMRVEKFIYENHPDVFSDSSMDHFQKFLPTVGGQLGTAGQGFSYSKSNGRGGSQAGGSGSAGQYGSDQQHHLGSGSGAGGTGGPAGQAGRGGAAGTAGVGETGSGDQAGGEAE""".replace('\n', '')
        fasta_D91_FATZ1 = """GPTVGGQLGTAGQGFSYSKSNGRGGSQAGGSGSAGQYGSDQQHHLGSGSGAGGTGGPAGQAGRGGAAGTAGVGETGSGDQAGGEGKHITVFKTYISPWERAMGVDPQQKMELGIDLLAYGAKAELPKYKSFNRTAMPYGGYEKASKRMTFQMPKFDLGPLLSEPLVLYNQNLSNRPSFNRTPIPWLSSGEPVDYNVDIGIPLDGETEEL""".replace('\n', '')
        proteins.loc['D91_FATZ1'] = dict(temp=293.15, expRg=3.86, expRgErr=0.024, pH=7.5, fasta=list(fasta_D91_FATZ1), ionic=0.192)
        proteins.loc['N_FATZ1'] = dict(temp=293.15, expRg=3.45, expRgErr=0.062, pH=7.5, fasta=list(fasta_N_FATZ1), ionic=0.192)

        # https://www.sciencedirect.com/science/article/pii/S2590028521000259?via%3Dihub
        # SASDL99  ED4
        # SASDL89  ED3
        # SASDL79  ED2
        # SASDL69  ED1
        fasta_ED3 = """MGSSHHHHHHSSGLVPRGSMAQRWRSENFERPVDLEGSGDDDSFPDDELDDLYSGSGSGYFEQESGIETAMETRFSPDVALAVSTTPAVLPTTNIQPVGTPFEELPSERPTLEPATSPLVVTEVPEEPSQRATTVSTTMETATTAATSTGDPTVATVPATVATATPSTPAAPPFTATTAVIRTTGVRRLLPLPLTTVATARATTPEAPSPPTTAAVLDTEAPTPRLVSTATSRPRALPRPATTQEPDIPERSTLPLGTTAPGPTEVAQTPTPETFLTTIRDEPEVPVSGGPSGDFELPEEETTQPDTANEVVAVGGAAAKASSPPGTLPKGARPGPGLLDNAIDSGSSAAQLPQKSILERKEVLVDYKDDDDK""".replace('\n', '')
        fasta_ED4 = """GSSHHHHHHSSGLVPRGSHMESIRETEVIDPQDLLEGRYFSGALPDDEDVVGPGQESDDFELSGSGDLDDLEDSMIGPEVVHPLVPLDNHIPERAGSGSQVPTEPKKLEENEVIPKRISPVEESEDVSNKVSMSSTVQGSNIFERTEVLAGCPEHDYKDDDDK""".replace('\n', '')
        proteins.loc['ED3'] = dict(temp=293.15, expRg=6.51, expRgErr=0.15, pH=7.4, fasta=list(fasta_ED3), ionic=0.153)  # recalculated Rg and err
        proteins.loc['ED4'] = dict(temp=293.15, expRg=4.06, expRgErr=0.11, pH=7.4, fasta=list(fasta_ED4), ionic=0.153)  # recalculated Rg and err

        # 10.1002/pro.4986
        fasta_PNtdeltaPG = """DWNNQSIVKTGERQHGIHIQGSDPGGVRTASGTTIKVSGRQAQGILLENPAAELQFRNGSVTSSGQLSDDGIRRFLGTVTVKAGKLVADHATLANVGDTWDDDGIALYVAGEQAQASIADSTLQGAGGVQIERGANVTVQRSAIVDGGLHIGALQSLQPEDLPPSRVVLRDTNVTAVPASGAPAAVSVLGASELTLDGGHITGGRAAGVAAMQGAVVHLQRATIRRGDALAPVLDGWYGVDVSGSSVELAQSIVEAPELGAAIRVGRGARVTVPGGSLSAPHGNVIETGGARRFAPQAAPLSITLQAGAH""".replace('\n', '')
        proteins.loc['PNtdeltaPG'] = dict(temp=293.15, expRg=4.99, expRgErr=0.18, pH=7.5, fasta=list(fasta_PNtdeltaPG), ionic=0.175)

    if giulio:
        rg_filtered_data_Giulio = pd.read_csv(path2giulio).set_index("Unnamed: 0", drop=True)
        rg_filtered_data_Giulio["fasta"] = rg_filtered_data_Giulio["fasta"].apply(lambda x: list(x))

        proteins = pd.concat((proteins, rg_filtered_data_Giulio))

    if francesco:  # for validation
        fasta_V2 = """GSGGYGSSQGGFFGGGDAGGNGDGSDFGGGYPSGSNQNSGGFSGYGNDSFQGSAGMFNGFKSASKFSNSGGYGGGGQGNNNGSGGGSSFRNRRRRSNYSGGGSGRGRRYGSNFGGMYGGRSGFGGNGPGRSGFGGSN""".replace('\n', '')
        proteins.loc['V2'] = dict(temp=298, expRg=2.31, expRgErr=0.11, pH=7.0, fasta=list(fasta_V2), ionic=0.15)
        fasta_V3 = """GSKQGGRGGNRSGSGNGNASGAGGGGRDGGSDGGFDGFDYQFSGGGNPSSQYYGSRGGSGRNSAGGYYFFRNSSGGNGSSGNMNPGNGYFGFSRSGGRGQNRGFFFGGMGGGGFGRSSNFGSYNSSNKSGSGGGGGG""".replace('\n', '')
        proteins.loc['V3'] = dict(temp=298, expRg=2.35, expRgErr=0.085, pH=7.0, fasta=list(fasta_V3), ionic=0.15)
        fasta_V4 = """GSGSNGGGSQSSGQGYGKSGGNRRRGRGGAGGGFGMGDGSNQYGYGPFRRGSGFNGNGDYANYGGNGDSNNFSNYRGGNSANGNFQSGGGGGFDNGGGSGFGGSFSMSGGSSSGKRRGSGGFFSGRSGSGFGGFYPS""".replace('\n', '')
        proteins.loc['V4'] = dict(temp=298, expRg=2.39, expRgErr=0.087, pH=7.0, fasta=list(fasta_V4), ionic=0.15)
        fasta_V5 = """GSGFSNMGNGFGGRFGGGRGFSRYSQQFSYYDGGQSSGGNGSSGGFNSYGGYNNGRNGSSFGGAGGGGRSSFGFSGGGGFGADGGYNRFSSGDRNNNGPSKGGGGGNGSGSRGFAGNGSMSDRGNSYGGGPGRQKGS""".replace('\n', '')
        proteins.loc['V5'] = dict(temp=298, expRg=2.48, expRgErr=0.14, pH=7.0, fasta=list(fasta_V5), ionic=0.15)
        # https://www.nature.com/articles/s41586-024-08400-1
        # SAXS data already included in experimentalSAXS
        # SASDSK2
        fasta_ER_NTD = """SNAMTMTLHTKASGMALLHQIQGNELEPLNRPQLKIPLERPLGEVYLDSSKPAVYNYPEGAAYEFNAAAAANAQVYGQTGLPYGPGSEAAAFGSNGLGGFPPLNSVSPSPLMLLHPPPQLSPFLQPHGQQVPYYLENEPSGYTVREAGPPAFYRPNSDNRRQGGRERLASTNDKGSMAMESAKETRY""".replace('\n', '')
        proteins.loc['ER_NTD'] = dict(temp=277.15,expRg=3.07,expRgErr=0.2,pH=7.4,fasta=list(fasta_ER_NTD),ionic=0.198)

        # https://www.nature.com/articles/s41594-024-01339-x
        # SASDU67
        # SAXS data already included in experimentalSAXS
        fasta_CTR_XRCC4 = """GSAAQEREKDIKQEGETAICSEMTADRDPVYDESTDEESENQTDLSGLASAAVSKDDSIISSLDVTDIAPSRKRRQRMQRNLGTEPKMAPQENQLQEKENSRPDSSLPETSKKEHISAENMSLETLRNSSPEDLFDEI""".replace('\n', '')
        proteins.loc['CTR_XRCC4'] = dict(temp=298,expRg=3.42,expRgErr=0.15,pH=6.5,fasta=list(fasta_CTR_XRCC4),ionic=0.16)

        # https://link.springer.com/article/10.1140/epje/s10189-024-00409-8
        # SASDT47
        # SAXS data already included in experimentalSAXS, aggregation?
        # fasta_Nlp441_543 = """MGCGPAYYNSHVQEEQTEVEETIEATKAEEAKDEPPSEGEAEEEEKEKEEGEEEEGAEEEEAAKDESEDTKEEEEGGEGEEEDTKESEEEEKKEESAGEEQVAKKKDGCG""".replace('\n', '')
        # proteins.loc['Nlp441_543'] = dict(temp=297.15,expRg=4.07,expRgErr=3.8,pH=8.0,fasta=list(fasta_Nlp441_543),ionic=0.261)

        # https://www.nature.com/articles/s42003-024-05856-9
        # SASDKH8, it could probably form some transient secondary structures, if the predictions are not good, just remove this protein;
        # SAXS data already included in experimentalSAXS
        fasta_CTir = """GPRRNQPAEQTTTTTTHTVVQQQTGGNTPAQGGTDATRAEDASLNRRDSQGSVASTHWSDSSSEVVNPYAEVGGARNSLSAHQPEEHIYDEVAADPGYSVIQNFSGSGPVTGRLIGTPGQGIQSTYALLANSGGLRLGMGGLTSGGESAVSSVNAAPTPGPVRFVWSHPQFEK""".replace('\n', '')
        proteins.loc['CTir'] = dict(temp=298.15,expRg=3.81,expRgErr=0.043,pH=6.5,fasta=list(fasta_CTir),ionic=0.179)

        # https://www.nature.com/articles/s41467-023-39808-4
        # SASDQM8, they also have some deletion proteins with SAXS data one can use;
        # SAXS data already included in experimentalSAXS
        fasta_TRPV4 = """ADPEDPRDAGDVLGDDSFPLSSLANLFEVEDTPSPAEPSRGPPGAVDGKQNLRMKFHGAFRKGPPKPMELLESTIYESSVVPAPKKAPMDSLFDYGTYRQHPSENKRWRRRVVEKPVAGTKGPAPNPPPILKV""".replace('\n', '')
        proteins.loc['TRPV4'] = dict(temp=293.15,expRg=3.2,expRgErr=0.26,pH=7.0,fasta=list(fasta_TRPV4),ionic=0.118)

        # https://www.nature.com/articles/s41467-023-36402-6
        # SASDKD6, the article didn't mention this protein in the main text; Remove it if the predictions are not good;
        # SAXS data already included in experimentalSAXS
        fasta_Red1 = """GAMGISLPLLKQDDWLSSSKPFGSSTPNVVIEFDSDDDGDDFSNSKIEQSNLEKPPSNSENGGSHHHHHH""".replace('\n', '')
        proteins.loc['Red1'] = dict(temp=293.15,expRg=2.43,expRgErr=0.18,pH=7.5,fasta=list(fasta_Red1),ionic=0.157)

        # https://pubs.acs.org/doi/10.1021/jacsau.4c00673
        # no SAXS data available yet, only Rg data; included in stickness project together with other proteins below
        fasta_dChMINUS_Rg = """GSGSCMGLPTGMEEKEEGTDESEQKPVVQTPAQPDDSAEVDSAALDQAESAKQQGPILTKHGCTLGPR""".replace('\n', '')
        proteins.loc['dChMINUS_Rg'] = dict(temp=295,expRg=2.58,expRgErr=0.05,pH=7.3,fasta=list(fasta_dChMINUS_Rg),ionic=0.172)
        fasta_sNhPLUS_Rg = """GSGSCPEEIETRKKDRKNRREMLKRTSALQPAAPKPTHKKPVPKRNVGAERKSTINEDLLPPCTLGPR""".replace('\n', '')
        proteins.loc['sNhPLUS_Rg'] = dict(temp=295,expRg=2.55,expRgErr=0.07,pH=7.3,fasta=list(fasta_sNhPLUS_Rg),ionic=0.172)

        rg_giulio_updated = pd.read_csv('rg_giulio_updated.csv',index_col=0)
        # remove "Eralpha_NTD", "Nup153_82", "ERNTD_S118A", "ERNTD_S118D"
        tmp = rg_giulio_updated.loc[['ERNTD_WT', 'HMPVP_NTD', 'SMAD4_linker', 'SMAD2_linker', 'NLS_Rg', 'IBB_Rg', 'NUL_Rg', 'NUS_Rg', 'Nsp1_Rg', 'Nup49_Rg', 'N98_Rg'], ['temp', 'expRg', 'expRgErr', 'pH', 'fasta', 'ionic']]
        tmp['fasta'] = tmp['fasta'].apply(lambda x: list(x))
        proteins = pd.concat((proteins, tmp))


    return proteins

def initIDPsFRET(validate=False):
    proteins = pd.DataFrame(columns=['temp','expRg','expRgErr','Rg','rgarray','eff','chi2_rg','weights','pH','ionic','fasta'],dtype=object)
    # Sic1 is already included in initIDPsRgs;
    if validate:
        fasta_NLS = """ACETNKRKREQISTDNEAKMQIQEEKSPKKKRKKRSSKANKPPECA""".replace('\n', '')
        proteins.loc['NLS'] = dict(temp=296, expRg=2.4, expRgErr=0.3, pH=7.4, fasta=list(fasta_NLS), ionic=0.15)
        fasta_NUS = """GCPSASPAFGANQTPTFGQSQGASQPNPPGFGSISSSTALFPTGSQPAPPTFGTVSSSSQPPVFGQQPSQSAFGSGTTPNCA""".replace('\n', '')
        proteins.loc['NUS'] = dict(temp=296, expRg=2.5, expRgErr=0.1, pH=7.4, fasta=list(fasta_NUS), ionic=0.15)
        fasta_IBB = """GCTNENANTPAARLHRFKNKGKDSTEMRRRRIEVNVELRKAKKDDQMLKRRNVSSFPDDATSPLQENRNNQGTVNWSVDDIVKGINSSNVENQLQATCA""".replace('\n', '')
        proteins.loc['IBB'] = dict(temp=296, expRg=3.2, expRgErr=0.2, pH=7.4, fasta=list(fasta_IBB), ionic=0.15)
        fasta_NUL = """GCGFKGFDTSSSSSNSAASSSFKFGVSSSSSGPSQTLTSTGNFKFGDQGGFKIGVSSDSGSINPMSEGFKFSKPIGDFKFGVSSESKPEEVKKDSKNDNFKFGLSSGLSNPVCA""".replace('\n', '')
        proteins.loc['NUL'] = dict(temp=296, expRg=3.0, expRgErr=0.3, pH=7.4, fasta=list(fasta_NUL), ionic=0.15)
        fasta_Nup49 = """GCQTSRGLFGNNNTNNINNSSSGMNNASAGLFGSKPCA""".replace('\n', '')
        proteins.loc['Nup49'] = dict(temp=296, expRg=1.6, expRgErr=0.1, pH=7.4, fasta=list(fasta_Nup49), ionic=0.15)
    return proteins