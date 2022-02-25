import warnings
import torch
from Bio import BiopythonWarning
from Bio.PDB import Selection
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import three_to_one, three_to_index, is_aa


NON_STANDARD_SUBSTITUTIONS = {
    '2AS':'ASP', '3AH':'HIS', '5HP':'GLU', 'ACL':'ARG', 'AGM':'ARG', 'AIB':'ALA', 'ALM':'ALA', 'ALO':'THR', 'ALY':'LYS', 'ARM':'ARG',
    'ASA':'ASP', 'ASB':'ASP', 'ASK':'ASP', 'ASL':'ASP', 'ASQ':'ASP', 'AYA':'ALA', 'BCS':'CYS', 'BHD':'ASP', 'BMT':'THR', 'BNN':'ALA',
    'BUC':'CYS', 'BUG':'LEU', 'C5C':'CYS', 'C6C':'CYS', 'CAS':'CYS', 'CCS':'CYS', 'CEA':'CYS', 'CGU':'GLU', 'CHG':'ALA', 'CLE':'LEU', 'CME':'CYS',
    'CSD':'ALA', 'CSO':'CYS', 'CSP':'CYS', 'CSS':'CYS', 'CSW':'CYS', 'CSX':'CYS', 'CXM':'MET', 'CY1':'CYS', 'CY3':'CYS', 'CYG':'CYS',
    'CYM':'CYS', 'CYQ':'CYS', 'DAH':'PHE', 'DAL':'ALA', 'DAR':'ARG', 'DAS':'ASP', 'DCY':'CYS', 'DGL':'GLU', 'DGN':'GLN', 'DHA':'ALA',
    'DHI':'HIS', 'DIL':'ILE', 'DIV':'VAL', 'DLE':'LEU', 'DLY':'LYS', 'DNP':'ALA', 'DPN':'PHE', 'DPR':'PRO', 'DSN':'SER', 'DSP':'ASP',
    'DTH':'THR', 'DTR':'TRP', 'DTY':'TYR', 'DVA':'VAL', 'EFC':'CYS', 'FLA':'ALA', 'FME':'MET', 'GGL':'GLU', 'GL3':'GLY', 'GLZ':'GLY',
    'GMA':'GLU', 'GSC':'GLY', 'HAC':'ALA', 'HAR':'ARG', 'HIC':'HIS', 'HIP':'HIS', 'HMR':'ARG', 'HPQ':'PHE', 'HTR':'TRP', 'HYP':'PRO',
    'IAS':'ASP', 'IIL':'ILE', 'IYR':'TYR', 'KCX':'LYS', 'LLP':'LYS', 'LLY':'LYS', 'LTR':'TRP', 'LYM':'LYS', 'LYZ':'LYS', 'MAA':'ALA', 'MEN':'ASN',
    'MHS':'HIS', 'MIS':'SER', 'MLE':'LEU', 'MPQ':'GLY', 'MSA':'GLY', 'MSE':'MET', 'MVA':'VAL', 'NEM':'HIS', 'NEP':'HIS', 'NLE':'LEU',
    'NLN':'LEU', 'NLP':'LEU', 'NMC':'GLY', 'OAS':'SER', 'OCS':'CYS', 'OMT':'MET', 'PAQ':'TYR', 'PCA':'GLU', 'PEC':'CYS', 'PHI':'PHE',
    'PHL':'PHE', 'PR3':'CYS', 'PRR':'ALA', 'PTR':'TYR', 'PYX':'CYS', 'SAC':'SER', 'SAR':'GLY', 'SCH':'CYS', 'SCS':'CYS', 'SCY':'CYS',
    'SEL':'SER', 'SEP':'SER', 'SET':'SER', 'SHC':'CYS', 'SHR':'LYS', 'SMC':'CYS', 'SOC':'CYS', 'STY':'TYR', 'SVA':'SER', 'TIH':'ALA',
    'TPL':'TRP', 'TPO':'THR', 'TPQ':'ALA', 'TRG':'LYS', 'TRO':'TRP', 'TYB':'TYR', 'TYI':'TYR', 'TYQ':'TYR', 'TYS':'TYR', 'TYY':'TYR'
}

RESIDUE_SIDECHAIN_POSTFIXES = {
    'A': ['B'],
    'R': ['B', 'G', 'D', 'E', 'Z', 'H1', 'H2'],
    'N': ['B', 'G', 'D1', 'D2'],
    'D': ['B', 'G', 'D1', 'D2'],
    'C': ['B', 'G'],
    'E': ['B', 'G', 'D', 'E1', 'E2'],
    'Q': ['B', 'G', 'D', 'E1', 'E2'],
    'G': [],
    'H': ['B', 'G', 'D1', 'D2', 'E1', 'E2'],
    'I': ['B', 'G1', 'G2', 'D1'],
    'L': ['B', 'G', 'D1', 'D2'],
    'K': ['B', 'G', 'D', 'E', 'Z'],
    'M': ['B', 'G', 'D', 'E'],
    'F': ['B', 'G', 'D1', 'D2', 'E1', 'E2', 'Z'],
    'P': ['B', 'G', 'D'],
    'S': ['B', 'G'],
    'T': ['B', 'G1', 'G2'],
    'W': ['B', 'G', 'D1', 'D2', 'E1', 'E2', 'E3', 'Z2', 'Z3', 'H2'],
    'Y': ['B', 'G', 'D1', 'D2', 'E1', 'E2', 'Z', 'H'],    
    'V': ['B', 'G1', 'G2'],
}

GLY_INDEX = 5
ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4



def augmented_three_to_one(three):
    if three in NON_STANDARD_SUBSTITUTIONS:
        three = NON_STANDARD_SUBSTITUTIONS[three]
    return three_to_one(three)


def augmented_three_to_index(three):
    if three in NON_STANDARD_SUBSTITUTIONS:
        three = NON_STANDARD_SUBSTITUTIONS[three]
    return three_to_index(three)


def augmented_is_aa(three):
    if three in NON_STANDARD_SUBSTITUTIONS:
        three = NON_STANDARD_SUBSTITUTIONS[three]
    return is_aa(three, standard=True)


def is_hetero_residue(res):
    return len(res.id[0].strip()) > 0


def get_atom_name_postfix(atom):
    name = atom.get_name()
    if name in ('N', 'CA', 'C', 'O'):
        return name
    if name[-1].isnumeric():
        return name[-2:]
    else:
        return name[-1:]


def get_residue_pos14(res):
    pos14 = torch.full([14, 3], float('inf'))
    suffix_to_atom = {get_atom_name_postfix(a):a for a in res.get_atoms()}
    atom_order = ['N', 'CA', 'C', 'O'] + RESIDUE_SIDECHAIN_POSTFIXES[augmented_three_to_one(res.get_resname())]
    for i, atom_suffix in enumerate(atom_order):
        if atom_suffix not in suffix_to_atom: continue
        pos14[i,0], pos14[i,1], pos14[i,2] = suffix_to_atom[atom_suffix].get_coord().tolist()
    return pos14


def parse_pdb(path, model_id=0):
    warnings.simplefilter('ignore', BiopythonWarning)
    parser = PDBParser()
    structure = parser.get_structure(None, path)
    return parse_complex(structure, model_id)


def parse_complex(structure, model_id=None):
    if model_id is not None:
        structure = structure[model_id]
    chains = Selection.unfold_entities(structure, 'C')

    aa, resseq, icode, seq = [], [], [], []
    pos14, pos14_mask = [], []
    chain_id, chain_seq = [], []
    for i, chain in enumerate(chains):
        seq_this = 0
        for res in chain:
            resname = res.get_resname()
            if not augmented_is_aa(resname): continue
            if not (res.has_id('CA') and res.has_id('C') and res.has_id('N')): continue

            # Chain
            chain_id.append(chain.get_id())
            chain_seq.append(i+1)

            # Residue types
            restype = augmented_three_to_index(resname)
            aa.append(restype)

            # Atom coordinates
            pos14_this = get_residue_pos14(res)
            pos14_mask_this = pos14_this.isfinite()
            pos14.append(pos14_this.nan_to_num(posinf=99999))
            pos14_mask.append(pos14_mask_this)
            
            # Sequential number
            resseq_this = int(res.get_id()[1])
            icode_this = res.get_id()[2]
            if seq_this == 0:
                seq_this = 1
            else:
                d_resseq = resseq_this - resseq[-1]
                if d_resseq == 0: seq_this += 1
                else: seq_this += d_resseq
            resseq.append(resseq_this)
            icode.append(icode_this)
            seq.append(seq_this)

    if len(aa) == 0:
        return None

    return {
        'name': structure.get_id(),

        # Chain
        'chain_id': ''.join(chain_id),
        'chain_seq': torch.LongTensor(chain_seq),

        # Sequence
        'aa': torch.LongTensor(aa), 
        'resseq': torch.LongTensor(resseq), 
        'icode': ''.join(icode), 
        'seq': torch.LongTensor(seq), 
        
        # Atom positions
        'pos14': torch.stack(pos14), 
        'pos14_mask': torch.stack(pos14_mask),
    }
