# DDG Predictor

![overview](./data/overview.png)

This repository contains the deep learning model introduced in the paper "Deep Learning-Guided Optimization of Human
Antibody Against SARS-CoV-2 Variants with Broad Neutralization". It predicts changes in binding energy upon mutation (
ddG) for protein-protein complexes.

## 📢 News

- Please check out our latest work on mutational effect prediction for protein-protein interactions: *Rotamer Density Estimator is an Unsupervised Learner of the Effect of Mutations on Protein-Protein Interaction* (ICLR 2023) [[Code]](https://github.com/luost26/RDE-PPI) [[Paper]](https://www.biorxiv.org/content/10.1101/2023.02.28.530137)

## Installation

The installation can be done with conda:

```bash
git clone https://github.com/HeliXonProtein/binding-ddg-predictor.git
cd binding-ddg-predictor
conda env create -f environment.yml
```

If you have satisfied all the dependencies, you can install the package after cloning the repo with:
No additional packages will be installed except for the ddg-predictor package.

```bash
pip install -e .
````

## Usage

The model requires two input PDB files: (1) a wild-type complex structure, and (2) a mutated complex structure. The
mutated structures are typically built by protein design packages such as Rosetta. Note that both structures must have
the same length. The ddG can be predicted for the two structures by running the command:

```bash
ddg_predict <path-to-wild-type-pdb> <path-to-mutant-pdb>
```

A quick example can be obtained by running:

```
ddg_predict ./data/example_wt.pdb ./data/example_mut.pdb
```

Alternatively, for batch processing you can use the following command which will search for PDBs in the given in path:

```
ddg_predict wt_pdb mutant_pdbs/ --mut_pdb_is_path 1
```

## Citation

Coming soon...

## Contact

Please contact luost[at]helixon.com for any questions related to the source code.
