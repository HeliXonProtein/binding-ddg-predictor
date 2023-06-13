# DDG Predictor

![overview](./data/overview.png)

This repository contains the deep learning model introduced in the paper "Deep Learning-Guided Optimization of Human Antibody Against SARS-CoV-2 Variants with Broad Neutralization". It predicts changes in binding energy upon mutation (ddG) for protein-protein complexes.

## Installation

The installation can be done with conda:

```bash
git clone https://github.com/HeliXonProtein/binding-ddg-predictor.git
cd binding-ddg-predictor
conda env create -f environment.yml
```

## Usage

The model requires two input PDB files: (1) a wild-type complex structure, and (2) a mutated complex structure. The mutated structures are typically built by protein design packages such as Rosetta. Note that both structures must have the same length. The ddG can be predicted for the two structures by running the command:

```bash
ddg_predict <path-to-wild-type-pdb> <path-to-mutant-pdb>
```

A quick example can be obtained by running:

```
ddg_predict ./data/example_wt.pdb ./data/example_mut.pdb
```

## Citation

Coming soon...

## Contact

Please contact luost[at]helixon.com for any questions related to the source code.
