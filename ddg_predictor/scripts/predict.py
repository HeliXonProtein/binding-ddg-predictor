import argparse
import json
import urllib.request
from pathlib import Path

import torch

from ddg_predictor.models.predictor import DDGPredictor
from ddg_predictor.utils.data import load_wt_mut_pdb_pair
from ddg_predictor.utils.misc import MODEL_WEIGHTS, CACHE_DIR, recursive_to


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('wt_pdb', type=str, help="Wildtype PDB structure.")
    parser.add_argument('mut_pdb', type=str, help="Mutation PDB structure")
    parser.add_argument('--out_file', type=str, help="Output file json (default: stdout)", default=None)
    parser.add_argument('--model', type=str, default=None,
                        help="Model.pt location, if None repository model.pt is used")
    parser.add_argument('--device', type=str, default='cuda', help="cuda if GPU should be used.")
    parser.add_argument('--verbose', type=int, default=0, help="Verbosity, 0=quite, 1=prints (default: 0).")
    parser.add_argument('--mut_pdb_is_path', type=int, default=0,
                        help="If 1, mut_pdb is a directory with multiple PDBs, otherwise a PDB file (default: 0).")

    args = parser.parse_args()
    wt_pdb = args.wt_pdb
    device = args.device
    out_file = args.out_file

    if args.model is None:
        model_in = CACHE_DIR / "model.pt"
        if not model_in.exists():
            if args.verbose:
                print(f"Downloading weights to: {model_in}")
            CACHE_DIR.mkdir(exist_ok=True, parents=True)
            urllib.request.urlretrieve(MODEL_WEIGHTS, model_in)
    else:
        model_in = args.model

    ckpt = torch.load(model_in)
    config = ckpt['config']
    weight = ckpt['model']

    mut_pdb = args.mut_pdb
    if args.mut_pdb_is_path:
        mut_pdbs = [str(i) for i in Path(mut_pdb).glob("*/*.pdb")]
    else:
        mut_pdbs = [mut_pdb]

    ddg_predictions = []
    for mut_pdb in mut_pdbs:
        try:
            batch = load_wt_mut_pdb_pair(wt_pdb, mut_pdb)
            batch = recursive_to(batch, device)

            model = DDGPredictor(config.model).to(device)
            model.load_state_dict(weight)

            with torch.no_grad():
                model.eval()
                pred = model(batch['wt'], batch['mut'])

            if args.verbose:
                print('Predicted ddG: %.2f' % pred.item())

            ddg_predictions.append(pred.item())

        except:
            print(f"Failed to predict for {mut_pdb}")
            ddg_predictions.append(None)

    if out_file:
        json_out = {"wt_pdb": [wt_pdb] * len(mut_pdbs), "mut_pdbs": mut_pdbs, "ddg_predictions": ddg_predictions}
        with open(out_file, 'w') as f:
            json.dump(json_out, f)


if __name__ == '__main__':
    main()
