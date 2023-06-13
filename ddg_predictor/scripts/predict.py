import argparse
import pkgutil
import torch
from pathlib import Path
import urllib.request
from ddg_predictor.models.predictor import DDGPredictor
from ddg_predictor.utils.misc import MODEL_WEIGHTS, CACHE_DIR, recursive_to
from ddg_predictor.utils.data import load_wt_mut_pdb_pair

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('wt_pdb', type=str, help="Wildtype PDB structure.")
    parser.add_argument('mut_pdb', type=str, help="Mutation PDB structure")
    parser.add_argument('--model', type=str, default=None,
                        help="Model.pt location, if None repository model.pt is used")
    parser.add_argument('--device', type=str, default='cuda', help="cuda if GPU should be used.")
    parser.add_argument('--verbose', type=int, default=0, help="Verbosity, 0=quite, 1=prints (default: 0).")
    args = parser.parse_args()

    if args.model is None:
        if args.verbose:
            print(f"Downloading weights to: {CACHE_DIR}")
        CACHE_DIR.mkdir(exist_ok=True, parents=True)
        urllib.request.urlretrieve(MODEL_WEIGHTS, CACHE_DIR / "model.pt")
        model_in = CACHE_DIR / "model.pt"
    else:
        model_in = args.model

    ckpt = torch.load(model_in)
    config = ckpt['config']
    weight = ckpt['model']

    batch = load_wt_mut_pdb_pair(args.wt_pdb, args.mut_pdb)
    batch = recursive_to(batch, args.device)

    model = DDGPredictor(config.model).to(args.device)
    model.load_state_dict(weight)

    with torch.no_grad():
        model.eval()
        pred = model(batch['wt'], batch['mut'])
        print('Predicted ddG: %.2f' % pred.item())


if __name__ == '__main__':
    main()
