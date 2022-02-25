import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import argparse
import torch

from models.predictor import DDGPredictor
from utils.misc import *
from utils.data import *
from utils.protein import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('wt_pdb', type=str)
    parser.add_argument('mut_pdb', type=str)
    parser.add_argument('--model', type=str, default='./data/model.pt')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    batch = load_wt_mut_pdb_pair(args.wt_pdb, args.mut_pdb)
    batch = recursive_to(batch, args.device)

    ckpt = torch.load(args.model)
    config = ckpt['config']
    weight = ckpt['model']
    model = DDGPredictor(config.model).to(args.device)
    model.load_state_dict(weight)

    with torch.no_grad():
        model.eval()
        pred = model(batch['wt'], batch['mut'])
        print('Predicted ddG: %.2f' % pred.item())
