import torch
import torch.nn as nn
import torch.nn.functional as F

from models.residue import PerResidueEncoder
from models.attention import GAEncoder
from models.common import get_pos_CB, construct_3d_basis
from utils.protein import ATOM_N, ATOM_CA, ATOM_C


class ComplexEncoder(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.relpos_embedding = nn.Embedding(cfg.max_relpos*2+2, cfg.pair_feat_dim)
        self.residue_encoder = PerResidueEncoder(cfg.node_feat_dim)

        if cfg.geomattn is not None:
            self.ga_encoder = GAEncoder(
                node_feat_dim = cfg.node_feat_dim,
                pair_feat_dim = cfg.pair_feat_dim,
                num_layers = cfg.geomattn.num_layers,
                spatial_attn_mode = cfg.geomattn.spatial_attn_mode,
            )
        else:
            self.out_mlp = nn.Sequential(
                nn.Linear(cfg.node_feat_dim, cfg.node_feat_dim), nn.ReLU(),
                nn.Linear(cfg.node_feat_dim, cfg.node_feat_dim), nn.ReLU(),
                nn.Linear(cfg.node_feat_dim, cfg.node_feat_dim),
            )

    def forward(self, pos14, aa, seq, chain, mask_atom):
        """
        Args:
            pos14:  (N, L, 14, 3).
            aa:     (N, L).
            seq:    (N, L).
            chain:  (N, L).
            mask_atom:  (N, L, 14)
        Returns:
            (N, L, node_ch)
        """
        same_chain = (chain[:, None, :] == chain[:, :, None])   # (N, L, L)
        relpos = (seq[:, None, :] - seq[:, :, None]).clamp(min=-self.cfg.max_relpos, max=self.cfg.max_relpos) + self.cfg.max_relpos # (N, L, L)
        relpos = torch.where(same_chain, relpos, torch.full_like(relpos, fill_value=self.cfg.max_relpos*2+1))
        pair_feat = self.relpos_embedding(relpos)   # (N, L, L, pair_ch)
        R = construct_3d_basis(pos14[:, :, ATOM_CA], pos14[:, :, ATOM_C], pos14[:, :, ATOM_N])

        # Residue encoder
        res_feat = self.residue_encoder(aa, pos14, mask_atom)

        # Geom encoder
        t = pos14[:, :, ATOM_CA]
        mask_residue = mask_atom[:, :, ATOM_CA]
        res_feat = self.ga_encoder(R, t, get_pos_CB(pos14, mask_atom), res_feat, pair_feat, mask_residue)

        return res_feat


class DDGReadout(nn.Module):

    def __init__(self, feat_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim*2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

        self.project = nn.Linear(feat_dim, 1, bias=False)


    def forward(self, node_feat_wt, node_feat_mut, mask=None):
        """
        Args:
            node_feat_wt:   (N, L, F).
            node_feat_mut:  (N, L, F).
            mask:   (N, L).
        """
        feat_wm = torch.cat([node_feat_wt, node_feat_mut], dim=-1)
        feat_mw = torch.cat([node_feat_mut, node_feat_wt], dim=-1)
        feat_diff = self.mlp(feat_wm) - self.mlp(feat_mw)       # (N, L, F)

        # feat_diff = self.mlp(node_feat_wt) - self.mlp(node_feat_mut)
        
        per_residue_ddg = self.project(feat_diff).squeeze(-1)   # (N, L)
        if mask is not None:
            per_residue_ddg = per_residue_ddg * mask
        ddg = per_residue_ddg.sum(dim=1)    # (N,)
        return ddg


class DDGPredictor(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.encoder = ComplexEncoder(cfg)
        self.ddG_readout = DDGReadout(cfg.node_feat_dim)

    def forward(self, complex_wt, complex_mut, ddG_true=None):
        mask_atom_wt  = complex_wt['pos14_mask'].all(dim=-1)    # (N, L, 14)
        mask_atom_mut = complex_mut['pos14_mask'].all(dim=-1)

        feat_wt  = self.encoder(complex_wt['pos14'], complex_wt['aa'], complex_wt['seq'], complex_wt['chain_seq'], mask_atom_wt)
        feat_mut = self.encoder(complex_mut['pos14'], complex_mut['aa'], complex_mut['seq'], complex_mut['chain_seq'], mask_atom_mut)
        
        mask_res = mask_atom_wt[:, :, ATOM_CA]
        ddG_pred = self.ddG_readout(feat_wt, feat_mut, mask_res)  # One mask is enough

        if ddG_true is None:
            return ddG_pred
        else:
            losses = {
                'ddG': F.mse_loss(ddG_pred, ddG_true),
            }
            return losses, ddG_pred

