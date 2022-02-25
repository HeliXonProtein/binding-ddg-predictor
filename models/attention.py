import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .common import mask_zero, global_to_local, local_to_global, normalize_vector


def _alpha_from_logits(logits, mask, inf=1e5):
    """
    Args:
        logits: Logit matrices, (N, L_i, L_j, num_heads).
        mask:   Masks, (N, L).
    Returns:
        alpha:  Attention weights.
    """
    N, L, _, _ = logits.size()
    mask_row = mask.view(N, L, 1, 1).expand_as(logits)      # (N, L, *, *)
    mask_pair = mask_row * mask_row.permute(0, 2, 1, 3)     # (N, L, L, *)
    
    logits = torch.where(mask_pair, logits, logits-inf)
    alpha = torch.softmax(logits, dim=2)  # (N, L, L, num_heads)
    alpha = torch.where(mask_row, alpha, torch.zeros_like(alpha))
    return alpha


def _heads(x, n_heads, n_ch):
    """
    Args:
        x:  (..., num_heads * num_channels)
    Returns:
        (..., num_heads, num_channels)
    """
    s = list(x.size())[:-1] + [n_heads, n_ch]
    return x.view(*s)


class GeometricAttention(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, spatial_attn_mode='CB', value_dim=16, query_key_dim=16, num_query_points=8, num_value_points=8, num_heads=12):
        super().__init__()
        self.node_feat_dim = node_feat_dim
        self.pair_feat_dim = pair_feat_dim
        self.value_dim = value_dim
        self.query_key_dim = query_key_dim
        self.num_query_points = num_query_points
        self.num_value_points = num_value_points
        self.num_heads = num_heads

        assert spatial_attn_mode in ('CB', 'vpoint')
        self.spatial_attn_mode = spatial_attn_mode

        # Node
        self.proj_query = nn.Linear(node_feat_dim, query_key_dim*num_heads, bias=False)
        self.proj_key   = nn.Linear(node_feat_dim, query_key_dim*num_heads, bias=False)
        self.proj_value = nn.Linear(node_feat_dim, value_dim*num_heads, bias=False)

        # Pair
        self.proj_pair_bias = nn.Linear(pair_feat_dim, num_heads, bias=False)

        # Spatial
        self.spatial_coef = nn.Parameter(torch.full([1, 1, 1, self.num_heads], fill_value=np.log(np.exp(1.) - 1.)), requires_grad=True)
        if spatial_attn_mode == 'vpoint':
            self.proj_query_point = nn.Linear(node_feat_dim, num_query_points*num_heads*3, bias=False)
            self.proj_key_point   = nn.Linear(node_feat_dim, num_query_points*num_heads*3, bias=False)
            self.proj_value_point = nn.Linear(node_feat_dim, num_value_points*num_heads*3, bias=False)

        # Output
        if spatial_attn_mode == 'CB':
            self.out_transform = nn.Linear(
                in_features = (num_heads*pair_feat_dim) + (num_heads*value_dim) + (num_heads*(3+3+1)),
                out_features = node_feat_dim,
            )
        elif spatial_attn_mode == 'vpoint':
            self.out_transform = nn.Linear(
                in_features = (num_heads*pair_feat_dim) + (num_heads*value_dim) + (num_heads*num_value_points*(3+3+1)),
                out_features = node_feat_dim,
            )
        self.layer_norm = nn.LayerNorm(node_feat_dim)

    def _node_logits(self, x):
        query_l = _heads(self.proj_query(x), self.num_heads, self.query_key_dim)    # (N, L, n_heads, qk_ch)
        key_l = _heads(self.proj_key(x), self.num_heads, self.query_key_dim)      # (N, L, n_heads, qk_ch)

        query_l = query_l.permute(0, 2, 1, 3)   # (N,L1,H,C) -> (N,H,L1,C)
        key_l = key_l.permute(0, 2, 3, 1)       # (N,L2,H,C) -> (N,H,C,L2)

        logits = torch.matmul(query_l, key_l)   # (N,H,L1,L2)
        logits = logits.permute(0, 2, 3, 1)     # (N,L1,L2,H)

        # logits = (query_l.unsqueeze(2) * key_l.unsqueeze(1) * (1 / np.sqrt(self.query_key_dim))).sum(-1)    # (N, L, L, num_heads)
        return logits

    def _pair_logits(self, z):
        logits_pair = self.proj_pair_bias(z)
        return logits_pair

    def _beta_logits(self, R, t, p_CB):
        N, L, _ = t.size()
        qk = p_CB[:, :, None, :].expand(N, L, self.num_heads, 3)
        sum_sq_dist = ((qk.unsqueeze(2) - qk.unsqueeze(1)) ** 2).sum(-1)    # (N, L, L, n_heads)
        gamma = F.softplus(self.spatial_coef)
        logtis_beta = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / 9)) / 2)
        return logtis_beta

    def _spatial_logits(self, R, t, x):
        N, L, _ = t.size()
        # Query
        query_points = _heads(self.proj_query_point(x), self.num_heads*self.num_query_points, 3)  # (N, L, n_heads * n_pnts, 3)
        query_points = local_to_global(R, t, query_points)  # Global query coordinates, (N, L, n_heads * n_pnts, 3)
        query_s = query_points.reshape(N, L, self.num_heads, -1)   # (N, L, n_heads, n_pnts*3)
        # Key
        key_points = _heads(self.proj_key_point(x), self.num_heads*self.num_query_points, 3)      # (N, L, 3, n_heads * n_pnts)
        key_points = local_to_global(R, t, key_points)      # Global key coordinates, (N, L, n_heads * n_pnts, 3)
        key_s = key_points.reshape(N, L, self.num_heads, -1)   # (N, L, n_heads, n_pnts*3)
        # Q-K Product
        sum_sq_dist = ((query_s.unsqueeze(2) - key_s.unsqueeze(1)) ** 2).sum(-1)    # (N, L, L, n_heads)
        gamma = F.softplus(self.spatial_coef)
        logits_spatial = sum_sq_dist * ((-1 * gamma * np.sqrt(2 / (9 * self.num_query_points))) / 2)  # (N, L, L, n_heads)
        return logits_spatial

    def _pair_aggregation(self, alpha, z):
        N, L = z.shape[:2]
        feat_p2n = alpha.unsqueeze(-1) * z.unsqueeze(-2)    # (N, L, L, n_heads, C)
        feat_p2n = feat_p2n.sum(dim=2)  # (N, L, n_heads, C)
        return feat_p2n.reshape(N, L, -1)

    def _node_aggregation(self, alpha, x):
        N, L = x.shape[:2]
        value_l = _heads(self.proj_value(x), self.num_heads, self.query_key_dim)  # (N, L, n_heads, v_ch)
        feat_node = alpha.unsqueeze(-1) * value_l.unsqueeze(1) # (N, L, L, n_heads, *) @ (N, *, L, n_heads, v_ch)
        feat_node = feat_node.sum(dim=2)  # (N, L, n_heads, v_ch)
        return feat_node.reshape(N, L, -1)

    def _beta_aggregation(self, alpha, R, t, p_CB, x):
        N, L, _ = t.size()
        v = p_CB[:, :, None, :].expand(N, L, self.num_heads, 3) # (N, L, n_heads, 3)
        aggr = alpha.reshape(N, L, L, self.num_heads, 1) * v.unsqueeze(1)   # (N, *, L, n_heads, 3)
        aggr = aggr.sum(dim=2)

        feat_points = global_to_local(R, t, aggr) # (N, L, n_heads, 3)
        feat_distance = feat_points.norm(dim=-1)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)

        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1), 
            feat_distance.reshape(N, L, -1), 
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def _spatial_aggregation(self, alpha, R, t, x):
        N, L, _ = t.size()
        value_points = _heads(self.proj_value_point(x), self.num_heads*self.num_value_points, 3)  # (N, L, n_heads * n_v_pnts, 3)
        value_points = local_to_global(R, t, value_points.reshape(N, L, self.num_heads, self.num_value_points, 3))     # (N, L, n_heads, n_v_pnts, 3)
        aggr_points = alpha.reshape(N, L, L, self.num_heads, 1, 1) * value_points.unsqueeze(1) # (N, *, L, n_heads, n_pnts, 3)
        aggr_points = aggr_points.sum(dim=2)    # (N, L, n_heads, n_pnts, 3)

        feat_points = global_to_local(R, t, aggr_points)    # (N, L, n_heads, n_pnts, 3)
        feat_distance = feat_points.norm(dim=-1)    # (N, L, n_heads, n_pnts)
        feat_direction = normalize_vector(feat_points, dim=-1, eps=1e-4)  # (N, L, n_heads, n_pnts, 3)
        
        feat_spatial = torch.cat([
            feat_points.reshape(N, L, -1), 
            feat_distance.reshape(N, L, -1), 
            feat_direction.reshape(N, L, -1),
        ], dim=-1)

        return feat_spatial

    def forward_beta(self, R, t, p_CB, x, z, mask):
        """
        Args:
            R:  Frame basis matrices, (N, L, 3, 3_index).
            t:  Frame external (absolute) coordinates, (N, L, 3).
            x:  Node-wise features, (N, L, F).
            z:  Pair-wise features, (N, L, L, C).
            mask:   Masks, (N, L).
        Returns:
            x': Updated node-wise features, (N, L, F).
        """
        # Attention logits
        logits_node = self._node_logits(x)
        logits_pair = self._pair_logits(z)
        logits_spatial = self._beta_logits(R, t, p_CB)
        # Summing logits up and apply `softmax`.
        logits_sum = logits_node + logits_pair + logits_spatial
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)  # (N, L, L, n_heads)

        # Aggregate features
        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_spatial = self._beta_aggregation(alpha, R, t, p_CB, x)

        # Finally
        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1)) # (N, L, F)
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm(x + feat_all)
        return x_updated

    
    def forward_vpoint(self, R, t, p_CB, x, z, mask):
        """
        Args:
            R:  Frame basis matrices, (N, L, 3, 3_index).
            t:  Frame external (absolute) coordinates, (N, L, 3).
            x:  Node-wise features, (N, L, F).
            z:  Pair-wise features, (N, L, L, C).
            mask:   Masks, (N, L).
        Returns:
            x': Updated node-wise features, (N, L, F).
        """
        # Attention logits
        logits_node = self._node_logits(x)
        logits_pair = self._pair_logits(z)
        logits_spatial = self._spatial_logits(R, t, x)
        # Summing logits up and apply `softmax`.
        logits_sum = logits_node + logits_pair + logits_spatial
        alpha = _alpha_from_logits(logits_sum * np.sqrt(1 / 3), mask)  # (N, L, L, n_heads)

        # Aggregate features
        feat_p2n = self._pair_aggregation(alpha, z)
        feat_node = self._node_aggregation(alpha, x)
        feat_spatial = self._spatial_aggregation(alpha, R, t, x)

        # Finally
        feat_all = self.out_transform(torch.cat([feat_p2n, feat_node, feat_spatial], dim=-1)) # (N, L, F)
        feat_all = mask_zero(mask.unsqueeze(-1), feat_all)
        x_updated = self.layer_norm(x + feat_all)
        return x_updated

    def forward(self, R, t, p_CB, x, z, mask):
        if self.spatial_attn_mode == 'CB':
            return self.forward_beta(R, t, p_CB, x, z, mask)
        else:
            return self.forward_vpoint(R, t, p_CB, x, z, mask)

class GAEncoder(nn.Module):

    def __init__(self, node_feat_dim, pair_feat_dim, num_layers, spatial_attn_mode='CB'):
        super().__init__()
        self.blocks = nn.ModuleList([
            GeometricAttention(node_feat_dim, pair_feat_dim, spatial_attn_mode=spatial_attn_mode)
            for _ in range(num_layers)
        ])

    def forward(self, R, t, p_CB, x, z, mask):
        for block in self.blocks:
            x = block(R, t, p_CB, x, z, mask)   # Residual connection within the block
        return x
