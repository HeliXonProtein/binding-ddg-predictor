import torch
import torch.nn as nn

from utils.protein import ATOM_CA, ATOM_CB


def get_pos_CB(pos14, atom_mask):
    """
    Args:
        pos14:  (N, L, 14, 3)
        atom_mask:  (N, L, 14)
    """
    N, L = pos14.shape[:2]
    mask_CB = atom_mask[:, :, ATOM_CB]  # (N, L)
    mask_CB = mask_CB[:, :, None].expand(N, L, 3)
    pos_CA = pos14[:, :, ATOM_CA]   # (N, L, 3)
    pos_CB = pos14[:, :, ATOM_CB]
    return torch.where(mask_CB, pos_CB, pos_CA)


def mask_zero(mask, value):
    return torch.where(mask, value, torch.zeros_like(value))


class PositionalEncoding(nn.Module):
    
    def __init__(self, num_funcs=6):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer('freq_bands', 2.0 ** torch.linspace(0.0, num_funcs-1, num_funcs))
    
    def get_out_dim(self, in_dim):
        return in_dim * (2 * self.num_funcs + 1)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1) # (..., d, 1)
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)   # (..., d, 2f+1)
        code = code.reshape(shape)
        return code



def safe_norm(x, dim=-1, keepdim=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):
    """
    Args:
        center: (N, L, 3), usually the position of C_alpha.
        p1:     (N, L, 3), usually the position of C.
        p2:     (N, L, 3), usually the position of N.
    Returns
        A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
        The matrix is composed of 3 column vectors: [e1, e2, e3].
    """
    v1 = p1 - center    # (N, L, 3)
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center    # (N, L, 3)
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)    # (N, L, 3)

    mat = torch.cat([
        e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
    ], dim=-1)  # (N, L, 3, 3_index)
    return mat


def local_to_global(R, t, p):
    """
    Description:
        Convert local (internal) coordinates to global (external) coordinates q.
        q <- Rp + t
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        p:  Local coordinates, (N, L, ..., 3).
    Returns:
        q:  Global coordinates, (N, L, ..., 3).
    """
    assert p.size(-1) == 3
    p_size = p.size()
    N, L = p_size[0], p_size[1]

    p = p.view(N, L, -1, 3).transpose(-1, -2)   # (N, L, *, 3) -> (N, L, 3, *)
    q = torch.matmul(R, p) + t.unsqueeze(-1)    # (N, L, 3, *)
    q = q.transpose(-1, -2).reshape(p_size)     # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return q


def global_to_local(R, t, q):
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        q:  Global coordinates, (N, L, ..., 3).
    Returns:
        p:  Local coordinates, (N, L, ..., 3).
    """
    assert q.size(-1) == 3
    q_size = q.size()
    N, L = q_size[0], q_size[1]

    q = q.reshape(N, L, -1, 3).transpose(-1, -2)   # (N, L, *, 3) -> (N, L, 3, *)
    if t is None: 
        p = torch.matmul(R.transpose(-1, -2), q)  # (N, L, 3, *)
    else: 
        p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))  # (N, L, 3, *)
    p = p.transpose(-1, -2).reshape(q_size)     # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return p

