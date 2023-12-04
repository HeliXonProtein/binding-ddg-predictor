import math
import torch
from torch.utils.data._utils.collate import default_collate

from .protein import ATOM_CA, parse_pdb


class PaddingCollate(object):

    def __init__(self, length_ref_key='mutation_mask', pad_values={'aa': 20, 'pos14': float('999'), 'icode': ' ', 'chain_id': '-'}, donot_pad={'foldx'}, eight=False):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.donot_pad = donot_pad
        self.eight = eight

    def _pad_last(self, x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        elif isinstance(x, str):
            if value == 0:  # Won't pad strings if not specified
                return x
            pad = value * (n - len(x))
            return x + pad
        elif isinstance(x, dict):
            padded = {}
            for k, v in x.items():
                if k in self.donot_pad:
                    padded[k] = v
                else:
                    padded[k] = self._pad_last(v, n, value=self._get_pad_value(k))
            return padded
        else:
            return x

    @staticmethod
    def _get_pad_mask(l, n):
        return torch.cat([
            torch.ones([l], dtype=torch.bool),
            torch.zeros([n-l], dtype=torch.bool)
        ], dim=0)

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        max_length = max([data[self.length_ref_key].size(0) for data in data_list])
        if self.eight:
            max_length = math.ceil(max_length / 8) * 8
        data_list_padded = []
        for data in data_list:
            data_padded = {
                k: self._pad_last(v, max_length, value=self._get_pad_value(k))
                for k, v in data.items() if k in ('wt', 'mut', 'ddG', 'mutation_mask', 'index', 'mutation')
            }
            data_padded['mask'] = self._get_pad_mask(data[self.length_ref_key].size(0), max_length)
            data_list_padded.append(data_padded)
        return default_collate(data_list_padded)


def _mask_list(l, mask):
    return [l[i] for i in range(len(l)) if mask[i]]


def _mask_string(s, mask):
    return ''.join([s[i] for i in range(len(s)) if mask[i]])


def _mask_dict_recursively(d, mask):
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor) and v.size(0) == mask.size(0):
            out[k] = v[mask]
        elif isinstance(v, list) and len(v) == mask.size(0):
            out[k] = _mask_list(v, mask)
        elif isinstance(v, str) and len(v) == mask.size(0):
            out[k] = _mask_string(v, mask)
        elif isinstance(v, dict):
            out[k] = _mask_dict_recursively(v, mask)
        else:
            out[k] = v
    return out


class KnnResidue(object):

    def __init__(self, num_neighbors=128):
        super().__init__()
        self.num_neighbors = num_neighbors

    def __call__(self, data):
        pos_CA = data['wt']['pos14'][:, ATOM_CA]
        pos_CA_mut = pos_CA[data['mutation_mask']]
        diff = pos_CA_mut.view(1, -1, 3) - pos_CA.view(-1, 1, 3)
        dist = torch.linalg.norm(diff, dim=-1)

        try:
            mask = torch.zeros([dist.size(0)], dtype=torch.bool)
            mask[ dist.min(dim=1)[0].argsort()[:self.num_neighbors] ] = True
        except IndexError as e:
            print(data)
            raise e

        return _mask_dict_recursively(data, mask)


def load_wt_mut_pdb_pair(wt_path, mut_path):

    data_wt = parse_pdb(wt_path)
    data_mut = parse_pdb(mut_path)

    transform = KnnResidue()
    collate_fn = PaddingCollate()
    mutation_mask = (data_wt['aa'] != data_mut['aa'])
    batch = collate_fn([transform({'wt': data_wt, 'mut': data_mut, 'mutation_mask': mutation_mask})])
    return batch
