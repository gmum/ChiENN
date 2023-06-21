from collections import OrderedDict

import torch


def get_local_structure_map(psi_indices):
    """
    Adapted from ChIRo repository.
    """
    LS_dict = OrderedDict()
    LS_map = torch.zeros(psi_indices.shape[1], dtype=torch.long).to(psi_indices.device)
    v = 0
    for i, indices in enumerate(psi_indices.T):
        tupl = (int(indices[1]), int(indices[2]))
        if tupl not in LS_dict:
            LS_dict[tupl] = v
            v += 1
        LS_map[i] = LS_dict[tupl]

    alpha_indices = torch.zeros((2, len(LS_dict)), dtype=torch.long)
    for i, tupl in enumerate(LS_dict):
        alpha_indices[:, i] = torch.LongTensor(tupl)

    return LS_map, alpha_indices.to(psi_indices.device)
