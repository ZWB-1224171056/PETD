import torch
import random
from torch.distributions import Binomial


def split(n_way, n_shot, n_query):
    permuted_ids = torch.zeros(n_way, n_shot + n_query).long()
    for i in range(n_way):
        permuted_ids[i, :].copy_(torch.randperm((n_shot + n_query)) + (n_shot + n_query) * i)
    if torch.cuda.is_available():
        permuted_ids = permuted_ids.cuda()
    sup_idx, que_idx = torch.split(permuted_ids, [n_shot, n_query], dim=1)
    sup_idx = sup_idx.reshape(-1)
    que_idx = que_idx.reshape(-1)
    return sup_idx, que_idx

def split_pro(task_num, batch_size, n_way, n_shot, n_query, prob_success):
    permuted_ids = torch.zeros(task_num, n_way, n_shot + n_query).long()

    for i in range(task_num):
        clsmap = torch.randperm(batch_size)[0:n_way]
        for j, clsid in enumerate(clsmap):
            binomial_dist = Binomial(n_query, prob_success)
            sample_neighboor_num = int(binomial_dist.sample().item())
            # sample_neighboor_num = args.m
            permuted_ids[i, j, :].copy_(
                torch.cat(((torch.randperm(n_shot + n_query)[0:(n_shot + n_query - sample_neighboor_num)] + (
                        n_shot + n_query) * clsid),
                           (torch.randperm(n_query)[0:sample_neighboor_num] + (
                                   n_shot + n_query) * batch_size + n_query * clsid)), dim=0))
    if torch.cuda.is_available():
        permuted_ids = permuted_ids.cuda()
    sup_idx, que_idx = torch.split(permuted_ids, [n_shot, n_query], dim=2)
    sup_idx = sup_idx.flatten()
    que_idx = que_idx.flatten()
    return sup_idx, que_idx

def split_neighboor(pesudo_label, task_num, batch_size, n_way, n_shot, n_query, prob_success):

    unique_values, inverse_indices = torch.unique(pesudo_label, return_inverse=True)
    permuted_ids = torch.zeros(task_num, n_way, n_shot + n_query).long()
    # if unique_values.size(0) < n_way:
    #     return split_pro(task_num, batch_size, n_way, n_shot, n_query, prob_success)
    for i in range(task_num):
        indices = torch.randperm(unique_values.size(0))[:n_way]
        positions = [random.choice(torch.nonzero(inverse_indices == idx).flatten()).item() for idx in indices]
        for j, clsid in enumerate(positions):
            binomial_dist = Binomial(n_query, prob_success)
            sample_neighboor_num = int(binomial_dist.sample().item())
            # sample_neighboor_num = m
            permuted_ids[i, j, :] = torch.cat(
            ((torch.randperm(n_shot + n_query)[0:(n_shot + n_query - sample_neighboor_num)] + (
                    n_shot + n_query) * clsid),
                       (torch.randperm(n_query)[0:sample_neighboor_num] + (
                               n_shot + n_query) * batch_size + n_query * clsid)), dim=0)
    if torch.cuda.is_available():
        permuted_ids = permuted_ids.cuda()
    sup_idx, que_idx = torch.split(permuted_ids, [n_shot, n_query], dim=2)
    sup_idx = sup_idx.flatten()
    que_idx = que_idx.flatten()
    return sup_idx, que_idx

