import torch


def get_codebook_usage(index_count: torch.Tensor):
    """
    :param index_count: (n, ) where n is the codebook size, express the number of times each index have been used.
    :return: prob of each index to be used: (n, ); perplexity: float; codebook_usage: float 0__1
    """

    # get used idx as probabilities
    used_indices = index_count / torch.sum(index_count)

    # perplexity
    perplexity = torch.exp(-torch.sum(used_indices * torch.log(used_indices + 1e-10), dim=-1)).sum().item()

    # get the percentage of used codebook
    n = index_count.shape[0]
    used_codebook = (torch.count_nonzero(used_indices).item() * 100) / n

    return used_indices, perplexity, used_codebook


def clipping(x: torch.Tensor):
    """
    clip tensor in range 0__1
    useful to prevent some conversion errors, for example before plotting or metric computation
    """

    x = torch.where(x < 0.0, torch.zeros(x.shape, device=x.device), x)
    x = torch.where(x > 1.0, torch.ones(x.shape, device=x.device), x)
    return x
