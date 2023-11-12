import torch

ExpBLEU = dict[int, list[list[float]]]


def build_acc_matrix(bleu_scores: ExpBLEU | torch.Tensor) -> torch.Tensor:
    """
    Builds the accuracy matrix by computing the mean BLEU score for each task.
    :param bleu_scores: a dict containing an int as key indicating the experience and a list of task scores (which are
        list of BLEU scores themselves). To give an example let's say that we have something like this:
        bleu_scores = {
            0: [[31.0, 32.5], [39.5, 40.0]],
            1: [[29.0, 28.9], [32.0, 33.0]],
        }
        Then, by using this method you would obtain:
        tensor([[31.7500, 39.7500],
                [28.9500, 32.500]])
        Keep in mind that you can also pass an accuracy matrix built by yourself to this method, and it will only check
        whether the such matrix is square.
    :return: the n times n accuracy matrix.
    """
    if not isinstance(bleu_scores, torch.Tensor):
        exps_bleu_lens = []
        bleu_scores_tensors = []
        for _, exp_bleu_scores in bleu_scores.items():
            exps_bleu_lens.append([len(exp) for exp in exp_bleu_scores])
            bleu_scores_tensors.append(torch.tensor(exp_bleu_scores))

        if not all(exps_bleu_lens[0] == exp_lens for exp_lens in exps_bleu_lens):
            raise ValueError("The number of computed BLEU scores must be the same for all the experiences.")

        acc_matrix = torch.stack(bleu_scores_tensors).mean(dim=-1)
    else:
        n, m = bleu_scores.size()
        if n != m:
            raise ValueError("The accuracy matrix must be a square matrix.")

        acc_matrix = bleu_scores

    return acc_matrix


def compute_acc(bleu_scores: ExpBLEU | torch.Tensor) -> torch.Tensor:
    """
    Compute the accuracy (mean BLEU scores) given a dict of experiences and their computed BLEU scores or an accuracy
    matrix.
    :param bleu_scores: a dict containing an int as key indicating the experience and a list of task scores (which are
        list of BLEU scores themselves). To give an example let's say that we have something like this:
        bleu_scores = {
            0: [[31.0, 32.5], [39.5, 40.0]],
            1: [[29.0, 28.9], [32.0, 33.0]],
        }
        Then, by using this method you would obtain:
        tensor([[31.7500, 39.7500],
                [28.9500, 32.500]])
        Keep in mind that you can also pass an accuracy matrix built by yourself to this method, and it will only check
        whether the such matrix is square.
    :return: the accuracy for each experience or the average one
    """
    acc_matrix = build_acc_matrix(bleu_scores)
    acc = acc_matrix.mean(dim=-1)
    return acc


def compute_bwt(bleu_scores: ExpBLEU | torch.Tensor) -> torch.Tensor:
    """
    Compute the backward transfer given a dict of experiences and their computed BLEU scores or an accuracy matrix.
    :param bleu_scores: a dict containing an int as key indicating the experience and a list of task scores (which are
        list of BLEU scores themselves). To give an example let's say that we have something like this:
        bleu_scores = {
            0: [[31.0, 32.5], [39.5, 40.0]],
            1: [[29.0, 28.9], [32.0, 33.0]],
        }
        Then, by using this method you would obtain:
        tensor([[31.7500, 39.7500],
                [28.9500, 32.500]])
        Keep in mind that you can also pass an accuracy matrix built by yourself to this method, and it will only check
        whether the such matrix is square.
    :return: the backward transfer for each experience or the average one.
    """
    acc_matrix = build_acc_matrix(bleu_scores)
    n = acc_matrix.size(0)
    bwt = torch.zeros(n)
    for i in range(1, n):
        diff = 0.0
        for j in range(i):
            diff += acc_matrix[i, j] - acc_matrix[j, j]

        bwt[i] = diff
        bwt[i] = bwt[i] / i

    return bwt


def compute_fwt(bleu_scores: ExpBLEU | torch.Tensor) -> torch.Tensor:
    """
    Compute the forward transfer given a dict of experiences and their computed BLEU scores or an accuracy matrix.
    :param bleu_scores: a dict containing an int as key indicating the experience and a list of task scores (which are
        list of BLEU scores themselves). To give an example let's say that we have something like this:
        bleu_scores = {
            0: [[31.0, 32.5], [39.5, 40.0]],
            1: [[29.0, 28.9], [32.0, 33.0]],
        }
        Then, by using this method you would obtain:
        tensor([[31.7500, 39.7500],
                [28.9500, 32.500]])
        Keep in mind that you can also pass an accuracy matrix built by yourself to this method, and it will only check
        whether the such matrix is square.
    :return: the forward transfer for each experience or the average one.
    """
    acc_matrix = build_acc_matrix(bleu_scores)
    n = acc_matrix.size(0)
    fwt = torch.zeros(n)
    for i in range(n - 1):
        fwt[i] = acc_matrix[i, i + 1 :].sum()

    return fwt
