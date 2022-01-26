import numpy as np
from torch import nn
from sklearn.metrics import confusion_matrix

# This file implements various utility functions.


def get_human_labels_outcomes(human_counts, true_labels, seed=0):
    """ Converts from the counts to an ordered list of votes. Also computes the 0/1 Bernoulli outcomes.
    """
    rng = np.random.default_rng(seed)

    human_labels_per_input = np.sum(human_counts, axis=1)
    min_human_labels = int(min(human_labels_per_input))
    n_rows = human_counts.shape[0]
    n_classes = human_counts.shape[1]

    human_labels = np.empty(shape=(n_rows, min_human_labels))
    human_outcomes = np.empty(shape=(n_rows, min_human_labels))
    for row in range(n_rows):
        temp = []
        for i in range(n_classes):
            temp += [i] * int(human_counts[row, i])
        rng.shuffle(temp)
        human_labels[row, :] = temp[:min_human_labels]
        human_outcomes[row, :] = (human_labels[row, :] == true_labels[row]).astype(int)

    return human_labels, human_outcomes


def simulate_single_human(human_counts, seed=0):
    rng = np.random.default_rng(seed)

    human_labels_per_input = np.sum(human_counts, axis=1)
    min_human_labels = int(min(human_labels_per_input))
    n_rows = human_counts.shape[0]
    n_classes = human_counts.shape[1]

    human_labels = np.empty(shape=(n_rows, min_human_labels))
    for row in range(n_rows):
        temp = []
        for i in range(n_classes):
            temp += [i] * int(human_counts[row, i])
        rng.shuffle(temp)
        human_labels[row, :] = temp[:min_human_labels]

    return human_labels[:, 0].astype(int)


class SoftLogLoss(nn.Module):
    # Implements the "soft-log-loss" for use with the EM algorithm
    def __init__(self):
        super(SoftLogLoss, self).__init__()

    def forward(self, input, target):
        # input is tensor of model logits (n_samples, n_cls)
        # target is tensor of weight matrix (n_samples, n_cls)
        # c.f. https://github.com/pytorch/pytorch/issues/7455
        log_probs = nn.functional.log_softmax(input, dim=-1)
        loss = -1. * (log_probs * target).sum(dim=-1).mean()
        return loss


def get_model_confidence_ratio(model_probs, y_true, h=None, y=None, y_h=None, mode='diff'):
    # args h / y : condition on Y = y and/or h(X) = h
    # arg mode: 'max' or 'diff' -- determines denominator

    if (h is None) and (y is None):  # Unconditional
        idxs = [True] * y_true.size
    elif h is None:  # Distribution conditioned on Y only
        idxs = (y_true == y)
    elif y is None:  # Distribution conditioned on h only
        idxs = (y_h == h)
    else:  # Distribution conditioned on y and h
        idxs = np.logical_and((y_true == y), (y_h == h))

    eps = 1e-16
    model_probs = model_probs.clip(eps, 1. - eps)

    n = y_true[idxs].size
    _model_probs = model_probs[idxs]
    _y_true = y_true[idxs]

    model_confidence_ratio = np.empty(n)
    for i in range(n):
        true_class_conf = _model_probs[i][y_true[i]]
        if mode == 'max':
            denom = np.max([conf for j, conf in enumerate(_model_probs[i]) if j != _y_true[i]])
        elif mode == 'diff':
            denom = 1. - true_class_conf
        model_confidence_ratio[i] = true_class_conf / denom

    return model_confidence_ratio


def get_human_confidence_ratio(y_h_tr, y_true_tr, y_h_te, y_true_te, n_cls, h=None, y=None, mode='diff'):
    # Estimate human confusion matrix
    # Entry [i, j]  is #(Y = i and h = j)
    conf_h = 1. * confusion_matrix(y_true_tr, y_h_tr, labels=np.arange(n_cls))
    # Swap so entry [i, j] is #(h = i and Y = j)
    conf_h = conf_h.T
    eps = 1e-50
    conf_h = np.clip(conf_h, eps, None)
    normalizer = np.sum(conf_h, axis=0, keepdims=True)
    # Normalize columns so entry [i, j] is P(h = i | Y = j)
    conf_h /= normalizer

    if (h is None) and (y is None):  # Unconditional
        idxs = [True] * y_true_te.size
    elif h is None:  # Distribution conditioned on Y only
        idxs = (y_true_te == y)
    elif y is None:  # Distribution conditioned on h only
        idxs = (y_h_te == h)
    else:  # Distribution conditioned on y and h
        return conf_h[h, y] / (1. - conf_h[h, y])

    n = y_true_te[idxs].size
    _y_true = y_true_te[idxs]
    human_confidence_ratio = np.empty(n)
    for i in range(n):
        true_class_conf = conf_h[y_h_te[i], _y_true[i]]
        if mode == 'max':
            denom = np.max([conf for j, conf in enumerate(conf_h[y_h_te[i], :]) if j != _y_true[i]])
        elif mode == 'diff':
            denom = 1. - true_class_conf
        human_confidence_ratio[i] = true_class_conf / denom

    return human_confidence_ratio


def get_dirichlet_params(acc, strength, n_cls):
    # acc: desired off-diagonal accuracy
    # strength: strength of prior

    # Returns alpha,beta where the prior is Dir((beta, beta, . . . , alpha, . . . beta))
    # where the alpha appears for the correct class

    beta = 0.1
    alpha = beta * (n_cls - 1) * acc / (1. - acc)

    alpha *= strength
    beta *= strength

    alpha += 1
    beta += 1

    return alpha, beta


def diversity(y1, y2, y_t):
    y1_outcomes = (y1 == y_t)
    y2_outcomes = (y2 == y_t)

    n = y_t.size
    both_correct = sum((y1_outcomes == 1) & (y2_outcomes == 1)) / n
    both_incorrect = sum((y1_outcomes == 0) & (y2_outcomes == 0)) / n
    y1c_y2w = sum((y1_outcomes == 1) & (y2_outcomes == 0)) / n
    y1w_y2c = sum((y1_outcomes == 0) & (y2_outcomes == 1)) / n

    return both_correct, both_incorrect, y1c_y2w, y1w_y2c