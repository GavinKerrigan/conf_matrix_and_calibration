# This source code is from sklearn.cluster.kmean (https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
# Copyright (c) 2012-2014 Awesome Inc.
# This source code is licensed under the MIT license found in the
# 3rd-party-licenses.txt file in the root directory of this source tree.

# Initialization heuristic, copied from sklearn.cluster.kmean
import numpy as np
from numpy.random import RandomState
import scipy.sparse as sp
from scipy import stats
from sklearn.utils.extmath import stable_cumsum, row_norms  
from sklearn.metrics.pairwise import euclidean_distances

def CE_mtx(logits_p_in, logits_q_in):
    logits_p = np.reshape(logits_p_in.astype(np.float64), [logits_p_in.shape[0], 1])
    logits_q = np.reshape(logits_q_in.astype(np.float64), [1, logits_q_in.shape[0]])
    CE_mtx   = - logits_q * (0.5 + 0.5*np.tanh(logits_p/2.)) + np.maximum(0., logits_q) + np.log(1. + np.exp(-abs(logits_q)))
    return CE_mtx   

def KL_mtx(logits_p_in, logits_q_in):
    logits_p = np.reshape(logits_p_in.astype(np.float64), [logits_p_in.shape[0], 1])
    logits_q = np.reshape(logits_q_in.astype(np.float64), [1, logits_q_in.shape[0]])
    KL_mtx   = (logits_p - logits_q) * (0.5 + 0.5*np.tanh(logits_p/2.)) + np.maximum(0., logits_q) + np.log(1. + np.exp(-abs(logits_q))) - np.maximum(0., logits_p) - np.log(1. + np.exp(-abs(logits_p)))
    #KL_mtx = - logits_q * (0.5 + 0.5*np.tanh(logits_p/2.)) + np.maximum(0., logits_q) + np.log(1. + np.exp(-abs(logits_q)))
    return KL_mtx   

def JSD_mtx(logits_p, logits_q):
    logits_p_a = np.reshape(logits_p.astype(np.float64), [logits_p.shape[0], 1])
    logits_q_a = np.reshape(logits_q.astype(np.float64), [1, logits_q.shape[0]])
    logits_q_a = logits_q_a * 0.5 + 0.5 * logits_p_a
    KL_mtx_a   = (logits_p_a - logits_q_a) * (0.5 + 0.5*np.tanh(logits_p_a/2.)) + np.maximum(0., logits_q_a) + np.log(1. + np.exp(-abs(logits_q_a))) - np.maximum(0., logits_p_a) - np.log(1. + np.exp(-abs(logits_p_a)))
        
    logits_p_b = np.reshape(logits_p.astype(np.float64), [1, logits_p.shape[0]])
    logits_q_b = np.reshape(logits_q.astype(np.float64), [logits_q.shape[0], 1])
    logits_p_b = logits_q_b * 0.5 + 0.5 * logits_p_b
    KL_mtx_b   = (logits_q_b - logits_p_b) * (0.5 + 0.5*np.tanh(logits_q_b/2.)) + np.maximum(0., logits_p_b) + np.log(1. + np.exp(-abs(logits_p_b))) - np.maximum(0., logits_q_b) - np.log(1. + np.exp(-abs(logits_q_b)))
    return KL_mtx_a * 0.5 + KL_mtx_b.transpose()*0.5        




def kmeans_pp_init(X, n_clusters, random_state, n_local_trials=None, mode = 'jsd'):
    """Init n_clusters seeds according to k-means++

    Parameters
    ----------
    X : array or sparse matrix, shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters : integer
        The number of seeds to choose

    x_squared_norms : array, shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : int, RandomState instance
        The generator used to initialize the centers. Use an int to make the
        randomness deterministic.
        See :term:`Glossary <random_state>`.

    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape
    random_state = np.random.RandomState(random_state)
    centers = np.empty((n_clusters, n_features), dtype=X.dtype)
    center_ids = np.empty((n_clusters,), dtype=np.int64)

    #assert x_squared_norms is not None, 'x_squared_norms None in _k_init'
    x_squared_norms = row_norms(X, squared=True)
    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    #test_id   = random_state.randint(n_samples)
    #assert test_id != center_id:
    center_ids[0] = center_id
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]
    
    # Initialize list of closest distances and calculate current potential
    if mode == 'euclidean':
        closest_dist_sq = euclidean_distances(centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms, squared=True)
    elif mode == 'kl':
    #def KL_div(logits_p, logits_q):
    #    assert logits_p.shape[1] == 1 or logits_q.shape[1] == 1
    #    return (logits_p - logits_q) * (np.tanh(logits_p/2.) * 0.5 + 0.5) + np.maximum(logits_q, 0.) + np.log(1.+np.exp(-abs(logits_q))) + np.maximum(logits_p, 0.) + np.log(1.+np.exp(-abs(logits_p)))
        closest_dist_sq = KL_mtx(X[:,0], centers[0]).transpose()
    elif mode == 'ce':
        closest_dist_sq = CE_mtx(X[:,0], centers[0]).transpose()
    elif mode == 'jsd':
        closest_dist_sq = JSD_mtx(X[:,0], centers[0]).transpose()
    else:
        raise ValueError("Unknown distance in Kmeans++ initialization")
    
    current_pot     = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rnd_samples = random_state.random_sample(n_local_trials) 
        test1       = random_state.random_sample(n_local_trials) 
        rand_vals   = rnd_samples * current_pot
        assert np.any(abs(test1 - rnd_samples) > 1e-4)

        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1, out=candidate_ids)

        # Compute distances to center candidates
        if mode == 'euclidean':
            distance_to_candidates = euclidean_distances(X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)
        elif mode == 'ce':
            distance_to_candidates = CE_mtx(X[:,0], X[candidate_ids,0]).transpose()
        elif mode == 'kl':
            distance_to_candidates = KL_mtx(X[:,0], X[candidate_ids,0]).transpose()
        else:
            distance_to_candidates = JSD_mtx(X[:,0], X[candidate_ids,0]).transpose()
        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates, out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate  = np.argmin(candidates_pot)
        current_pot     = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate  = candidate_ids[best_candidate]
        center_ids[c]   = best_candidate
        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]

    return centers, center_ids





