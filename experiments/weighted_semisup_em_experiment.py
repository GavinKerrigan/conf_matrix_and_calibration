import sys
sys.path.insert(0, '../')

from data_utils import *
from combination_methods import *
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import calibration as cal
import csv
import os.path

# note: these experiments do not appear in our paper and may contain outdated code.


def run_experiment(out_fpath=None):
    """ Evaluates the oracle and EM algorithms (in terms of accuracy and calibration) on Noisy ImageNet

    Done in a semi-supervised fashion.
    """
    assert out_fpath is not None, 'Must specify output filepath'

    unsupervised_weights = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]

    # Create CSV output file if needed, write header
    with open(out_fpath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_name', 'noise_level', 'epochs', 'trial', 'n_l', 'n_u',
                         'acc_h_te', 'acc_m_te',
                         'acc_comb_oracle_te', 'acc_comb_unsup_te']
                        + ['acc_comb_semisup_te_weight{}'.format(v) for v in unsupervised_weights] +
                        ['ece_m_te', 'ece_m_calibrated_oracle_te', 'ece_m_calibrated_unsup_te']
                        + ['ece_m_calibrated_semisup_te_weight{}'.format(v) for v in unsupervised_weights] +
                        ['ece_combo_oracle_te', 'ece_combo_unsup_te']
                        + ['ece_combo_semisup_te_weight{}'.format(v) for v in unsupervised_weights])

    # Experiment parameters
    # model_names = ['alexnet', 'densenet161', 'googlenet', 'resnet152', 'vgg19']
    model_names = ['densenet161']
    noise_levels = [80, 95, 110, 125]
    # epochs = [None, 0, 1, 10]
    epochs = [None, 0, 10]
    noise_type = 'phase'

    n_runs = 10

    for model_name in tqdm(model_names, desc='Models', position=0, leave=True):
        for noise_level in tqdm(noise_levels, position=1, leave=False, desc='Noise Levels'):
            for epoch in tqdm(epochs, position=2, leave=False, desc='Epochs'):
                human_ids, y_h, y_true, model_probs = load_old_noisy_imagenet(noise_level, model_name,
                                                                         n_epochs=epoch, noise_type=noise_type)
                for i in tqdm(range(n_runs), position=3, leave=False, desc='Runs'):
                    # Train/test split 70/30
                    y_h_tr, y_h_te, model_probs_tr, model_probs_te, y_true_tr, y_true_te = train_test_split(
                        y_h, model_probs, y_true, test_size=0.3, random_state=i)

                    n_tr = y_h_tr.size  # Number of training points

                    # Evaluate accuracies of things that don't change with n_l
                    acc_h_te = np.mean(y_h_te == y_true_te)
                    y_m_te = np.argmax(model_probs_te, axis=1)
                    acc_m_te = np.mean(y_m_te == y_true_te)
                    # Evaluate calibration of things that don't change with n_l
                    ece_m_te = cal.get_ece(model_probs_te, y_true_te)

                    # ----- Unsupervised EM, all unlabeled data
                    # Edge case with n_l = 0
                    unsupervised_EM_combiner = UnsupervisedEMCombiner()
                    unsupervised_EM_combiner.fit(model_probs_tr, y_h_tr)
                    y_comb_prob_unsup_te = unsupervised_EM_combiner.combine_proba(model_probs_te, y_h_te)
                    y_comb_unsup_te = unsupervised_EM_combiner.combine(model_probs_te, y_h_te)

                    acc_combo_unsup_all_te = np.mean(y_comb_unsup_te == y_true_te)
                    ece_m_calibrated_unsup_all_te = cal.get_ece(unsupervised_EM_combiner.calibrate(model_probs_te),
                                                                y_true_te)
                    ece_combo_unsup_all_te = cal.get_ece(y_comb_prob_unsup_te, y_true_te)

                    with open(out_fpath, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([model_name, noise_level, epoch, i, 0, n_tr,
                                         acc_h_te, acc_m_te,
                                         None, acc_combo_unsup_all_te] + [None] * len(unsupervised_weights) +
                                        [ece_m_te, None, ece_m_calibrated_unsup_all_te]
                                        + [None] * len(unsupervised_weights) +
                                        [None, ece_combo_unsup_all_te]
                                        + [None] * len(unsupervised_weights))

                    # ----- Fully supervised (oracle) combo
                    # Edge case with n_l = all
                    oracle_combiner = OracleCombiner()
                    oracle_combiner.fit(model_probs_tr, y_h_tr, y_true_tr)
                    y_comb_prob_oracle_te = oracle_combiner.combine_proba(model_probs_te, y_h_te)
                    y_comb_oracle_te = oracle_combiner.combine(model_probs_te, y_h_te)

                    acc_combo_oracle_all_te = np.mean(y_comb_oracle_te == y_true_te)
                    ece_m_calibrated_oracle_all_te = cal.get_ece(oracle_combiner.calibrate(model_probs_te),
                                                                 y_true_te)
                    ece_combo_oracle_all_te = cal.get_ece(y_comb_prob_oracle_te, y_true_te)
                    with open(out_fpath, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([model_name, noise_level, epoch, i, n_tr, 0,
                                         acc_h_te, acc_m_te,
                                         acc_combo_oracle_all_te, None] + [None] * len(unsupervised_weights) +
                                        [ece_m_te, ece_m_calibrated_oracle_all_te, None]
                                        + [None] * len(unsupervised_weights) +
                                        [ece_combo_oracle_all_te, None]
                                        + [None] * len(unsupervised_weights))

                    n_l_sizes = [10, 50, 100, 250, 500, 1000, 2500, 4500]  # Amount of labeled data to use
                    for n_l in tqdm(n_l_sizes, leave=False, desc='Num. Labels'):
                        # Split into labeled / unlabeled datasets
                        n_u = n_tr - n_l
                        y_h_tr_u, y_h_tr_l = y_h_tr[n_l:], y_h_tr[:n_l]
                        model_probs_tr_u, model_probs_tr_l = model_probs_tr[n_l:], model_probs_tr[:n_l]
                        y_true_tr_l = y_true_tr[:n_l]

                        # ----- Labeled data only
                        oracle_combiner = OracleCombiner()
                        oracle_combiner.fit(model_probs_tr_l, y_h_tr_l, y_true_tr_l)
                        y_comb_prob_oracle_te = oracle_combiner.combine_proba(model_probs_te, y_h_te)
                        y_comb_oracle_te = oracle_combiner.combine(model_probs_te, y_h_te)
                        model_probs_calibrated_oracle_te = oracle_combiner.calibrate(model_probs_te)

                        # ----- Semi-Supervised EM
                        # Fit once for each choice in unsupervised_weights
                        y_comb_prob_semisup_te = dict.fromkeys(unsupervised_weights)
                        y_comb_semisup_te = dict.fromkeys(unsupervised_weights)
                        model_probs_calibrated_semisup_te = dict.fromkeys(unsupervised_weights)
                        for unsupervised_weight in tqdm(unsupervised_weights, leave=False, desc='Weights'):
                            semisup_combiner = SemiSupervisedEMCombiner(unsupervised_weight=unsupervised_weight)
                            semisup_combiner.fit(model_probs_tr_u, y_h_tr_u, model_probs_tr_l, y_h_tr_l, y_true_tr_l)
                            y_comb_prob_semisup_te[unsupervised_weight] = semisup_combiner.combine_proba(model_probs_te,
                                                                                                         y_h_te)
                            y_comb_semisup_te[unsupervised_weight] = semisup_combiner.combine(model_probs_te, y_h_te)
                            model_probs_calibrated_semisup_te[unsupervised_weight] = semisup_combiner.calibrate(
                                model_probs_te)

                        # ----- Unsupervised EM
                        unsup_combiner = UnsupervisedEMCombiner()
                        unsup_combiner.fit(model_probs_tr_u, y_h_tr_u)
                        y_comb_prob_unsup_te = unsup_combiner.combine_proba(model_probs_te, y_h_te)
                        y_comb_unsup_te = unsup_combiner.combine(model_probs_te, y_h_te)
                        model_probs_calibrated_unsup_te = unsup_combiner.calibrate(model_probs_te)

                        # ----- Evaluate accuracies
                        acc_comb_oracle_te = np.mean(y_comb_oracle_te == y_true_te)
                        acc_comb_semisup_te = dict.fromkeys(unsupervised_weights)
                        for unsupervised_weight in unsupervised_weights:
                            acc_comb_semisup_te[unsupervised_weight] = np.mean(
                                y_comb_semisup_te[unsupervised_weight] == y_true_te)
                        acc_comb_unsup_te = np.mean(y_comb_unsup_te == y_true_te)

                        # ----- Evaluate calibration
                        # Evaluate ECE of just model
                        ece_m_calibrated_oracle_te = cal.get_ece(model_probs_calibrated_oracle_te, y_true_te)
                        ece_m_calibrated_unsup_te = cal.get_ece(model_probs_calibrated_unsup_te, y_true_te)
                        ece_m_calibrated_semisup_te = dict.fromkeys(unsupervised_weights)
                        for unsupervised_weight in unsupervised_weights:
                            ece_m_calibrated_semisup_te[unsupervised_weight] = cal.get_ece(
                                model_probs_calibrated_semisup_te[unsupervised_weight], y_true_te)

                        # Evaluate ECE of combination
                        ece_combo_oracle_te = cal.get_ece(y_comb_prob_oracle_te, y_true_te)
                        ece_combo_unsup_te = cal.get_ece(y_comb_prob_unsup_te, y_true_te)
                        ece_combo_semisup_te = dict.fromkeys(unsupervised_weights)
                        for unsupervised_weight in unsupervised_weights:
                            ece_combo_semisup_te[unsupervised_weight] = cal.get_ece(
                                y_comb_prob_semisup_te[unsupervised_weight], y_true_te)

                        # Write results to CSV
                        with open(out_fpath, 'a', newline='') as f:
                            writer = csv.writer(f)
                            writer.writerow([model_name, noise_level, epoch, i, n_l, n_u,
                                             acc_h_te, acc_m_te,
                                             acc_comb_oracle_te, acc_comb_unsup_te]
                                            + [v for v in acc_comb_semisup_te.values()] +
                                            [ece_m_te, ece_m_calibrated_oracle_te, ece_m_calibrated_unsup_te]
                                            + [v for v in ece_m_calibrated_semisup_te.values()] +
                                            [ece_combo_oracle_te, ece_combo_unsup_te]
                                            + [v for v in ece_combo_semisup_te.values()])


if __name__ == '__main__':
    out_fpath = './output/'
    if os.path.exists(out_fpath):
        print('Output filepath exists, dont overwrite it!')
        quit()
    run_experiment(out_fpath)
