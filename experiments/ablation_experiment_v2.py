import sys
sys.path.insert(0, '../')

from data_utils import *
from utils import *
from combination_methods import *
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import calibration as cal
import csv
import numpy as np
import os
from imax_calib.evaluations import calibration_metrics as cal_metrics
from scipy.special import softmax

# note: this experiment does not appear in our paper and may contain outdated code.


def get_cw_ECE(probs, y_true):
    evals = cal_metrics.compute_top_1_and_CW_ECEs(probs, y_true, list_approximators=['mECE'])
    return evals['cw_mECE']


def run_experiment_cifar10(out_fpath=None):
    """ Evaluates the oracle and EM algorithms (in terms of accuracy and calibration) on CIFAR10
    """
    assert out_fpath is not None, 'Must specify output filepath'
    model_names = ['r_low_acc', 'resnet-110', 'preresnet-110', 'densenet-bc-L190-k40']
    test_size = 0.2
    n_runs = 25

    for model_name in tqdm(model_names, desc='Models', leave=True):
        output_file = out_fpath + f'{model_name}_ablation.csv'
        assert not os.path.exists(output_file), 'Output filepath already exists'
        # Create CSV output file, write header
        with open(output_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['trial', 'acc_h', 'acc_m',
                             'acc_comb_TS', 'acc_comb_ETS', 'acc_comb_nocal', 'acc_comb_doubleconf',
                             'ce_m', 'ce_m_TS', 'ce_m_ETS',
                             'ce_combo_TS', 'ce_combo_ETS', 'ce_combo_nocal', 'ce_combo_doubleconf'])

        human_counts, model_probs, y_true = load_CIFAR10H(model_name)
        y_h = simulate_single_human(human_counts)
        for i in tqdm(range(n_runs), leave=False, desc='Runs'):
            # Train/test split
            y_h_tr, y_h_te, model_probs_tr, model_probs_te, y_true_tr, y_true_te = train_test_split(
                y_h, model_probs, y_true, test_size=test_size, random_state=i)

            # ----- Calibrator: temperature scaling
            oracle_combiner_TS = OracleCombiner(calibration_method='temperature scaling')
            oracle_combiner_TS.fit(model_probs_tr, y_h_tr, y_true_tr)
            y_comb_prob_TS = oracle_combiner_TS.combine_proba(model_probs_te, y_h_te)
            y_comb_TS = oracle_combiner_TS.combine(model_probs_te, y_h_te)

            # ----- Calibrator: ensemble temperature scaling
            oracle_combiner_ETS = OracleCombiner(calibration_method='ensemble temperature scaling')
            oracle_combiner_ETS.fit(model_probs_tr, y_h_tr, y_true_tr)
            y_comb_prob_ETS = oracle_combiner_ETS.combine_proba(model_probs_te, y_h_te)
            y_comb_ETS = oracle_combiner_ETS.combine(model_probs_te, y_h_te)

            # ----- Calibrator: None
            oracle_combiner_nocal = OracleCombiner(calibration_method=None)
            oracle_combiner_nocal.fit(model_probs_tr, y_h_tr, y_true_tr)
            y_comb_prob_nocal = oracle_combiner_nocal.combine_proba(model_probs_te, y_h_te)
            y_comb_nocal = oracle_combiner_nocal.combine(model_probs_te, y_h_te)

            # ----- Only estimate model's confusion matrix
            double_conf_combiner = DoubleConfusionCombiner()
            double_conf_combiner.fit(model_probs_tr, y_h_tr, y_true_tr)
            y_comb_prob_doubleconf = double_conf_combiner.combine_proba(model_probs_te, y_h_te)
            y_comb_doubleconf = double_conf_combiner.combine(model_probs_te, y_h_te)

            # ----- Evaluate accuracies
            acc_comb_oracle_TS = np.mean(y_comb_TS == y_true_te)
            acc_comb_oracle_ETS = np.mean(y_comb_ETS == y_true_te)
            acc_comb_oracle_nocal = np.mean(y_comb_nocal == y_true_te)
            acc_comb_doubleconf = np.mean(y_comb_doubleconf == y_true_te)
            acc_h_te = np.mean(y_h_te == y_true_te)
            y_m_te = np.argmax(model_probs_te, axis=1)
            acc_m_te = np.mean(y_m_te == y_true_te)

            # ----- Evaluate calibration
            # NB: This is the marginal L1 CE (debiase)
            ce_m_te = cal.get_calibration_error(model_probs_te, y_true_te,
                                                p=1, debias=False, mode='marginal')
            ce_m_TS = cal.get_calibration_error(oracle_combiner_TS.calibrate(model_probs_te), y_true_te,
                                                p=1, debias=False, mode='marginal')
            ce_m_ETS = cal.get_calibration_error(oracle_combiner_ETS.calibrate(model_probs_te), y_true_te,
                                                 p=1, debias=False, mode='marginal')
            ce_combo_TS = cal.get_calibration_error(y_comb_prob_TS, y_true_te,
                                                    p=1, debias=False, mode='marginal')
            ce_combo_ETS = cal.get_calibration_error(y_comb_prob_ETS, y_true_te,
                                                     p=1, debias=False, mode='marginal')
            ce_combo_nocal = cal.get_calibration_error(y_comb_prob_nocal, y_true_te,
                                                       p=1, debias=False, mode='marginal')
            ce_combo_doubleconf = cal.get_calibration_error(y_comb_prob_doubleconf, y_true_te,
                                                            p=1, debias=False, mode='marginal')

            # Write results to CSV
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([i, acc_h_te, acc_m_te,
                                 acc_comb_oracle_TS, acc_comb_oracle_ETS, acc_comb_oracle_nocal, acc_comb_doubleconf,
                                 ce_m_te, ce_m_TS, ce_m_ETS,
                                 ce_combo_TS, ce_combo_ETS, ce_combo_nocal, ce_combo_doubleconf])


def run_experiment_noisy_imagenet(out_fpath=None):
    """ Evaluates the oracle and EM algorithms (in terms of accuracy and calibration) on noisy ImageNet
    """
    assert out_fpath is not None, 'Must specify output filepath'
    model_acc_levels = ['low', 'med', 'high']
    noise_levels = [80, 95, 110, 125]
    test_size = 0.2
    n_runs = 25

    for model_level in tqdm(model_acc_levels, desc='Models', leave=True):
        for noise_level in tqdm(noise_levels, desc='Noise Levels'):
            output_file = out_fpath + f'vgg19{model_level}_noise{noise_level}_ablation.csv'
            assert not os.path.exists(output_file), 'Output filepath already exists'
            # Create CSV output file, write header
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['trial', 'acc_h', 'acc_m',
                                 'acc_comb_TS', 'acc_comb_ETS', 'acc_comb_imax_CW', 'acc_comb_imax_sCW',
                                 'acc_comb_nocal', 'acc_comb_doubleconf',
                                 'ce_m', 'ce_m_TS', 'ce_m_ETS', 'ce_m_imax_CW', 'ce_m_imax_sCW',
                                 'ce_combo_TS', 'ce_combo_ETS', 'ce_combo_imax_CW', 'ce_combo_imax_sCW',
                                 'ce_combo_nocal', 'ce_combo_doubleconf'])

            y_true, y_h, model_probs = load_noisy_imagenet(noise_level, model_level)
            for i in tqdm(range(n_runs), leave=False, desc='Runs'):
                # Train/test split
                y_h_tr, y_h_te, model_probs_tr, model_probs_te, y_true_tr, y_true_te = train_test_split(
                    y_h, model_probs, y_true, test_size=test_size, random_state=i)

                # ----- Calibrator: temperature scaling
                oracle_combiner_TS = OracleCombiner(calibration_method='temperature scaling')
                oracle_combiner_TS.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_TS = oracle_combiner_TS.combine_proba(model_probs_te, y_h_te)
                y_comb_TS = oracle_combiner_TS.combine(model_probs_te, y_h_te)

                # ----- Calibrator: ensemble temperature scaling
                oracle_combiner_ETS = OracleCombiner(calibration_method='ensemble temperature scaling')
                oracle_combiner_ETS.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_ETS = oracle_combiner_ETS.combine_proba(model_probs_te, y_h_te)
                y_comb_ETS = oracle_combiner_ETS.combine(model_probs_te, y_h_te)

                # ----- Calibrator: imax binning (CW)
                oracle_combiner_imax_CW = OracleCombiner(calibration_method='imax binning', mode='CW')
                oracle_combiner_imax_CW.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_imax_CW = oracle_combiner_imax_CW.combine_proba(model_probs_te, y_h_te)
                y_comb_imax_CW = oracle_combiner_imax_CW.combine(model_probs_te, y_h_te)

                # ----- Calibrator: imax binning (sCW)
                oracle_combiner_imax_sCW = OracleCombiner(calibration_method='imax binning', mode='sCW')
                oracle_combiner_imax_sCW.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_imax_sCW = oracle_combiner_imax_sCW.combine_proba(model_probs_te, y_h_te)
                y_comb_imax_sCW = oracle_combiner_imax_sCW.combine(model_probs_te, y_h_te)

                # ----- Calibrator: None
                oracle_combiner_nocal = OracleCombiner(calibration_method=None)
                oracle_combiner_nocal.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_nocal = oracle_combiner_nocal.combine_proba(model_probs_te, y_h_te)
                y_comb_nocal = oracle_combiner_nocal.combine(model_probs_te, y_h_te)

                # ----- Only estimate model's confusion matrix
                double_conf_combiner = DoubleConfusionCombiner()
                double_conf_combiner.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_doubleconf = double_conf_combiner.combine_proba(model_probs_te, y_h_te)
                y_comb_doubleconf = double_conf_combiner.combine(model_probs_te, y_h_te)

                # ----- Evaluate accuracies
                acc_comb_oracle_TS = np.mean(y_comb_TS == y_true_te)
                acc_comb_oracle_ETS = np.mean(y_comb_ETS == y_true_te)
                acc_comb_oracle_nocal = np.mean(y_comb_nocal == y_true_te)
                acc_comb_doubleconf = np.mean(y_comb_doubleconf == y_true_te)
                acc_comb_imax_CW = np.mean(y_comb_imax_CW == y_true_te)
                acc_comb_imax_sCW = np.mean(y_comb_imax_sCW == y_true_te)
                acc_h_te = np.mean(y_h_te == y_true_te)
                y_m_te = np.argmax(model_probs_te, axis=1)
                acc_m_te = np.mean(y_m_te == y_true_te)

                # ----- Evaluate calibration
                ce_m_te = cal.get_calibration_error(model_probs_te, y_true_te,
                                                    p=1, debias=False, mode='marginal')
                ce_m_TS = cal.get_calibration_error(oracle_combiner_TS.calibrate(model_probs_te), y_true_te,
                                                    p=1, debias=False, mode='marginal')
                ce_m_ETS = cal.get_calibration_error(oracle_combiner_ETS.calibrate(model_probs_te), y_true_te,
                                                     p=1, debias=False, mode='marginal')
                ce_m_imax_CW = cal.get_calibration_error(oracle_combiner_imax_CW.calibrate(model_probs_te), y_true_te,
                                                         p=1, debias=False, mode='marginal')
                ce_m_imax_sCW = cal.get_calibration_error(oracle_combiner_imax_sCW.calibrate(model_probs_te), y_true_te,
                                                          p=1, debias=False, mode='marginal')
                # ----- Of combination
                ce_combo_TS = cal.get_calibration_error(y_comb_prob_TS, y_true_te,
                                                        p=1, debias=False, mode='marginal')
                ce_combo_ETS = cal.get_calibration_error(y_comb_prob_ETS, y_true_te,
                                                         p=1, debias=False, mode='marginal')
                ce_combo_nocal = cal.get_calibration_error(y_comb_prob_nocal, y_true_te,
                                                           p=1, debias=False, mode='marginal')
                ce_combo_doubleconf = cal.get_calibration_error(y_comb_prob_doubleconf, y_true_te,
                                                                p=1, debias=False, mode='marginal')
                ce_combo_imax_CW = cal.get_calibration_error(y_comb_prob_imax_CW, y_true_te,
                                                             p=1, debias=False, mode='marginal')
                ce_combo_imax_sCW = cal.get_calibration_error(y_comb_prob_imax_sCW, y_true_te,
                                                              p=1, debias=False, mode='marginal')

                # Write results to CSV
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, acc_h_te, acc_m_te,
                                     acc_comb_oracle_TS, acc_comb_oracle_ETS, acc_comb_imax_CW, acc_comb_imax_sCW,
                                     acc_comb_oracle_nocal, acc_comb_doubleconf,
                                     ce_m_te, ce_m_TS, ce_m_ETS, ce_m_imax_CW, ce_m_imax_sCW,
                                     ce_combo_TS, ce_combo_ETS, ce_combo_imax_CW, ce_combo_imax_sCW,
                                     ce_combo_nocal, ce_combo_doubleconf])


def run_experiment_noisy_imagenet_logit(out_fpath=None):
    """ Evaluates the oracle and EM algorithms (in terms of accuracy and calibration) on Noisy ImageNet
    """
    assert out_fpath is not None, 'Must specify output filepath'
    model_acc_levels = ['low', 'med', 'high']
    noise_levels = [80, 95, 110, 125]
    test_size = 0.2
    n_runs = 25

    for model_level in tqdm(model_acc_levels, desc='Models', leave=True):
        for noise_level in tqdm(noise_levels, desc='Noise Levels'):
            output_file = out_fpath + f'vgg19{model_level}_noise{noise_level}_ablation.csv'
            assert not os.path.exists(output_file), 'Output filepath already exists'
            # Create CSV output file, write header
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['trial', 'acc_h', 'acc_m',
                                 'acc_comb_TS', 'acc_comb_ETS', 'acc_comb_imax_CW', 'acc_comb_imax_sCW',
                                 'acc_comb_dirichlet', 'acc_comb_nocal', 'acc_comb_doubleconf',
                                 'ce_m', 'ce_m_TS', 'ce_m_ETS', 'ce_m_imax_CW', 'ce_m_imax_sCW', 'ce_m_dirichlet',
                                 'ce_combo_TS', 'ce_combo_ETS', 'ce_combo_imax_CW', 'ce_combo_imax_sCW',
                                 'ce_combo_dirichlet', 'ce_combo_nocal', 'ce_combo_doubleconf'])

            y_true, y_h, model_logits = load_noisy_imagenet_logits(noise_level, model_level)
            model_probs = softmax(model_logits, axis=1)
            for i in tqdm(range(n_runs), leave=False, desc='Runs'):
                # Train/test split
                y_h_tr, y_h_te, model_logits_tr, model_logits_te, \
                model_probs_tr, model_probs_te, y_true_tr, y_true_te = train_test_split(
                    y_h, model_logits, model_probs, y_true, test_size=test_size, random_state=i)

                # ----- Calibrator: temperature scaling
                oracle_combiner_TS = OracleCombiner(calibration_method='temperature scaling')
                oracle_combiner_TS.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_TS = oracle_combiner_TS.combine_proba(model_probs_te, y_h_te)
                y_comb_TS = oracle_combiner_TS.combine(model_probs_te, y_h_te)

                # ----- Calibrator: ensemble temperature scaling
                oracle_combiner_ETS = OracleCombiner(calibration_method='ensemble temperature scaling')
                oracle_combiner_ETS.fit(model_probs_tr, y_h_tr, y_true_tr, model_logits=model_logits_tr)
                y_comb_prob_ETS = oracle_combiner_ETS.combine_proba(model_probs_te, y_h_te)
                y_comb_ETS = oracle_combiner_ETS.combine(model_probs_te, y_h_te)

                # ----- Calibrator: imax binning (CW)
                oracle_combiner_imax_CW = OracleCombiner(calibration_method='imax binning', mode='CW', num_bins=20)
                oracle_combiner_imax_CW.fit(model_probs_tr, y_h_tr, y_true_tr, model_logits=model_logits_tr)
                y_comb_prob_imax_CW = oracle_combiner_imax_CW.combine_proba(model_probs_te, y_h_te)
                y_comb_imax_CW = oracle_combiner_imax_CW.combine(model_probs_te, y_h_te)

                # ----- Calibrator: imax binning (sCW)
                oracle_combiner_imax_sCW = OracleCombiner(calibration_method='imax binning', mode='sCW', num_bins=20)
                oracle_combiner_imax_sCW.fit(model_probs_tr, y_h_tr, y_true_tr, model_logits=model_logits_tr)
                y_comb_prob_imax_sCW = oracle_combiner_imax_sCW.combine_proba(model_probs_te, y_h_te)
                y_comb_imax_sCW = oracle_combiner_imax_sCW.combine(model_probs_te, y_h_te)

                # ----- Calibrator: Dirichlet
                oracle_combiner_dirichlet = OracleCombiner(calibration_method='dirichlet')
                oracle_combiner_dirichlet.fit(model_probs_tr, y_h_tr, y_true_tr, model_logits=model_logits_tr)
                y_comb_prob_dirichlet = oracle_combiner_dirichlet.combine_proba(model_probs_te, y_h_te)
                y_comb_dirichlet = oracle_combiner_dirichlet.combine(model_probs_te, y_h_te)

                # ----- Calibrator: None
                oracle_combiner_nocal = OracleCombiner(calibration_method=None)
                oracle_combiner_nocal.fit(model_probs_tr, y_h_tr, y_true_tr, model_logits=model_logits_tr)
                y_comb_prob_nocal = oracle_combiner_nocal.combine_proba(model_probs_te, y_h_te)
                y_comb_nocal = oracle_combiner_nocal.combine(model_probs_te, y_h_te)

                # ----- Only estimate model's confusion matrix
                double_conf_combiner = DoubleConfusionCombiner()
                double_conf_combiner.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_doubleconf = double_conf_combiner.combine_proba(model_probs_te, y_h_te)
                y_comb_doubleconf = double_conf_combiner.combine(model_probs_te, y_h_te)

                # ----- Evaluate accuracies
                acc_comb_oracle_TS = np.mean(y_comb_TS == y_true_te)
                acc_comb_oracle_ETS = np.mean(y_comb_ETS == y_true_te)
                acc_comb_oracle_nocal = np.mean(y_comb_nocal == y_true_te)
                acc_comb_doubleconf = np.mean(y_comb_doubleconf == y_true_te)
                acc_comb_imax_CW = np.mean(y_comb_imax_CW == y_true_te)
                acc_comb_imax_sCW = np.mean(y_comb_imax_sCW == y_true_te)
                acc_comb_dirichlet = np.mean(y_comb_dirichlet == y_true_te)
                acc_h_te = np.mean(y_h_te == y_true_te)
                y_m_te = np.argmax(model_probs_te, axis=1)
                acc_m_te = np.mean(y_m_te == y_true_te)

                # ----- Evaluate calibration
                # Model only
                ce_m = get_cw_ECE(model_probs_te, y_true_te)
                ce_m_TS = get_cw_ECE(oracle_combiner_TS.calibrate(model_probs_te), y_true_te)
                ce_m_ETS = get_cw_ECE(oracle_combiner_ETS.calibrate(model_probs_te), y_true_te)
                ce_m_imax_CW = get_cw_ECE(oracle_combiner_imax_CW.calibrate(model_probs_te), y_true_te)
                ce_m_imax_sCW = get_cw_ECE(oracle_combiner_imax_sCW.calibrate(model_probs_te), y_true_te)
                ce_m_dirichlet = get_cw_ECE(oracle_combiner_dirichlet.calibrate(model_probs_te), y_true_te)

                # Combination
                ce_combo_doubleconf = get_cw_ECE(y_comb_prob_doubleconf, y_true_te)
                ce_combo_nocal = get_cw_ECE(y_comb_prob_nocal, y_true_te)
                ce_combo_TS = get_cw_ECE(y_comb_prob_TS, y_true_te)
                ce_combo_ETS = get_cw_ECE(y_comb_prob_ETS, y_true_te)
                ce_combo_imax_CW = get_cw_ECE(y_comb_prob_imax_CW, y_true_te)
                ce_combo_imax_sCW = get_cw_ECE(y_comb_prob_imax_sCW, y_true_te)
                ce_combo_dirichlet = get_cw_ECE(y_comb_prob_dirichlet, y_true_te)

                # Write results to CSV
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, acc_h_te, acc_m_te,
                                     acc_comb_oracle_TS, acc_comb_oracle_ETS, acc_comb_imax_CW, acc_comb_imax_sCW,
                                     acc_comb_dirichlet, acc_comb_oracle_nocal, acc_comb_doubleconf,
                                     ce_m, ce_m_TS, ce_m_ETS, ce_m_imax_CW, ce_m_imax_sCW, ce_m_dirichlet,
                                     ce_combo_TS, ce_combo_ETS, ce_combo_imax_CW, ce_combo_imax_sCW,
                                     ce_combo_dirichlet, ce_combo_nocal, ce_combo_doubleconf])


def run_experiment_noisy_imagenet(out_fpath=None):
    """ Evaluates the oracle and EM algorithms (in terms of accuracy and calibration) on noisy ImageNet
    """
    assert out_fpath is not None, 'Must specify output filepath'
    model_name = 'densenet161'
    model_acc_levels = [None, 0, 10]
    noise_levels = [80, 95, 110, 125]
    test_size = 0.2
    n_runs = 10

    for epochs in tqdm(model_acc_levels, desc='Models', leave=True):
        for noise_level in tqdm(noise_levels, desc='Noise Levels'):
            output_file = out_fpath + f'{model_name}_epoch{epochs}_noise{noise_level}_ablation.csv'
            # assert not os.path.exists(output_file), 'Output filepath already exists'
            # Create CSV output file, write header
            with open(output_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['trial', 'acc_h', 'acc_m',
                                 'acc_comb_TS', 'acc_comb_ETS', 'acc_comb_imax_CW', 'acc_comb_imax_sCW',
                                 'acc_comb_nocal', 'acc_comb_doubleconf',
                                 'ce_m', 'ce_m_TS', 'ce_m_ETS', 'ce_m_imax_CW', 'ce_m_imax_sCW',
                                 'ce_combo_TS', 'ce_combo_ETS', 'ce_combo_imax_CW', 'ce_combo_imax_sCW',
                                 'ce_combo_nocal', 'ce_combo_doubleconf'])

                y_true, y_h, model_probs = load_old_noisy_imagenet_data(noise_level, model_name, n_epochs=epochs)
            for i in tqdm(range(n_runs), leave=False, desc='Runs'):
                # Train/test split
                y_h_tr, y_h_te, model_probs_tr, model_probs_te, y_true_tr, y_true_te = train_test_split(
                    y_h, model_probs, y_true, test_size=test_size, random_state=i)

                # ----- Calibrator: temperature scaling
                oracle_combiner_TS = OracleCombiner(calibration_method='temperature scaling')
                oracle_combiner_TS.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_TS = oracle_combiner_TS.combine_proba(model_probs_te, y_h_te)
                y_comb_TS = oracle_combiner_TS.combine(model_probs_te, y_h_te)

                """
                print('\n \n')
                print(f' Combo acc {np.mean(y_comb_TS == y_true_te)}')
                print(f' human tr {np.mean(y_h_tr == y_true_tr)}')
                print(f' human te {np.mean(y_h_te == y_true_te)}')
                print(f' model tr {np.mean(np.argmax(model_probs_tr, axis=1) == y_true_tr)}')
                print(f' model te {np.mean(np.argmax(model_probs_te, axis=1) == y_true_te)}')
                print(f' model all {np.mean(np.argmax(model_probs, axis=1) == y_true)}')
                quit()
                """

                # ----- Calibrator: ensemble temperature scaling
                oracle_combiner_ETS = OracleCombiner(calibration_method='ensemble temperature scaling')
                oracle_combiner_ETS.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_ETS = oracle_combiner_ETS.combine_proba(model_probs_te, y_h_te)
                y_comb_ETS = oracle_combiner_ETS.combine(model_probs_te, y_h_te)

                # ----- Calibrator: imax binning (CW)
                oracle_combiner_imax_CW = OracleCombiner(calibration_method='imax binning', mode='CW')
                oracle_combiner_imax_CW.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_imax_CW = oracle_combiner_imax_CW.combine_proba(model_probs_te, y_h_te)
                y_comb_imax_CW = oracle_combiner_imax_CW.combine(model_probs_te, y_h_te)

                # ----- Calibrator: imax binning (sCW)
                oracle_combiner_imax_sCW = OracleCombiner(calibration_method='imax binning', mode='sCW')
                oracle_combiner_imax_sCW.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_imax_sCW = oracle_combiner_imax_sCW.combine_proba(model_probs_te, y_h_te)
                y_comb_imax_sCW = oracle_combiner_imax_sCW.combine(model_probs_te, y_h_te)

                # ----- Calibrator: None
                oracle_combiner_nocal = OracleCombiner(calibration_method=None)
                oracle_combiner_nocal.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_nocal = oracle_combiner_nocal.combine_proba(model_probs_te, y_h_te)
                y_comb_nocal = oracle_combiner_nocal.combine(model_probs_te, y_h_te)

                # ----- Only estimate model's confusion matrix
                double_conf_combiner = DoubleConfusionCombiner()
                double_conf_combiner.fit(model_probs_tr, y_h_tr, y_true_tr)
                y_comb_prob_doubleconf = double_conf_combiner.combine_proba(model_probs_te, y_h_te)
                y_comb_doubleconf = double_conf_combiner.combine(model_probs_te, y_h_te)

                # ----- Evaluate accuracies
                acc_comb_oracle_TS = np.mean(y_comb_TS == y_true_te)
                acc_comb_oracle_ETS = np.mean(y_comb_ETS == y_true_te)
                acc_comb_oracle_nocal = np.mean(y_comb_nocal == y_true_te)
                acc_comb_doubleconf = np.mean(y_comb_doubleconf == y_true_te)
                acc_comb_imax_CW = np.mean(y_comb_imax_CW == y_true_te)
                acc_comb_imax_sCW = np.mean(y_comb_imax_sCW == y_true_te)
                acc_h_te = np.mean(y_h_te == y_true_te)
                y_m_te = np.argmax(model_probs_te, axis=1)
                acc_m_te = np.mean(y_m_te == y_true_te)

                # ----- Evaluate calibration
                ce_m_te = cal.get_calibration_error(model_probs_te, y_true_te,
                                                    p=1, debias=False, mode='marginal')
                ce_m_TS = cal.get_calibration_error(oracle_combiner_TS.calibrate(model_probs_te), y_true_te,
                                                    p=1, debias=False, mode='marginal')
                ce_m_ETS = cal.get_calibration_error(oracle_combiner_ETS.calibrate(model_probs_te), y_true_te,
                                                     p=1, debias=False, mode='marginal')
                ce_m_imax_CW = cal.get_calibration_error(oracle_combiner_imax_CW.calibrate(model_probs_te), y_true_te,
                                                         p=1, debias=False, mode='marginal')
                ce_m_imax_sCW = cal.get_calibration_error(oracle_combiner_imax_sCW.calibrate(model_probs_te), y_true_te,
                                                          p=1, debias=False, mode='marginal')
                # ----- Of combination
                ce_combo_TS = cal.get_calibration_error(y_comb_prob_TS, y_true_te,
                                                        p=1, debias=False, mode='marginal')
                ce_combo_ETS = cal.get_calibration_error(y_comb_prob_ETS, y_true_te,
                                                         p=1, debias=False, mode='marginal')
                ce_combo_nocal = cal.get_calibration_error(y_comb_prob_nocal, y_true_te,
                                                           p=1, debias=False, mode='marginal')
                ce_combo_doubleconf = cal.get_calibration_error(y_comb_prob_doubleconf, y_true_te,
                                                                p=1, debias=False, mode='marginal')
                ce_combo_imax_CW = cal.get_calibration_error(y_comb_prob_imax_CW, y_true_te,
                                                             p=1, debias=False, mode='marginal')
                ce_combo_imax_sCW = cal.get_calibration_error(y_comb_prob_imax_sCW, y_true_te,
                                                              p=1, debias=False, mode='marginal')

                # Write results to CSV
                with open(output_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i, acc_h_te, acc_m_te,
                                     acc_comb_oracle_TS, acc_comb_oracle_ETS, acc_comb_imax_CW, acc_comb_imax_sCW,
                                     acc_comb_oracle_nocal, acc_comb_doubleconf,
                                     ce_m_te, ce_m_TS, ce_m_ETS, ce_m_imax_CW, ce_m_imax_sCW,
                                     ce_combo_TS, ce_combo_ETS, ce_combo_imax_CW, ce_combo_imax_sCW,
                                     ce_combo_nocal, ce_combo_doubleconf])


if __name__ == '__main__':
    """
    out_fpath = './output/cifar10h/'
    run_experiment_cifar10(out_fpath)
    """

    out_fpath = './output/'
    run_experiment_noisy_imagenet(out_fpath)
