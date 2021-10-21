import sys
sys.path.insert(0, '../')

from data_utils import *
import torch
from calibrators import *
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import calibration as cal
from combination_methods import *
import csv

# note: this experiment does not appear in our paper and may contain outdated code.

"""
def em_combo_experiment():
    # TODO : This code is old and needs updated for CIFAR-10H experiments
    # Calibrates and estimates confusion on held-out train set
    # Combines on test set and evaluates

    human_counts, model_probs, true_labels = load_CIFAR10H()
    true_labels = true_labels.astype(int)
    model_logits = np.log(model_probs)
    # Simulate a single human labeler
    y_h = get_human_labels_outcomes(human_counts, true_labels)[0][:, 0].astype(int)
    n = y_h.size

    temp, conf_h = fit_EM(model_probs, y_h)
    calibrated_model_probs = calibrate_probs_TS(model_probs, temp)
    y_comb = combine(calibrated_model_probs, y_h, conf_h)

    return np.mean(y_comb == true_labels)
"""


def run_experiment(out_fpath=None):
    """ Evaluates the oracle and EM algorithms (in terms of accuracy and calibration) on Noisy ImageNet
    """
    assert out_fpath is not None, 'Must specify output filepath'
    # Create CSV output file if needed, write header
    with open(out_fpath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_name', 'noise_level', 'epochs', 'trial',
                         'acc_h_te', 'acc_m_te', 'acc_comb_oracle_te', 'acc_comb_te',
                         'ce_m_te', 'ce_m_calibrated_te', 'ce_combo_te', 'ce_oracle_combo_te', 'ce_m_oracle_te',
                                                                                               'ece_m_te',
                         'ece_m_calibrated_te', 'ece_combo_te', 'ece_oracle_combo_te', 'ece_m_oracle_te',
                                                                                       'frobenius_distance_conf_te'])

    # model_names = ['alexnet', 'densenet161', 'googlenet', 'resnet152', 'vgg19']
    model_names = ['densenet161']
    noise_levels = [80, 95, 110, 125]
    # epochs = [None, 0, 1, 10]
    epochs = [None, 0, 10]
    noise_type = 'phase'
    n_runs = 5
    for model_name in tqdm(model_names, desc='Models', position=0, leave=True):
        for noise_level in tqdm(noise_levels, position=1, leave=False, desc='Noise Levels'):
            for epoch in tqdm(epochs, position=2, leave=False, desc='Epochs'):
                for i in tqdm(range(n_runs), position=3, leave=False, desc='Runs'):
                    human_ids, y_h, y_true, model_probs = load_old_noisy_imagenet(noise_level, model_name,
                                                                             n_epochs=epoch, noise_type=noise_type)
                    # Train/test split 70/30
                    y_h_tr, y_h_te, model_probs_tr, model_probs_te, y_true_tr, y_true_te = train_test_split(
                        y_h, model_probs, y_true, test_size=0.3, random_state=i)

                    # ----- 'Orancle' Experiment
                    y_comb_oracle_soft_te = oracle_combo(y_h_tr, model_probs_tr, y_true_tr, model_probs_te, y_h_te)
                    y_comb_oracle_te = np.argmax(y_comb_oracle_soft_te, axis=1)

                    temp_oracle = temperature_scaling(torch.from_numpy(np.log(model_probs_tr)),
                                                      torch.from_numpy(y_true_tr))['temperature'].item()
                    calibrated_probs_te_oracle = calibrate_probs_TS(model_probs_te, temp_oracle)

                    # ----- EM Experiment
                    # Fit EM parameters on train set
                    calibrator, conf_h = fit_EM(model_probs_tr, y_h_tr)  # TODO: Different calibration methods
                    # Calibrate predictions on test set
                    # TODO: Different calibration methods
                    model_probs_calibrated_te = calibrator.calibrate(model_probs_te)
                    # Combine calibrated model predictions with human labels on test set
                    y_comb_te_soft = combine(model_probs_calibrated_te, y_h_te, conf_h)
                    y_comb_te = np.argmax(y_comb_te_soft, axis=1)

                    # ----- Evaluate accuracies
                    acc_comb_te = np.mean(y_comb_te == y_true_te)
                    acc_comb_oracle_te = np.mean(y_comb_oracle_te == y_true_te)
                    acc_h_te = np.mean(y_h_te == y_true_te)
                    y_m_te = np.argmax(model_probs_te, axis=1)
                    acc_m_te = np.mean(y_m_te == y_true_te)

                    # ----- Evaluate calibration
                    # NB: This is the \ell_2, debiased, marginal calibration error
                    ce_m_te = cal.get_calibration_error(model_probs_te, y_true_te,
                                                        p=2, debias=True, mode='marginal')
                    ce_m_calibrated_te = cal.get_calibration_error(model_probs_calibrated_te, y_true_te,
                                                                   p=2, debias=True, mode='marginal')
                    ce_combo_te = cal.get_calibration_error(y_comb_te_soft, y_true_te,
                                                            p=2, debias=True, mode='marginal')
                    ce_oracle_combo_te = cal.get_calibration_error(y_comb_oracle_soft_te, y_true_te,
                                                                   p=2, debias=True, mode='marginal')
                    ce_m_oracle_te = cal.get_calibration_error(calibrated_probs_te_oracle, y_true_te,
                                                               p=2, debias=True, mode='marginal')

                    # NB: This is the usual ECE
                    ece_m_te = cal.get_ece(model_probs_te, y_true_te)
                    ece_m_calibrated_te = cal.get_ece(model_probs_calibrated_te, y_true_te)
                    ece_combo_te = cal.get_ece(y_comb_te_soft, y_true_te)
                    ece_oracle_combo_te = cal.get_ece(y_comb_oracle_soft_te, y_true_te)
                    ece_m_oracle_te = cal.get_ece(calibrated_probs_te_oracle, y_true_te)

                    # Evaluate confusion matrix
                    # Entry [i,j] is P(h = i | Y = j)
                    conf_h_te = confusion_matrix(y_true_te, y_h_te, normalize='pred').T
                    # Computes the Frobenius-norm (RMSE) distance between:
                    # (i) human confusion matrix estimated via EM
                    # (ii) human confusion matrix directly estimated via ground-truth on test set
                    frobenius_distance_conf_te = np.linalg.norm(conf_h_te - conf_h)

                    # Write results to CSV
                    with open(out_fpath, 'a', newline='') as f:
                        # TODO : Save model name, noise level, epoch
                        writer = csv.writer(f)
                        writer.writerow([model_name, noise_level, epoch, i,
                                         acc_h_te, acc_m_te, acc_comb_oracle_te, acc_comb_te,
                                         ce_m_te, ce_m_calibrated_te, ce_combo_te, ce_oracle_combo_te, ce_m_oracle_te,
                                         ece_m_te, ece_m_calibrated_te, ece_combo_te, ece_oracle_combo_te,
                                         ece_m_oracle_te,
                                         frobenius_distance_conf_te])


if __name__ == '__main__':
    out_fpath = './output/'
    run_experiment(out_fpath)
