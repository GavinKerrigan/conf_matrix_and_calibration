import sys
sys.path.insert(0, '../')

from data_utils import *
from combination_methods import *
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
import calibration as cal
import csv
import os.path

# note: this experiment does not appear in our paper.

def run_experiment(out_fpath=None):
    """ Evaluates the oracle and EM algorithms (in terms of accuracy and calibration) on noisy ImageNet
    """
    assert out_fpath is not None, 'Must specify output filepath'
    # Create CSV output file if needed, write header
    with open(out_fpath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model_name', 'noise_level', 'epochs', 'trial',
                         'acc_h', 'acc_m', 'acc_combo_TS', 'acc_combo_dir',
                         'ce_m', 'ce_m_TS', 'ce_m_dir', 'ce_combo_TS', 'ce_combo_dir',
                         'ece_m_te', 'ece_m_TS', 'ece_m_dir', 'ece_combo_TS', 'ece_combo_dir'])

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
                    human_ids, y_h, y_true, model_probs = load_old_noisy_imagenet_data(noise_level, model_name,
                                                                             n_epochs=epoch, noise_type=noise_type)
                    # Train/test split 70/30
                    y_h_tr, y_h_te, model_probs_tr, model_probs_te, y_true_tr, y_true_te = train_test_split(
                        y_h, model_probs, y_true, test_size=0.3, random_state=i)

                    # ----- Calibrator: temperature scaling
                    oracle_combiner_TS = OracleCombiner(calibration_method='temperature scaling')
                    oracle_combiner_TS.fit(model_probs_tr, y_h_tr, y_true_tr)
                    y_comb_prob_TS = oracle_combiner_TS.combine_proba(model_probs_te, y_h_te)
                    y_comb_TS = oracle_combiner_TS.combine(model_probs_te, y_h_te)

                    # ----- Calibrator: Dirichlet calibration
                    oracle_combiner_dirichlet = OracleCombiner(calibration_method='dirichlet')
                    oracle_combiner_dirichlet.fit(model_probs_tr, y_h_tr, y_true_tr)
                    y_comb_prob_dirichlet = oracle_combiner_dirichlet.combine_proba(model_probs_te, y_h_te)
                    y_comb_dirichlet = oracle_combiner_dirichlet.combine(model_probs_te, y_h_te)

                    # ----- Evaluate accuracies
                    acc_comb_oracle_TS = np.mean(y_comb_TS == y_true_te)
                    acc_comb_oracle_dirichlet = np.mean(y_comb_dirichlet == y_true_te)
                    acc_h_te = np.mean(y_h_te == y_true_te)
                    y_m_te = np.argmax(model_probs_te, axis=1)
                    acc_m_te = np.mean(y_m_te == y_true_te)

                    # ----- Evaluate calibration
                    # NB: This is the \ell_2, debiased, marginal calibration error
                    ce_m_te = cal.get_calibration_error(model_probs_te, y_true_te,
                                                        p=2, debias=True, mode='marginal')
                    ce_m_TS = cal.get_calibration_error(oracle_combiner_TS.calibrate(model_probs_te), y_true_te,
                                                        p=2, debias=True, mode='marginal')
                    ce_m_dirichlet = cal.get_calibration_error(oracle_combiner_dirichlet.calibrate(model_probs_te),
                                                               y_true_te,
                                                               p=2, debias=True, mode='marginal')
                    ce_combo_TS = cal.get_calibration_error(y_comb_prob_TS, y_true_te,
                                                            p=2, debias=True, mode='marginal')
                    ce_combo_dirichlet = cal.get_calibration_error(y_comb_prob_dirichlet, y_true_te,
                                                                   p=2, debias=True, mode='marginal')

                    # NB: This is the usual ECE
                    ece_m_te = cal.get_ece(model_probs_te, y_true_te)
                    ece_m_TS = cal.get_ece(oracle_combiner_TS.calibrate(model_probs_te), y_true_te)
                    ece_m_dirichlet = cal.get_ece(oracle_combiner_dirichlet.calibrate(model_probs_te),
                                                  y_true_te)
                    ece_combo_TS = cal.get_ece(y_comb_prob_TS, y_true_te)
                    ece_combo_dirichlet = cal.get_ece(y_comb_prob_dirichlet, y_true_te)

                    # Write results to CSV
                    with open(out_fpath, 'a', newline='') as f:
                        # TODO : Save model name, noise level, epoch
                        writer = csv.writer(f)
                        writer.writerow([model_name, noise_level, epoch, i,
                                         acc_h_te, acc_m_te, acc_comb_oracle_TS, acc_comb_oracle_dirichlet,
                                         ce_m_te, ce_m_TS, ce_m_dirichlet, ce_combo_TS, ce_combo_dirichlet,
                                         ece_m_te, ece_m_TS, ece_m_dirichlet, ece_combo_TS, ece_combo_dirichlet])


if __name__ == '__main__':
    out_fpath = './output/'
    run_experiment(out_fpath)
