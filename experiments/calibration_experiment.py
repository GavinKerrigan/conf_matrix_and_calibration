import sys
sys.path.insert(0, '../')

from data_utils import *
from utils import *
from combination_methods import *
from tqdm.auto import tqdm
import torch
from sklearn.model_selection import train_test_split
from metrics import *
import csv
import numpy as np
import os


# Generates the data for Appendix C in our paper.

def _run_experiment(y_h=None, model_probs=None, y_true=None, **kwargs):
    seed = kwargs.pop('seed', 0)
    n_runs = kwargs.pop('n_runs', 25)
    test_size = kwargs.pop('test_size', 0.3)
    calibration_methods = kwargs.pop('calibration_methods', ['none'])
    calibration_metrics = kwargs.pop('calibration_metrics', {'ECE': get_ECE})
    output_file_acc = kwargs.pop('output_file_acc', './acc.csv')
    output_file_calibration = kwargs.pop('output_file_calibration', './cal.csv')

    acc_data = []
    cal_data = []
    for i in tqdm(range(n_runs), leave=False, desc='Runs'):
        # Train/test split
        y_h_tr, y_h_te, model_probs_tr, model_probs_te, y_true_tr, y_true_te = train_test_split(
            y_h, model_probs, y_true, test_size=test_size, random_state=i * seed)

        acc_h = get_acc(y_h_te, y_true_te)
        acc_m = get_acc(np.argmax(model_probs_te, axis=1), y_true_te)

        _acc_data = [acc_h, acc_m]
        _cal_data = []
        for calibration_method in calibration_methods:
            if calibration_method == 'confusion':
                combiner = DoubleConfusionCombiner()
                combiner.fit(model_probs_tr, y_h_tr, y_true_tr)
            else:
                combiner = OracleCombiner(calibration_method=calibration_method)
                combiner.fit(model_probs_tr, y_h_tr, y_true_tr)

            y_comb_te = combiner.combine(model_probs_te, y_h_te)
            acc_comb = get_acc(y_comb_te, y_true_te)
            _acc_data.append(acc_comb)

            model_probs_calibrated_te = combiner.calibrate(model_probs_te)
            y_comb_prob_te = combiner.combine_proba(model_probs_te, y_h_te)
            for metric, fxn in calibration_metrics.items():
                cal_m = fxn(model_probs_calibrated_te, y_true_te)
                cal_comb = fxn(y_comb_prob_te, y_true_te)
                _cal_data.append([calibration_method, metric, cal_m, cal_comb])

        acc_data += [_acc_data]
        cal_data += _cal_data

    # Save data to CSV
    header_acc = ['human', 'model'] + [f'comb {cal_m}' for cal_m in calibration_methods]
    with open(output_file_acc, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_acc)
        writer.writerows(acc_data)
    header_cal = ['calibration method', 'metric', 'model', 'comb']
    with open(output_file_calibration, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header_cal)
        writer.writerows(cal_data)


def run_experiment_cifar10(out_fpath=None, experiment_args=None, seed=0):
    model_names = ['r_low_acc', 'resnet-110', 'preresnet-110', 'densenet-bc-L190-k40']
    for model_name in tqdm(model_names, desc='Models', leave=True):
        # Specify output files
        output_file_acc = out_fpath + f'{model_name}_accuracy.csv'
        output_file_calibration = out_fpath + f'{model_name}_calibration.csv'
        assert not os.path.exists(output_file_acc), 'Output filepath already exists'
        assert not os.path.exists(output_file_calibration), 'Output filepath already exists'
        experiment_args['output_file_acc'] = output_file_acc
        experiment_args['output_file_calibration'] = output_file_calibration

        # Load data
        human_counts, model_probs, y_true = load_CIFAR10H(model_name)
        y_h = simulate_single_human(human_counts, seed=seed)

        _run_experiment(y_h=y_h, model_probs=model_probs, y_true=y_true, **experiment_args)


def run_experiment_noisy_imagenet(out_fpath=None, experiment_args=None, seed=0):
    model_acc_levels = ['high']  # ['low', 'med', 'high']
    noise_levels = [80, 95, 110, 125]
    model_names = ['vgg19', 'googlenet']

    for model_name in model_names:
        for model_level in tqdm(model_acc_levels, desc='Models', leave=True):
            for noise_level in tqdm(noise_levels, desc='Noise Levels'):
                # Specify output files
                output_file_acc = out_fpath + f'{model_name}_n{noise_level}_l{model_level}_accuracy.csv'
                output_file_calibration = out_fpath + f'{model_name}_n{noise_level}_l{model_level}_calibration.csv'
                assert not os.path.exists(output_file_acc), 'Output filepath already exists'
                assert not os.path.exists(output_file_calibration), 'Output filepath already exists'
                experiment_args['output_file_acc'] = output_file_acc
                experiment_args['output_file_calibration'] = output_file_calibration

                # Load data
                y_true, y_h, model_probs = load_noisy_imagenet(model_name, noise_level, model_level)

                _run_experiment(y_h=y_h, model_probs=model_probs, y_true=y_true, **experiment_args)


if __name__ == '__main__':
    seed = 9658
    torch.manual_seed(seed)
    np.random.seed(seed)

    calibration_methods = ['none', 'confusion', 'temperature scaling', 'ensemble temperature scaling', 'imax binning']
    """
    calibration_metrics = {'ECE width': lambda probs, y: get_ECE(probs, y, mode='width'),
                           'ECE mass': lambda probs, y: get_ECE(probs, y, mode='mass'),
                           'cwECE thresh width': lambda probs, y: get_cw_ECE(probs, y, mode='width'),
                           'cwECE thresh mass': lambda probs, y: get_cw_ECE(probs, y, mode='mass'),
                           'cwECE nothresh width': lambda probs, y: get_cw_ECE(probs, y, mode='width',
                                                                               threshold_mode=None),
                           'cwECE nothresh mass': lambda probs, y: get_cw_ECE(probs, y, mode='mass',
                                                                              threshold_mode=None),
                           'kumar MCE': get_MCE,
                           'kumar MCE (bin)': lambda probs, y: cal.get_binning_ce(probs, y,
                                                                                  p=1, debias=False, mode='marginal'),
                           'kumar MCE (scale)': lambda probs, y: cal.lower_bound_scaling_ce(probs, y,
                                                                                            p=1, debias=False,
                                                                                            mode='marginal'),
                           'kumar ECE': cal.get_ece}
    """
    calibration_metrics = {'ECE (W)': lambda probs, y: get_ECE(probs, y, mode='width'),
                           'ECE (M)': lambda probs, y: get_ECE(probs, y, mode='mass'),
                           'cwECE (WT)': lambda probs, y: get_cw_ECE(probs, y, mode='width'),
                           'cwECE (MT)': lambda probs, y: get_cw_ECE(probs, y, mode='mass'),
                           'cwECE (WNT)': lambda probs, y: get_cw_ECE(probs, y, mode='width', threshold_mode=None),
                           'cwECE (MNT)': lambda probs, y: get_cw_ECE(probs, y, mode='mass', threshold_mode=None),
                           'NLL': get_NLL}

    args = {'n_runs': 25,
            'test_size': 0.3,
            'calibration_methods': calibration_methods,
            'calibration_metrics': calibration_metrics,
            'seed': seed
            }

    out_fpath = './output/cifar10h/final/fully_sup_CI/'
    run_experiment_cifar10(out_fpath=out_fpath, experiment_args=args, seed=seed)

    out_fpath = './output/noisy_imagenet/final/fully_sup_CI/'
    run_experiment_noisy_imagenet(out_fpath=out_fpath, experiment_args=args, seed=seed)
