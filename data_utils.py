import os
from combination_methods import *
import numpy as np
import pandas as pd

rng = np.random.default_rng(1234)


def load_CIFAR10H(model_name):
    """ Loads the CIFAR-10H predictions (human and model) and true labels.
    """
    dirname = os.path.dirname(__file__)
    if model_name == 'r_low_acc':
        data_path = os.path.join(dirname, 'data/cifar10h/human_model_truth_cifar10h.csv')
        data = np.genfromtxt(data_path, delimiter=',')

        human_counts = data[:, :10]
        model_probs = data[:, 10:20]
        true_labels = data[:, -1]

        true_labels -= 1  # data has labels 1-10 -- shifting so that they are zero-indexed.
    else:
        data_path = os.path.join(dirname, f'data/cifar10h/{model_name}.csv')
        data = np.genfromtxt(data_path, delimiter=',')

        true_labels = data[:, 0]
        human_counts = data[:, 1:11]
        model_probs = data[:, 11:]

    true_labels = true_labels.astype(int)

    return human_counts, model_probs, true_labels


def load_CIFAR10H_individual(model_name):
    """ Loads the CIFAR-10H predictions, but keeps track of the individual IDs.
    """
    dirname = os.path.dirname(__file__)

    # ---- Model data
    if model_name == 'r_low_acc':
        data_path = os.path.join(dirname, 'data/cifar10h/human_model_truth_cifar10h.csv')
        data = np.genfromtxt(data_path, delimiter=',')
        model_probs = data[:, 10:20]
    else:
        data_path = os.path.join(dirname, f'data/cifar10h/{model_name}.csv')
        data = np.genfromtxt(data_path, delimiter=',')
        model_probs = data[:, 11:]

    # ----- Human data
    # Load raw human-generated labels
    human_data_path = os.path.join(dirname, 'data/cifar10h/cifar10h-raw.csv')
    human_data = pd.read_csv(human_data_path)
    # drop attention checks
    human_data = human_data[human_data['is_attn_check'] == 0]
    # create dict which maps annotator_id --> [y_h, y_true, model_probs]
    # consisting of the individual's guesses, true labels, and model probabilities
    # (for the images labeled by the person)
    annotator_ids = human_data['annotator_id'].unique()
    n_annotators = annotator_ids.size
    individual_level_data = dict.fromkeys(np.arange(n_annotators))
    for i, idx in enumerate(annotator_ids):
        single_annotator_data = human_data[human_data['annotator_id'] == idx]
        image_idxs = single_annotator_data['cifar10_test_test_idx'].to_numpy()
        model_probs_single_annotator = model_probs[image_idxs]

        individual_level_data[i] = {'y_h': single_annotator_data['chosen_label'].to_numpy(),
                                    'model_probs': model_probs_single_annotator,
                                    'y_m': np.argmax(model_probs_single_annotator, axis=1),
                                    'y_true': single_annotator_data['true_label'].to_numpy(),

                                    }

    return individual_level_data


def load_old_noisy_imagenet(noise_level, model_name, n_epochs=None, noise_type='phase', reaction_time=False):
    """
    """
    assert noise_type in ['phase'], 'Invalid noise type'
    assert noise_level in [80, 95, 110, 125], 'Invalid noise level'
    assert model_name in ['alexnet', 'densenet161', 'googlenet', 'resnet152', 'vgg19'], 'Invalid model name'

    image_labels = ['airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair',
                    'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
    image_labels_numeric = np.arange(0, 16)

    data_dir = './data/old_noisy_imagenet_data'
    # data_dir = '../data/old_noisy_imagenet_data'
    human_data_fpath = data_dir + '/human_only_classification_6per_img_export.csv'
    model_data_prefix = '/imagenet_0016_category_phase_noise_all_predictions_'
    epoch_suffix = {None: 'baseline.csv',
                    0: 'epoch00.csv',
                    1: 'epoch01.csv',
                    10: 'epoch10.csv'}
    model_data_fpath = data_dir + model_data_prefix + epoch_suffix[n_epochs]

    human_data = pd.read_csv(human_data_fpath)
    human_data.replace(to_replace=image_labels, value=image_labels_numeric, inplace=True)
    human_data = human_data[human_data['noise_level'] == noise_level]
    columns = ['participant_id', 'image_name', 'image_category', 'participant_classification']
    if reaction_time:
        columns.append('classification_time')
    human_data = human_data[columns]
    if reaction_time:
        arbitrary_reaction_time_cutoff = 650  # 650 ms
        human_data = human_data[human_data.classification_time > arbitrary_reaction_time_cutoff]

    model_data = pd.read_csv(model_data_fpath)
    model_data.replace(to_replace=image_labels, value=image_labels_numeric, inplace=True)
    model_data = model_data[(model_data['noise_type'] == noise_type) &
                            (model_data['noise_level'] == noise_level) &
                            (model_data['model_name'] == model_name)]
    model_data.drop(columns=['noise_type', 'noise_level', 'model_name', 'correct', 'category'], inplace=True)

    # Merge based on input image
    dataset = pd.merge(human_data, model_data, on='image_name', how='left')

    # Map to numpy
    human_ids = dataset['participant_id'].to_numpy(dtype=int)
    y_h = dataset['participant_classification'].to_numpy(dtype=int)
    y_true = dataset['image_category'].to_numpy(dtype=int)
    # Get model_probs in numeric order, i.e. model_probs[0][0] corresponds to 'airplane'
    model_probs = dataset[image_labels].to_numpy(dtype=float)
    # Normalize each model output (via summing)
    model_probs /= model_probs.sum(axis=1, keepdims=True)

    if reaction_time:
        reaction_time = dataset['classification_time'].to_numpy(dtype=float)
        return y_true, y_h, model_probs, reaction_time
    return y_true, y_h, model_probs


def load_noisy_imagenet(model_name, noise_level, model_acc_level):
    model_acc_level = model_acc_level.lower()
    assert model_acc_level in ['low', 'med', 'high']
    assert noise_level in [80, 95, 110, 125]
    if model_name == 'vgg19':
        model_name_dict = {'low': 'vgg19_01',
                           'med': 'vgg19_06',
                           'high': 'vgg19_48'}
    elif model_name == 'googlenet':
        model_name_dict = {'low': 'googlenet_01',
                           'med': 'googlenet_06',
                           'high': 'googlenet_47'}
    else:
        raise NotImplementedError

    # Read data CSVs
    data_path_model = f'./data/noisy_imagenet/{model_name}.csv'
    data_path_human = './data/noisy_imagenet/human_only_classification_6per_img_export.csv'
    data_model = pd.read_csv(data_path_model)
    data_human = pd.read_csv(data_path_human)

    image_labels = ['airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair',
                    'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
    image_labels_numeric = np.arange(0, 16)

    # Replace string labels with numeric labels and get appropriate subset of data
    data_human.replace(to_replace=image_labels, value=image_labels_numeric, inplace=True)
    data_human = data_human[data_human['noise_level'] == noise_level]
    data_human = data_human[['image_name', 'image_category', 'participant_classification']]

    #  Get appropriate subset of model data
    data_model.rename(columns=dict(zip(image_labels, image_labels_numeric)), inplace=True)
    data_model = data_model[(data_model['noise_level'] == noise_level) &
                            (data_model['model_name'] == model_name_dict[model_acc_level])]
    data_model = data_model[['image_name'] + image_labels_numeric.tolist()]

    dataset = pd.merge(data_human, data_model, how='left')
    dataset.drop(columns=['image_name'], inplace=True)

    y_true = dataset['image_category'].to_numpy().astype(int)
    y_h = dataset['participant_classification'].to_numpy().astype(int)
    model_probs = dataset[image_labels_numeric].to_numpy()

    return y_true, y_h, model_probs


def load_noisy_imagenet_logits(noise_level, model_acc_level):
    model_acc_level = model_acc_level.lower()
    assert model_acc_level in ['low', 'med', 'high']
    assert noise_level in [80, 95, 110, 125]
    model_name_dict = {'low': 'vgg19_01',
                       'med': 'vgg19_06',
                       'high': 'vgg19_48'}

    # Read data CSVs
    data_path_model = './data/noisy_imagenet/vgg19_logits.csv'
    data_path_human = './data/noisy_imagenet/human_only_classification_6per_img_export.csv'
    data_model = pd.read_csv(data_path_model)
    data_human = pd.read_csv(data_path_human)

    image_labels = ['airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair',
                    'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
    image_labels_numeric = np.arange(0, 16)

    # Replace string labels with numeric labels and get appropriate subset of data
    data_human.replace(to_replace=image_labels, value=image_labels_numeric, inplace=True)
    data_human = data_human[data_human['noise_level'] == noise_level]
    data_human = data_human[['image_name', 'image_category', 'participant_classification']]

    #  Get appropriate subset of model data
    data_model.rename(columns=dict(zip(image_labels, image_labels_numeric)), inplace=True)
    data_model = data_model[(data_model['noise_level'] == noise_level) &
                            (data_model['model_name'] == model_name_dict[model_acc_level])]
    data_model = data_model[['image_name'] + image_labels_numeric.tolist()]

    dataset = pd.merge(data_human, data_model, how='left')
    dataset.drop(columns=['image_name'], inplace=True)

    y_true = dataset['image_category'].to_numpy().astype(int)
    y_h = dataset['participant_classification'].to_numpy().astype(int)
    model_probs = dataset[image_labels_numeric].to_numpy()

    return y_true, y_h, model_probs


def load_noisy_imagenet_individual(model_name, noise_level, model_acc_level):
    dirname = os.path.dirname(__file__)

    model_acc_level = model_acc_level.lower()
    assert model_acc_level in ['low', 'med', 'high']
    assert noise_level in [80, 95, 110, 125]
    if model_name == 'vgg19':
        model_name_dict = {'low': 'vgg19_01',
                           'med': 'vgg19_06',
                           'high': 'vgg19_48'}
    elif model_name == 'googlenet':
        model_name_dict = {'low': 'googlenet_01',
                           'med': 'googlenet_06',
                           'high': 'googlenet_47'}
    else:
        raise NotImplementedError

    # Read data CSVs
    data_path_model = os.path.join(dirname, f'./data/noisy_imagenet/{model_name}.csv')
    data_path_human = os.path.join(dirname, './data/noisy_imagenet/human_only_classification_6per_img_export.csv')
    data_model = pd.read_csv(data_path_model)
    data_human = pd.read_csv(data_path_human)

    image_labels = ['airplane', 'bear', 'bicycle', 'bird', 'boat', 'bottle', 'car', 'cat', 'chair',
                    'clock', 'dog', 'elephant', 'keyboard', 'knife', 'oven', 'truck']
    image_labels_numeric = np.arange(0, 16)

    # Replace string labels with numeric labels and get appropriate subset of data
    data_human.replace(to_replace=image_labels, value=image_labels_numeric, inplace=True)
    data_human = data_human[data_human['noise_level'] == noise_level]

    #  Get appropriate subset of model data
    data_model.rename(columns=dict(zip(image_labels, image_labels_numeric)), inplace=True)
    data_model = data_model[(data_model['noise_level'] == noise_level) &
                            (data_model['model_name'] == model_name_dict[model_acc_level])]
    data_model = data_model[['image_name'] + image_labels_numeric.tolist()]

    # ----- Human data
    # Load raw human-generated labels
    # create dict which maps annotator_id --> [y_h, y_true, model_probs]
    # consisting of the individual's guesses, true labels, and model probabilities
    # (for the images labeled by the person)
    annotator_ids = data_human['participant_id'].unique()
    n_annotators = annotator_ids.size
    individual_level_data = dict.fromkeys(np.arange(n_annotators))
    for i, idx in enumerate(annotator_ids):
        single_annotator_data = data_human[data_human['participant_id'] == idx]
        image_idxs = single_annotator_data['image_name'].to_numpy()

        # Iterate over images human labeled, get model predictions
        model_probs_single_annotator = np.empty(shape=(image_idxs.size, 16))
        for j, image_idx in enumerate(image_idxs):
            model_probs_single_image = data_model[data_model['image_name'] == image_idx]
            model_probs_single_image = model_probs_single_image[image_labels_numeric].to_numpy()[0]
            model_probs_single_annotator[j] = model_probs_single_image

        individual_level_data[i] = {'y_h': single_annotator_data['participant_classification'].to_numpy(),
                                    'model_probs': model_probs_single_annotator,
                                    'y_m': np.argmax(model_probs_single_annotator, axis=1),
                                    'y_true': single_annotator_data['image_category'].to_numpy(),
                                    }

    return individual_level_data
