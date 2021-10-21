from calibrators import *
import torch
from utils import *
from tqdm.auto import tqdm
import warnings
import torch
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from utils import *
from imax_calib.utils import safe_log_diff
from sklearn.linear_model import LogisticRegression
from calibrators import *

# This file implements the various combination methods.


class EMCombiner:
    """ An abstract class for EM combination methods.
    """
    def __init__(self, calibration_method):
        self.calibrator = None
        self.confusion_matrix = None  # conf[i, j] is assumed to be P(h = i | Y = j)

        self.n_train_u = None  # Amount of unlabeled training data
        self.n_train_l = None  # Amount of labeled training data
        self.n_cls = None  # Number of classes

        self.eps = 1e-50  # Tiny value for clipping

        self.calibration_method = calibration_method

    def initialize_confusion_matrix(self, p=0.9):
        # Creates an initial estimate of confusion matrix
        # Diagonal values are given by p, and off-diagonal entries are (1. - p) / (n-1) so rows are normalized
        off_diag = (1. - p) / (self.n_cls - 1)
        init_conf = np.ones((self.n_cls, self.n_cls)) * off_diag + np.eye(self.n_cls) * (p - off_diag)

        return init_conf

    def e_step(self, probs_m, y_h, conf_h):
        weight_matrix = np.empty((self.n_train_u, self.n_cls))  # Entry [i, j] is P(Y = j | h_i, m_i, theta_t)
        for i in range(self.n_train_u):
            weight_matrix[i] = probs_m[i] * conf_h[y_h[i]]
            normalizer = weight_matrix[i].sum()
            weight_matrix[i] /= normalizer

        return weight_matrix

    def get_calibrator(self, **kwargs):
        if self.calibration_method == 'temperature scaling':
            raise NotImplementedError
            #return SoftTSCalibrator()
        elif self.calibration_method == 'hard temperature scaling':
            return TSCalibrator()
        elif self.calibration_method == 'dirichlet':
            # reg_norm : bool, true if regularization is used
            # reg_mu : None or float, if None regular L2 regularization is used
            # reg_lambda : 0 or float, l2 regularization term
            from dirichlet_python.dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
            # Default parameter suggested by the authors, c.f. https://arxiv.org/pdf/1910.12656.pdf , suppl. figure 8
            reg_lambda = 1e-3
            return FullDirichletCalibrator(reg_norm=True, reg_lambda=reg_lambda, reg_mu=None)
        elif self.calibration_method == 'MAP temperature scaling':
            mu_beta = kwargs.pop('mu_beta', 0.5)
            sigma_beta = kwargs.pop('sigma_beta', 0.5)
            return TSCalibratorMAP(prior_mu=mu_beta, prior_sigma=sigma_beta)

    def calibrate(self, model_probs):
        return self.calibrator.calibrate(model_probs)

    def combine_proba(self, model_probs, y_h):
        """ Combines model probabilities with hard labels via the calibrate-confuse equation given the confusion matrix.

        Args:
            p_m: Array of model probabilities ; shape (n_samples, n_classes)
            y_h: List of hard labels ; shape (n_samples,)

        Returns:
            Normalized posterior probabilities P(Y | m, h). Entry [i, j] is P(Y = j | h_i, m_i)
        """
        assert model_probs.shape[0] == y_h.size, 'Size mismatch between model probs and human labels'
        assert model_probs.shape[1] == self.n_cls, 'Size mismatch between model probs and number of classes'

        n_samples = model_probs.shape[0]
        calibrated_model_probs = self.calibrate(model_probs)
        calibrated_model_probs = np.clip(calibrated_model_probs, self.eps, None)

        y_comb = np.empty((n_samples, self.n_cls))
        for i in range(n_samples):
            y_comb[i] = calibrated_model_probs[i] * self.confusion_matrix[y_h[i]]

        #  Normalize probabilities
        y_comb /= np.sum(y_comb, axis=1, keepdims=True)
        return y_comb

    def combine(self, model_probs, y_h):
        """ Combines model probs and y_h to return hard labels
        """
        y_comb_soft = self.combine_proba(model_probs, y_h)
        return np.argmax(y_comb_soft, axis=1)


class UnsupervisedEMCombiner(EMCombiner):
    """ Fully unsupervised EM combination (fit using maximum likelihood)
    """

    def __init__(self, calibration_method='temperature scaling'):
        super().__init__(calibration_method)

    def fit(self, model_probs, y_h, num_steps=750):
        # Initialize
        self.n_train_u, self.n_cls = model_probs.shape
        conf_h = self.initialize_confusion_matrix(self.n_cls)
        model_probs_clipped = np.clip(model_probs, self.eps, None)
        model_logits = np.log(model_probs_clipped)
        calibrated_model_probs = np.copy(model_probs_clipped)

        # Optimization parameters
        progbar = tqdm(total=num_steps, leave=False, desc='EM Steps (Unsupervised)')
        eps = 1e-15  # Clipping parameter to avoid log(0)
        loss_rel_tol = 1e-6  # Minimum relative change in loss - for early stopping
        step = 0
        prev_loss = 1e15
        loss_tr = []
        min_steps = 50

        converged = False
        while not converged:
            weight_matrix = self.e_step(calibrated_model_probs, y_h, conf_h)
            calibrator, conf_h = self.m_step(y_h, model_logits, weight_matrix)

            # Evaluate loss
            calibrated_model_probs = calibrator.calibrate(model_probs)
            calibrated_model_probs_clipped = np.clip(calibrated_model_probs, eps, 1)
            conf_h_clipped = np.clip(conf_h[y_h], eps, 1)
            loss = np.sum(weight_matrix * (np.log(calibrated_model_probs_clipped) + np.log(conf_h_clipped)))

            step += 1
            if step > num_steps:
                warnings.warn('(Unsupervised EM) Maximum number of steps reached -- may not have converged')
            converged = (step > num_steps) or (np.abs(loss - prev_loss) / np.abs(prev_loss) < loss_rel_tol)
            if step < min_steps:
                converged = False

            prev_loss = loss
            loss_tr.append(loss)

            progbar.update(1)
        progbar.close()

        self.calibrator = calibrator
        self.confusion_matrix = conf_h

    def m_step(self, y_h, model_logits, weight_matrix):
        # Get new confusion matrix parameters
        confusion_matrix = np.empty((self.n_cls, self.n_cls))
        for b in range(self.n_cls):
            for a in range(self.n_cls):
                # Get entry P(h = a | Y = b)
                confusion_matrix[a, b] = weight_matrix[y_h == a, b].sum()

        confusion_matrix = np.clip(confusion_matrix, self.eps, None)
        normalizer = np.sum(confusion_matrix, axis=0, keepdims=True)
        confusion_matrix /= normalizer

        # Get new calibration parameters
        calibrator = self.get_calibrator()
        calibrator.fit(model_logits, weight_matrix)

        return calibrator, confusion_matrix


class UnsupervisedEMCombinerMAP(EMCombiner):
    """ Fully unsupervised EM Combination (fit using MAP estimation)
    NB: This is referred to in our paper as "P+L-EM"
    """

    def __init__(self, calibration_method='MAP temperature scaling', diag_acc=0.75, strength=1., mu_beta=0.5, sigma_beta=0.5):
        super().__init__(calibration_method)

        self.diag_acc = diag_acc
        self.strength = strength
        self.prior_alpha = None
        self.prior_beta = None
        self.mu_beta = mu_beta
        self.sigma_beta = sigma_beta

    def fit(self, model_probs, y_h, num_steps=750):
        # Initialize
        self.n_train_u, self.n_cls = model_probs.shape
        self.prior_alpha, self.prior_beta = get_dirichlet_params(self.diag_acc, self.strength, self.n_cls)
        conf_h = self.initialize_confusion_matrix(self.n_cls)
        model_probs_clipped = np.clip(model_probs, self.eps, None)
        model_logits = np.log(model_probs_clipped)
        calibrated_model_probs = np.copy(model_probs_clipped)

        # Optimization parameters
        progbar = tqdm(total=num_steps, leave=False, desc='EM Steps (Unsupervised)')
        eps = 1e-15  # Clipping parameter to avoid log(0)
        loss_rel_tol = 1e-6  # Minimum relative change in loss - for early stopping
        step = 0
        prev_loss = 1e15
        loss_tr = []
        min_steps = 50

        converged = False
        while not converged:
            weight_matrix = self.e_step(calibrated_model_probs, y_h, conf_h)
            calibrator, conf_h = self.m_step(y_h, model_logits, weight_matrix)

            # Evaluate loss
            calibrated_model_probs = calibrator.calibrate(model_probs)
            calibrated_model_probs_clipped = np.clip(calibrated_model_probs, eps, 1)
            conf_h_clipped = np.clip(conf_h[y_h], eps, 1)
            loss = np.sum(weight_matrix * (np.log(calibrated_model_probs_clipped) + np.log(conf_h_clipped)))

            step += 1
            if step > num_steps:
                warnings.warn('(Unsupervised EM) Maximum number of steps reached -- may not have converged')
            converged = (step > num_steps) or (np.abs(loss - prev_loss) / np.abs(prev_loss) < loss_rel_tol)
            if step < min_steps:
                converged = False

            prev_loss = loss
            loss_tr.append(loss)

            progbar.update(1)
        progbar.close()

        self.calibrator = calibrator
        self.confusion_matrix = conf_h

    def m_step(self, y_h, model_logits, weight_matrix):
        # Get new confusion matrix parameters
        confusion_matrix = np.empty((self.n_cls, self.n_cls))
        for b in range(self.n_cls):
            for a in range(self.n_cls):
                # Get entry P(h = a | Y = b)
                confusion_matrix[a, b] = weight_matrix[y_h == a, b].sum()
                if a == b:
                    confusion_matrix[a, b] += self.prior_alpha
                else:
                    confusion_matrix[a, b] += self.prior_beta

        confusion_matrix = np.clip(confusion_matrix, self.eps, None)
        normalizer = np.sum(confusion_matrix, axis=0, keepdims=True)
        confusion_matrix = (confusion_matrix - np.eye(self.n_cls)) / (normalizer - self.n_cls)

        # Get new calibration parameters
        calibrator = self.get_calibrator(mu_beta=self.mu_beta, sigma_beta=self.sigma_beta)
        calibrator.fit(model_logits, weight_matrix)

        return calibrator, confusion_matrix


class SemiSupervisedEMCombiner(EMCombiner):
    """ Semi-Supervised EM Combination
    """

    def __init__(self, calibration_method='temperature scaling', unsupervised_weight=1.):
        super().__init__(calibration_method)
        self.unsupervised_weight = unsupervised_weight  # Used to down-weight the unsupervised data in the M-step

    def fit(self, model_probs_u, y_h_u, model_probs_l, y_h_l, y_true_l, num_steps=750):
        # Initialize
        self.n_train_u, self.n_cls = model_probs_u.shape
        self.n_train_l = model_probs_l.shape[0]
        conf_h = self.initialize_confusion_matrix(self.n_cls)
        model_probs_u_clipped = np.clip(model_probs_u, self.eps, None)
        model_probs_l_clipped = np.clip(model_probs_l, self.eps, None)
        model_logits_u, model_logits_l = np.log(model_probs_u_clipped), np.log(model_probs_l_clipped)
        calibrated_model_probs_u = np.copy(model_probs_u_clipped)
        calibrated_model_probs_l = np.copy(model_probs_l_clipped)

        nll = nn.NLLLoss()

        # Optimization parameters
        progbar = tqdm(total=num_steps, leave=False, desc='EM Steps (Semi-Supervised)')
        eps = 1e-15  # Clipping parameter to avoid log(0)
        loss_rel_tol = 1e-6  # Minimum relative change in loss - for early stopping
        step = 0
        prev_loss = 1e15
        loss_tr = []
        min_steps = 50

        converged = False
        while not converged:
            # Weight matrix for unlabeled examples
            weight_matrix_u = self.e_step(calibrated_model_probs_u, y_h_u, conf_h)
            calibrator, conf_h = self.m_step(y_h_u, model_logits_u, weight_matrix_u,
                                             y_h_l, model_logits_l, y_true_l)

            calibrated_model_probs_u = calibrator.calibrate(model_probs_u)
            calibrated_model_probs_l = calibrator.calibrate(model_probs_l)

            # Evaluate loss
            loss_u = np.sum(weight_matrix_u * (np.log(np.clip(calibrated_model_probs_u, eps, 1)) +
                                               np.log(np.clip(conf_h[y_h_u], eps, 1))))
            loss_l = nll(torch.log(torch.from_numpy(np.clip(calibrated_model_probs_l, eps, 1))),
                         torch.from_numpy(y_true_l)) + np.log(np.clip(conf_h[y_h_l, y_true_l], eps, 1)).sum()
            loss = self.unsupervised_weight * loss_u + loss_l

            step += 1
            if step > num_steps:
                warnings.warn('(SemiSup EM) Maximum number of steps reached -- may not have converged')
            converged = (step > num_steps) or (np.abs(loss - prev_loss) / np.abs(prev_loss) < loss_rel_tol)
            if step < min_steps:
                converged = False

            prev_loss = loss
            loss_tr.append(loss)

            progbar.update(1)
        progbar.close()

        self.calibrator = calibrator
        self.confusion_matrix = conf_h

    def m_step(self, y_h_u, model_logits_u, weight_matrix_u, y_h_l, model_logits_l, y_true_l):
        # Get new confusion matrix parameters
        # Get entry psi_ab = P(h = a | y = b)
        confusion_matrix = np.empty((self.n_cls, self.n_cls))
        for b in range(self.n_cls):
            for a in range(self.n_cls):
                # Sum of soft entries P(Y = b | h_i, m_i) where h_i = a
                confusion_matrix[a, b] = self.unsupervised_weight * weight_matrix_u[y_h_u == a, b].sum()
                # Count of labeled entries where h_i = a and y_true = b
                confusion_matrix[a, b] += ((y_h_l == a) & (y_true_l == b)).sum()

        confusion_matrix = np.clip(confusion_matrix, self.eps, None)
        normalizer = np.sum(confusion_matrix, axis=0, keepdims=True)
        confusion_matrix /= normalizer

        # Get new calibration parameters
        calibrator = self.get_calibrator()
        calibrator.unsupervised_weight = self.unsupervised_weight
        calibrator.fit(model_logits_u, weight_matrix_u,
                       model_logits_l, y_true_l)

        return calibrator, confusion_matrix


class CalibrateFirstCombiner(EMCombiner):
    """ This will implement the following combination model:

    - First calibrate on labeled data (small amount)
    - Then fit confusion matrix only using semisup-EM
    """

    def __init__(self, calibration_method='hard temperature scaling', unsupervised_weight=1.):
        super().__init__(calibration_method)
        self.unsupervised_weight = unsupervised_weight

    def fit(self, model_probs_u, y_h_u, model_probs_l, y_h_l, y_true_l, num_steps=750):
        # Initialize
        self.n_train_u, self.n_cls = model_probs_u.shape
        self.n_train_l = model_probs_l.shape[0]
        conf_h = self.initialize_confusion_matrix(self.n_cls)
        model_logits_u, model_logits_l = np.log(model_probs_u), np.log(model_probs_l)

        # Fit calibration map on labeled data
        calibrator = self.get_calibrator()
        calibrator.fit(model_logits_l, y_true_l)
        calibrated_model_probs_u = calibrator.calibrate(model_probs_u)
        calibrated_model_probs_l = calibrator.calibrate(model_probs_l)

        nll = nn.NLLLoss()

        # Optimization parameters
        progbar = tqdm(total=num_steps, leave=False, desc='EM Steps (Calibrate First)')
        eps = 1e-15  # Clipping parameter to avoid log(0)
        loss_rel_tol = 1e-6  # Minimum relative change in loss - for early stopping
        step = 0
        prev_loss = 1e15
        loss_tr = []
        min_steps = 50

        converged = False
        while not converged:
            # Weight matrix for unlabeled examples
            weight_matrix_u = self.e_step(calibrated_model_probs_u, y_h_u, conf_h)
            conf_h = self.m_step(y_h_u, weight_matrix_u, y_h_l, y_true_l)

            # Evaluate loss
            loss_u = np.sum(weight_matrix_u * (np.log(np.clip(calibrated_model_probs_u, eps, 1)) +
                                               np.log(np.clip(conf_h[y_h_u], eps, 1))))
            loss_l = nll(torch.log(torch.from_numpy(np.clip(calibrated_model_probs_l, eps, 1))),
                         torch.from_numpy(y_true_l)) + np.log(np.clip(conf_h[y_h_l, y_true_l], eps, 1)).sum()
            loss = loss_u + loss_l

            step += 1
            if step > num_steps:
                warnings.warn('(SemiSup EM) Maximum number of steps reached -- may not have converged')
            converged = (step > num_steps) or (np.abs(loss - prev_loss) / np.abs(prev_loss) < loss_rel_tol)
            if step < min_steps:
                converged = False

            prev_loss = loss
            loss_tr.append(loss)

            progbar.update(1)
        progbar.close()

        self.calibrator = calibrator
        self.confusion_matrix = conf_h

    def m_step(self, y_h_u, weight_matrix_u, y_h_l, y_true_l):
        # Only updates the confusion matrix, calibration parameters left fix
        # Get new confusion matrix parameters
        # Get entry psi_ab = P(h = a | y = b)
        confusion_matrix = np.empty((self.n_cls, self.n_cls))
        for b in range(self.n_cls):
            for a in range(self.n_cls):
                # Sum of soft entries P(Y = b | h_i, m_i) where h_i = a
                confusion_matrix[a, b] = self.unsupervised_weight * weight_matrix_u[y_h_u == a, b].sum()
                # Count of labeled entries where h_i = a and y_true = b
                confusion_matrix[a, b] += ((y_h_l == a) & (y_true_l == b)).sum()

        confusion_matrix = np.clip(confusion_matrix, self.eps, None)
        normalizer = np.sum(confusion_matrix, axis=0, keepdims=True)
        confusion_matrix /= normalizer

        return confusion_matrix


class OracleCombiner:
    """ Implements the P+L combination method, fit using maximum likelihood
    """
    def __init__(self, calibration_method='temperature scaling', **kwargs):
        self.calibrator = None
        self.confusion_matrix = None  # conf[i, j] is assumed to be P(h = i | Y = j)

        self.n_train_u = None  # Amount of unlabeled training data
        self.n_train_l = None  # Amount of labeled training data
        self.n_cls = None  # Number of classes

        self.eps = 1e-50

        self.use_cv = False
        self.calibration_method = calibration_method
        if self.calibration_method == 'temperature scaling':
            self.calibrator = TSCalibrator()
        elif self.calibration_method == 'dirichlet':
            # reg_norm : bool, true if regularization is used
            # reg_mu : None or float, if None regular L2 regularization is used
            # reg_lambda : 0 or float, l2 regularization term
            from dirichlet_python.dirichletcal.calib.fulldirichlet import FullDirichletCalibrator
            self.calibrator = FullDirichletCalibrator(reg_norm=True, reg_lambda=0.0, reg_mu=None)
            self.use_cv = True
        elif self.calibration_method == 'ensemble temperature scaling':
            self.calibrator = EnsembleTSCalibrator()
        elif self.calibration_method == 'imax binning':
            mode = kwargs.pop('mode', 'sCW')
            num_bins = kwargs.pop('num_bins', 15)
            self.calibrator = IMaxCalibrator(mode=mode, num_bins=num_bins)
        elif self.calibration_method == 'none':
            self.calibrator = IdentityCalibrator()

    def calibrate(self, model_probs):
        return self.calibrator.calibrate(model_probs)

    def fit(self, model_probs, y_h, y_true):
        self.n_cls = model_probs.shape[1]

        # Estimate human confusion matrix
        # Entry [i, j]  is #(Y = i and h = j)
        conf_h = 1. * confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls))
        # Swap so entry [i, j] is #(h = i and Y = j)
        conf_h = conf_h.T
        conf_h = np.clip(conf_h, self.eps, None)
        normalizer = np.sum(conf_h, axis=0, keepdims=True)
        # Normalize columns so entry [i, j] is P(h = i | Y = j)
        conf_h /= normalizer
        self.confusion_matrix = conf_h

        # Calibrate model probabilities
        if self.use_cv:
            self.fit_calibrator_cv(model_probs, y_true)
        else:
            self.fit_calibrator(model_probs, y_true)

    def fit_bayesian(self, model_probs, y_h, y_true, alpha=0.1, beta=0.1):
        """ This is the "plus one" parameterization, i.e. alpha,beta just need to be > 0
        Really corresponds to a Dirichlet(alpha+1, beta+1, beta+1, . . . ,beta+1) distribution
        """
        self.n_cls = model_probs.shape[1]

        prior_matr = np.eye(self.n_cls) * alpha + (np.ones(self.n_cls) - np.eye(self.n_cls)) * beta

        conf_h = 1. * confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls))
        conf_h += prior_matr
        # Swap so entry [i, j] is #(h = i and Y = j)
        conf_h = conf_h.T
        #conf_h = np.clip(conf_h, self.eps, None)
        normalizer = np.sum(conf_h, axis=0, keepdims=True)
        # Normalize columns so entry [i, j] is P(h = i | Y = j)
        conf_h = conf_h / normalizer
        self.confusion_matrix = conf_h

        # Calibrate model probabilities
        if self.use_cv:
            self.fit_calibrator_cv(model_probs, y_true)
        else:
            self.fit_calibrator(model_probs, y_true)

    def fit_calibrator(self, model_probs, y_true):
        clipped_model_probs = np.clip(model_probs, self.eps, 1)
        model_logits = np.log(clipped_model_probs)
        self.calibrator.fit(model_logits, y_true)

    def fit_calibrator_cv(self, model_probs, y_true):
        # Fits calibration maps that require hyperparameters, using cross-validation
        if self.calibration_method == 'dirichlet':
            reg_lambda_vals = [10., 1., 0., 5e-1, 1e-1, 1e-2, 1e-3]
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
            gscv = GridSearchCV(self.calibrator, param_grid={'reg_lambda': reg_lambda_vals,
                                                             'reg_mu': [None]},
                                cv=skf, scoring='neg_log_loss', refit=True)
            gscv.fit(model_probs, y_true)
            self.calibrator = gscv.best_estimator_
        else:
            raise NotImplementedError

    def combine_proba(self, model_probs, y_h):
        """ Combines model probabilities with hard labels via the calibrate-confuse equation given the confusion matrix.

        Args:
            p_m: Array of model probabilities ; shape (n_samples, n_classes)
            y_h: List of hard labels ; shape (n_samples,)

        Returns:
            Normalized posterior probabilities P(Y | m, h). Entry [i, j] is P(Y = j | h_i, m_i)
        """
        assert model_probs.shape[0] == y_h.size, 'Size mismatch between model probs and human labels'
        assert model_probs.shape[1] == self.n_cls, 'Size mismatch between model probs and number of classes'

        n_samples = model_probs.shape[0]
        calibrated_model_probs = self.calibrate(model_probs)

        y_comb = np.empty((n_samples, self.n_cls))
        for i in range(n_samples):
            y_comb[i] = calibrated_model_probs[i] * self.confusion_matrix[y_h[i]]
            if np.allclose(y_comb[i], 0):  # Handle zero rows
                y_comb[i] = np.ones(self.n_cls) * (1./self.n_cls)

        # Don't forget to normalize :)
        assert np.all(np.isfinite(np.sum(y_comb, axis=1)))
        assert np.all(np.sum(y_comb, axis=1) > 0)
        y_comb /= np.sum(y_comb, axis=1, keepdims=True)
        return y_comb

    def combine(self, model_probs, y_h):
        """ Combines model probs and y_h to return hard labels
        """
        y_comb_soft = self.combine_proba(model_probs, y_h)
        return np.argmax(y_comb_soft, axis=1)


class DoubleConfusionCombiner:
    """ Implements the double-confusion matrix combiner ("L+L") using maximum likelihood inference
    """
    def __init__(self, calibration_method='temperature scaling'):
        self.confusion_matrix_h = None  # entry [i, j] is P(h = i | Y = j)
        self.confusion_matrix_m = None  # entry [i, j] is P(Y = j | m = i)

        self.n_train_u = None  # Amount of unlabeled training data
        self.n_train_l = None  # Amount of labeled training data
        self.n_cls = None  # Number of classes

        self.eps = 1e-50

    def fit(self, model_probs, y_h, y_true):
        self.n_cls = model_probs.shape[1]
        self.n_train_l = y_true.size

        # Estimate human confusion matrix
        # Entry [i, j]  is #(Y = i and h = j)
        conf_h = 1. * confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls))
        # Swap so entry [i, j] is #(h = i and Y = j)
        conf_h = conf_h.T
        conf_h = np.clip(conf_h, self.eps, None)
        normalizer = np.sum(conf_h, axis=0, keepdims=True)
        # Normalize columns so entry [i, j] is P(h = i | Y = j)
        conf_h /= normalizer
        self.confusion_matrix_h = conf_h

        # Estimate model confusion matrix
        y_m = np.argmax(model_probs, axis=1)
        # [i, j] = #(Y = i and m = j)
        conf_m = 1. * confusion_matrix(y_true, y_m, labels=np.arange(self.n_cls))
        conf_m = conf_m.T  # [i, j] = #(m = i and Y = j)
        conf_m = np.clip(conf_m, self.eps, None)
        normalizer = np.sum(conf_m, axis=1, keepdims=True)  # NB: normalize rows here, not columns!
        conf_m /= normalizer
        self.confusion_matrix_m = conf_m

    def combine_proba(self, model_probs, y_h):
        """ Combines model probabilities with hard labels via the calibrate-confuse equation given the confusion matrix.

        Args:
            p_m: Array of model probabilities ; shape (n_samples, n_classes)
            y_h: List of hard labels ; shape (n_samples,)

        Returns:
            Normalized posterior probabilities P(Y | m, h). Entry [i, j] is P(Y = j | h_i, m_i)
        """
        assert model_probs.shape[0] == y_h.size, 'Size mismatch between model probs and human labels'
        assert model_probs.shape[1] == self.n_cls, 'Size mismatch between model probs and number of classes'

        n_samples = model_probs.shape[0]
        y_m = np.argmax(model_probs, axis=1)

        y_comb = np.empty((n_samples, self.n_cls))
        for i in range(n_samples):
            y_comb[i] = self.confusion_matrix_m[y_m[i]] * self.confusion_matrix_h[y_h[i]]

        # Don't forget to normalize :)
        y_comb /= np.sum(y_comb, axis=1, keepdims=True)
        return y_comb

    def combine(self, model_probs, y_h):
        """ Combines model probs and y_h to return hard labels
        """
        y_comb_soft = self.combine_proba(model_probs, y_h)
        return np.argmax(y_comb_soft, axis=1)

    def calibrate(self, model_probs):
        preds = np.argmax(model_probs, axis=1)
        probs = self.confusion_matrix_m[preds, :]
        return probs


class BayesianOracleCombiner(OracleCombiner):
    """ Fully Bayesian P+L combiner
    """
    def __init__(self, diag_acc=0.75, strength=1., mu_beta=0.5, sigma_beta=0.5, **kwargs):
        super().__init__()
        self.calibrator = None
        self.prior_params = {'mu_beta': mu_beta,
                             'sigma_beta': sigma_beta
        }
        self.n_cls = None
        self.diag_acc = diag_acc
        self.strength = strength

    def calibrate(self, model_probs):
        logits = torch.from_numpy(np.clip(model_probs, 1e-50, 1))
        return self.calibrator.calibrate(logits)

    def fit(self, model_probs, y_h, y_true, model_logits=None):
        self.n_cls = model_probs.shape[1]

        # Get posterior distribution over confusion matrix parameters
        alpha, beta = get_dirichlet_params(self.diag_acc, self.strength, self.n_cls)
        prior_matr = np.eye(self.n_cls) * alpha + (np.ones(self.n_cls) - np.eye(self.n_cls)) * beta
        posterior_matr = 1. * confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls))
        posterior_matr += prior_matr
        posterior_matr = posterior_matr.T
        posterior_matr /= np.sum(posterior_matr, axis=0, keepdims=True)
        self.confusion_matrix = posterior_matr

        # Get samples from posterior distribution over temperature
        self.calibrator = BayesianTemperingCalibrator(self.prior_params, self.n_cls)
        logits = np.log(np.clip(model_probs, 1e-50, 1))
        logits = torch.from_numpy(logits)
        self.calibrator.fit(logits, torch.from_numpy(y_true))


class MAPOracleCombiner(OracleCombiner):
    """ P+L combination method, fit using MAP estimates
    This is our preferred combination method.
    """
    def __init__(self, diag_acc=0.75, strength=1., mu_beta=0.5, sigma_beta=0.5, **kwargs):
        super().__init__()
        self.calibrator = None
        self.prior_params = {'mu_beta': mu_beta,
                             'sigma_beta': sigma_beta
        }
        #self.n_cls = None
        self.diag_acc = diag_acc
        self.strength = strength

    def fit(self, model_probs, y_h, y_true, model_logits=None):
        self.n_cls = model_probs.shape[1]

        # Get MAP estimate of confusion matrix
        alpha, beta = get_dirichlet_params(self.diag_acc, self.strength, self.n_cls)
        prior_matr = np.eye(self.n_cls) * alpha + (np.ones(self.n_cls) - np.eye(self.n_cls)) * beta
        posterior_matr = 1. * confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls))
        posterior_matr += prior_matr
        posterior_matr = posterior_matr.T
        posterior_matr = (posterior_matr - np.ones(self.n_cls)) / (np.sum(posterior_matr, axis=0, keepdims=True) - self.n_cls)
        self.confusion_matrix = posterior_matr

        self.calibrator = TSCalibratorMAP()
        logits = np.log(np.clip(model_probs, 1e-50, 1))
        self.calibrator.fit(logits, y_true)


class LRCombiner:
    """ Implements a logistic regression model for combining ("LR" in our paper)
    """
    def __init__(self):
        self.n_cls = None
        self.clf = None

    def fit(self, model_probs, y_h, y_true):
        self.n_cls = model_probs.shape[1]

        y_h_onehot = np.eye(self.n_cls)[y_h]
        X = np.hstack((model_probs, y_h_onehot))

        self.clf = LogisticRegression()
        self.clf.fit(X, y_true)

    def calibrate(self, model_probs):
        # TODO
        raise NotImplementedError

    def combine_proba(self, model_probs, y_h):
        y_h_onehot = np.eye(self.n_cls)[y_h]
        X = np.hstack((model_probs, y_h_onehot))

        pred_proba = self.clf.predict_proba(X)
        # Need to set classes manually when data is limited
        probs = np.zeros((pred_proba.shape[0], self.n_cls))
        probs[:, self.clf.classes_] = pred_proba

        return probs

    def combine(self, model_probs, y_h):
        y_h_onehot = np.eye(self.n_cls)[y_h]
        X = np.hstack((model_probs, y_h_onehot))

        return self.clf.predict(X)


class DoubleConfusionCombinerMAP:
    """ Implements the double-confusion matrix combiner ("L+L") using MAP inference
    """
    # Combines via the model and human's confusion matrices (i.e. no calibration)
    def __init__(self, calibration_method='temperature scaling', diag_acc=0.75, strength=1.):
        self.confusion_matrix_h = None  # entry [i, j] is P(h = i | Y = j)
        self.confusion_matrix_m = None  # entry [i, j] is P(Y = j | m = i)

        self.n_train_u = None  # Amount of unlabeled training data
        self.n_train_l = None  # Amount of labeled training data
        self.n_cls = None  # Number of classes

        self.diag_acc = diag_acc
        self.strength = strength

        self.eps = 1e-50

    def fit(self, model_probs, y_h, y_true):
        self.n_cls = model_probs.shape[1]
        self.n_train_l = y_true.size

        alpha, beta = get_dirichlet_params(self.diag_acc, self.strength, self.n_cls)
        prior_matr = np.eye(self.n_cls) * alpha + (np.ones(self.n_cls) - np.eye(self.n_cls)) * beta

        conf_h = 1. * confusion_matrix(y_true, y_h, labels=np.arange(self.n_cls))
        conf_h += prior_matr
        conf_h = conf_h.T
        normalizer = np.sum(conf_h, axis=0, keepdims=True)
        conf_h = (conf_h - np.ones(self.n_cls)) / (normalizer - self.n_cls)
        self.confusion_matrix_h = conf_h

        # Estimate model confusion matrix
        y_m = np.argmax(model_probs, axis=1)
        conf_m = 1. * confusion_matrix(y_true, y_m, labels=np.arange(self.n_cls))
        conf_m += prior_matr
        conf_m = conf_m.T  # [i, j] = #(m = i and Y = j)
        normalizer = np.sum(conf_m, axis=1, keepdims=True)  # NB: normalize rows here, not columns!
        conf_m = (conf_m - np.ones(self.n_cls)) / (normalizer - self.n_cls)
        self.confusion_matrix_m = conf_m

    def combine_proba(self, model_probs, y_h):
        """ Combines model probabilities with hard labels via the calibrate-confuse equation given the confusion matrix.

        Args:
            p_m: Array of model probabilities ; shape (n_samples, n_classes)
            y_h: List of hard labels ; shape (n_samples,)

        Returns:
            Normalized posterior probabilities P(Y | m, h). Entry [i, j] is P(Y = j | h_i, m_i)
        """
        assert model_probs.shape[0] == y_h.size, 'Size mismatch between model probs and human labels'
        assert model_probs.shape[1] == self.n_cls, 'Size mismatch between model probs and number of classes'

        n_samples = model_probs.shape[0]
        y_m = np.argmax(model_probs, axis=1)

        y_comb = np.empty((n_samples, self.n_cls))
        for i in range(n_samples):
            y_comb[i] = self.confusion_matrix_m[y_m[i]] * self.confusion_matrix_h[y_h[i]]

        # Don't forget to normalize :)
        y_comb /= np.sum(y_comb, axis=1, keepdims=True)
        return y_comb

    def combine(self, model_probs, y_h):
        """ Combines model probs and y_h to return hard labels
        """
        y_comb_soft = self.combine_proba(model_probs, y_h)
        return np.argmax(y_comb_soft, axis=1)

    def calibrate(self, model_probs):
        preds = np.argmax(model_probs, axis=1)
        probs = self.confusion_matrix_m[preds, :]
        return probs


class SingleHumanParamCombinerMAP(OracleCombiner):
    """ Implements a simble ablation where the human's confusion matrix is only estimated with a single parameter
    """
    def __init__(self, calibration_method='temperature scaling', diag_acc=0.75, strength=1.):
        super().__init__(calibration_method=calibration_method)
        self.confusion_matrix_h = None  # entry [i, j] is P(h = i | Y = j)
        self.confusion_matrix_m = None  # entry [i, j] is P(Y = j | m = i)

        self.n_train_u = None  # Amount of unlabeled training data
        self.n_train_l = None  # Amount of labeled training data
        self.n_cls = None  # Number of classes

        self.diag_acc = diag_acc
        self.strength = strength

        self.eps = 1e-50

    def fit(self, model_probs, y_h, y_true):
        self.n_cls = model_probs.shape[1]

        b = 1.1
        a = self.diag_acc / (1 - self.diag_acc) * b

        a *= self.strength
        b *= self.strength

        n_human_correct = np.sum(y_h == y_true)
        diag_val = (n_human_correct + a - 1) / (y_h.size + a + b - 2)
        conf_h = np.eye(self.n_cls) * diag_val + (np.ones(self.n_cls) - np.eye(self.n_cls)) * (1. - diag_val ) / (self.n_cls-1)
        self.confusion_matrix = conf_h

        # Calibrate model probabilities
        self.fit_calibrator(model_probs, y_true)

