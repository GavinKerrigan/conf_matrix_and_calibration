import warnings
import numpy as np
import torch
from torch import nn, optim
from torch.distributions.log_normal import LogNormal

import imax_calib.io as io
import imax_calib.utils as imax_utils
import imax_calib.calibration as imax_calibration

import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from torch.nn.functional import softmax

# This file implements various calibration methods.


class BaseCalibrator:
    """ Abstract calibrator class
    """
    def __init__(self):
        self.n_classes = None

    def fit(self, logits, y):
        raise NotImplementedError

    def calibrate(self, probs):
        raise NotImplementedError


class IdentityCalibrator(BaseCalibrator):
    """ A class that implements no recalibration.
    """

    def fit(self, probs, y):
        return

    def calibrate(self, probs):
        return probs


class TSCalibrator(BaseCalibrator):
    """ Maximum likelihood temperature scaling (Guo et al., 2017)
    """

    def __init__(self, temperature=1.):
        super().__init__()
        self.temperature = temperature

        self.loss_trace = None

    def fit(self, logits, y):
        """ Fits temperature scaling using hard labels.
        """
        # Pre-processing
        self.n_classes = logits.shape[1]
        _model_logits = torch.from_numpy(logits)
        _y = torch.from_numpy(y)
        _temperature = torch.tensor(self.temperature, requires_grad=True)

        # Optimization parameters
        nll = nn.CrossEntropyLoss()  # Supervised hard-label loss
        num_steps = 7500
        learning_rate = 0.05
        grad_tol = 1e-3  # Gradient tolerance for early stopping
        min_temp, max_temp = 1e-2, 1e4  # Upper / lower bounds on temperature

        optimizer = optim.Adam([_temperature], lr=learning_rate)

        loss_trace = []  # Track loss over iterations
        step = 0
        converged = False
        while not converged:

            optimizer.zero_grad()
            loss = nll(_model_logits / _temperature, _y)
            loss.backward()
            optimizer.step()
            loss_trace.append(loss.item())

            with torch.no_grad():
                _temperature.clamp_(min=min_temp, max=max_temp)

            step += 1
            if step > num_steps:
                warnings.warn('Maximum number of steps reached -- may not have converged (TS)')
            converged = (step > num_steps) or (np.abs(_temperature.grad) < grad_tol)

        self.loss_trace = loss_trace
        self.temperature = _temperature.item()

    def calibrate(self, probs):
        calibrated_probs = probs ** (1. / self.temperature)  # Temper
        calibrated_probs /= np.sum(calibrated_probs, axis=1, keepdims=True)  # Normalize
        return calibrated_probs


class EnsembleTSCalibrator(BaseCalibrator):
    """ Ensemble Temperature Scaling (Zhang et al., 2020)
    This is just a thin wrapper around ensemble_ts.py for convenience.
    """

    def __init__(self, temperature=1.):
        super().__init__()
        self.temperature = temperature
        self.weights = None

    def calibrate(self, probs):
        p1 = probs
        tempered_probs = probs ** (1. / self.temperature)  # Temper
        tempered_probs /= np.sum(tempered_probs, axis=1, keepdims=True)  # Normalize
        p0 = tempered_probs
        p2 = np.ones_like(p0) / self.n_classes

        calibrated_probs = self.weights[0] * p0 + self.weights[1] * p1 + self.weights[2] * p2

        return calibrated_probs

    def fit(self, logits, y):
        from ensemble_ts import ets_calibrate
        self.n_classes = logits.shape[1]

        # labels need to be one-hot for ETS
        _y = np.eye(self.n_classes)[y]

        t, w = ets_calibrate(logits, _y, self.n_classes, loss='mse')  # loss = 'ce'
        self.temperature = t
        self.weights = w


class TSCalibratorMAP(BaseCalibrator):
    """ MAP Temperature Scaling
    """

    def __init__(self, temperature=1., prior_mu=0.5, prior_sigma=0.5):
        super().__init__()
        self.temperature = temperature
        self.loss_trace = None

        self.prior_mu = torch.tensor(prior_mu)
        self.prior_sigma = torch.tensor(prior_sigma)

    def fit(self, model_logits, y):
        """ Fits temperature scaling using hard labels.
        """
        # Pre-processing
        _model_logits = torch.from_numpy(model_logits)
        _y = torch.from_numpy(y)
        _temperature = torch.tensor(self.temperature, requires_grad=True)

        prior = LogNormal(self.prior_mu, self.prior_sigma)
        # Optimization parameters
        nll = nn.CrossEntropyLoss()  # Supervised hard-label loss
        num_steps = 7500
        learning_rate = 0.05
        grad_tol = 1e-3  # Gradient tolerance for early stopping
        min_temp, max_temp = 1e-2, 1e4  # Upper / lower bounds on temperature

        optimizer = optim.Adam([_temperature], lr=learning_rate)

        loss_trace = []  # Track loss over iterations
        step = 0
        converged = False
        while not converged:

            optimizer.zero_grad()
            loss = nll(_model_logits / _temperature, _y)
            loss += -1 * prior.log_prob(_temperature)  # This step adds the prior
            loss.backward()
            optimizer.step()
            loss_trace.append(loss.item())

            with torch.no_grad():
                _temperature.clamp_(min=min_temp, max=max_temp)

            step += 1
            if step > num_steps:
                warnings.warn('Maximum number of steps reached -- may not have converged (TS)')
            converged = (step > num_steps) or (np.abs(_temperature.grad) < grad_tol)

        self.loss_trace = loss_trace
        self.temperature = _temperature.item()

    def calibrate(self, probs):
        calibrated_probs = probs ** (1. / self.temperature)  # Temper
        calibrated_probs /= np.sum(calibrated_probs, axis=1, keepdims=True)  # Normalize
        return calibrated_probs


class IMaxCalibrator(BaseCalibrator):
    """ I-Max Binning calibration (Patel et al., 2021)
    https://arxiv.org/pdf/2006.13092.pdf
    """

    def __init__(self, mode='CW', num_bins=15):
        super().__init__()
        # mode in ['cw', 'sCW', 'top1']
        self.cfg = io.AttrDict(dict(
            # All
            cal_setting=mode,  # CW, sCW or top1  # CW seems to be much better than sCW
            num_bins=num_bins,
            # Binning
            Q_method="imax",
            Q_binning_stage="raw",  # bin the raw logodds or the 'scaled' logodds
            Q_binning_repr_scheme="sample_based",
            Q_bin_repr_during_optim="pred_prob_based",
            Q_rnd_seed=928163,
            Q_init_mode="kmeans"
        ))
        self.calibrator = None

    def calibrate(self, probs):
        logits = np.log(np.clip(probs, 1e-50, 1))
        logodds = imax_utils.quick_logits_to_logodds(logits, probs=probs)
        cal_logits, cal_logodds, cal_probs, assigned = self.calibrator(logits, logodds)
        return cal_probs

    def fit(self, logits, y):
        n_samples, n_classes = logits.shape
        self.n_classes = n_classes
        self.cfg['n_classes'] = n_classes
        # y must be one-hot
        if y.ndim == 1:
            y_onehot = np.eye(self.n_classes)[y]
        else:
            y_onehot = y

        logodds = imax_utils.quick_logits_to_logodds(logits)
        self.calibrator = imax_calibration.learn_calibrator(self.cfg,
                                                            logits=logits,
                                                            logodds=logodds,
                                                            y=y_onehot)


class BayesianTemperingCalibrator:
    """ This class implements the fully Bayesian temperature scaling calibrator.
    """

    def __init__(self, prior_params, num_classes, **kwargs):
        self.num_classes = num_classes
        # Inference parameters
        self.NUTS_params = {'adapt_step_size': kwargs.pop('adapt_step_size', True),
                            'target_accept_prob': kwargs.pop('target_accept_prob', 0.8),
                            'max_plate_nesting': 1
                            }
        self.mcmc_params = {'num_samples': kwargs.pop('num_samples', 250),
                            'warmup_steps': kwargs.pop('num_warmup', 1000),
                            'num_chains': kwargs.pop('num_chains', 4)
                            }

        # Prior parameters on beta / delta ; assumes each delta is iid
        self.prior_params = {'mu_beta': prior_params['mu_beta'],
                             'sigma_beta': prior_params['sigma_beta']}

        # Posterior parameters after ADF
        self.posterior_params = {'mu_beta': None,
                                 'sigma_beta': None}

        # Drift parameters for sequential updating
        self.sigma_drift = kwargs.pop('sigma_drift', 0.0)

        # Tracking params
        # TODO: Prior/posterior trace
        self.timestep = 0
        self.mcmc = None  # Contains the most recent Pyro MCMC api object
        self.verbose = kwargs.pop('verbose', False)

        if self.verbose:
            print('\nInitializing BT model:\n'
                  '----| Prior: {} \n----| Inference Method: NUTS \n'
                  '----| MCMC parameters: {}'
                  ''.format(prior_params, self.mcmc_params))

    def fit(self, logits, labels):
        """ Performs an update given new observations.

        Args:
            logits: tensor ; shape (batch_size, num_classes)
            labels: tensor ; shape (batch_size, )
        """
        assert len(labels.shape) == 1, 'Got label tensor with shape {} -- labels must be dense'.format(labels.shape)
        assert len(logits.shape) == 2, 'Got logit tensor with shape {}'.format(logits.shape)
        assert (labels.shape[0] == logits.shape[0]), 'Shape mismatch between logits ({}) and labels ({})' \
            .format(logits.shape[0], labels.shape[0])

        logits = logits.detach().clone().requires_grad_()
        labels = labels.detach().clone()

        batch_size = labels.shape[0]
        if self.verbose:
            print('----| Updating HBC model\n--------| Got a batch size of: {}'.format(batch_size))

        self._update_prior_params()
        if self.verbose:
            print('--------| Updated priors: {}'.format(self.prior_params))
            print('--------| Running inference ')
        nuts_kernel = NUTS(bt_model, **self.NUTS_params)
        self.mcmc = MCMC(nuts_kernel, **self.mcmc_params, disable_progbar=not self.verbose,
                         mp_context='spawn')  # Progbar if verbose
        self.mcmc.run(self.prior_params, logits, labels)

        self._update_posterior_params()
        self.timestep += 1

        return self.mcmc

    def _update_prior_params(self):
        """ Updates the prior parameters using the ADF posterior from the previous timestep, plus the drift.

        If this is the first batch, i.e. timestep == 0, do nothing.
        """
        if self.timestep > 0:
            self.prior_params['mu_beta'] = self.posterior_params['mu_beta']
            self.prior_params['sigma_beta'] = self.posterior_params['sigma_beta'] + self.sigma_drift

    def _update_posterior_params(self):
        """ Fits a normal distribution to the current beta samples using moment matching.
        """
        beta_samples = self.get_current_posterior_samples()
        self.posterior_params['mu_beta'] = beta_samples.mean().item()
        self.posterior_params['sigma_beta'] = beta_samples.std().item()

    def get_current_posterior_samples(self):
        """ Returns the current posterior samples for beta.
        """
        if self.mcmc is None:
            return None

        posterior_samples = self.mcmc.get_samples()['beta']

        return posterior_samples

    def calibrate(self, probs):
        """ Calibrates the given batch of logits using the current posterior samples.

        Args:
            logit: tensor ; shape (batch_size, num_classes)
        """
        # Get beta samples
        beta_samples = self.get_current_posterior_samples()  # Shape (num_samples, num_classes)
        n_samples = beta_samples.size()[0]

        # Map betas to temperatures
        temperature_samples = torch.exp(beta_samples)  # Shape (num_samples, )

        # Shape (n_inputs, n_cls, n_mcmc_samples)
        tempered_probs = torch.empty((probs.shape[0], probs.shape[1], n_samples))
        for i, temperature in enumerate(temperature_samples):
            tempered_probs[:, :, i] = probs ** (1. / temperature)
            tempered_probs[:, :, i] /= torch.sum(tempered_probs[:, :, i], dim=1, keepdim=True)

        # Average over the sampled probabilities to get Monte Carlo estimate
        calibrated_probs = tempered_probs.mean(dim=-1)  # Shape (batch_size, num_classes)
        calibrated_probs = np.asarray(calibrated_probs)
        return calibrated_probs

    def get_MAP_temperature(self, logits, labels):
        """ Performs MAP estimation using the current prior and given data.
         NB: This should only be called after .update() if used in a sequential setting, as this method
         does not update the prior with sigma_drift.

         See: https://pyro.ai/examples/mle_map.html
         """
        pyro.clear_param_store()
        svi = pyro.infer.SVI(model=bt_model, guide=MAP_guide,
                             optim=pyro.optim.Adam({'lr': 0.001}), loss=pyro.infer.Trace_ELBO())

        loss = []
        num_steps = 5000
        for _ in range(num_steps):
            loss.append(svi.step(self.prior_params, logits, labels))

        eps = 2e-2
        loss_sddev = np.std(loss[-25:])
        if loss_sddev > eps:
            warnings.warn('MAP optimization may not have converged ; sddev {}'.format(loss_sddev))
            print('Here is the last few loss terms for inspection: \n', loss[-50:])

        MAP_temperature = torch.exp(pyro.param('beta_MAP')).item()
        return MAP_temperature


def bt_model(prior_params, logits, labels):
    """ Helper function for fully Bayesian temperature scaling.
    """
    n_obs = logits.shape[0]  # Batch size

    # Prior over global temperature Beta ~ N( beta_mu, beta_sigma^2 )
    prior_beta_mu = prior_params['mu_beta']
    prior_beta_sigma = prior_params['sigma_beta']
    beta = pyro.sample('beta', dist.Normal(prior_beta_mu, prior_beta_sigma))  # Shape (1, )

    probs = softmax(torch.exp(-1. * beta) * logits, dim=1)  # Shape (n_obs, n_classes) ; tempered probabilities

    # Observation plate ; vectorized
    with pyro.plate('obs', size=n_obs):
        a = pyro.sample('cat_obs', dist.Categorical(probs=probs), obs=labels)


def MAP_guide(prior_params, logits, labels):
    """ Helper function for fully Bayesian temperature scaling.
    """
    beta_MAP = pyro.param('beta_MAP', torch.tensor(1., requires_grad=True))
    pyro.sample('beta', dist.Delta(beta_MAP))
