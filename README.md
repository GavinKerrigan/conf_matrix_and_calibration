# Combining Human Predictions with Model Probabilities via Confusion Matrices and Calibration

This repo contains the code for our NeurIPS 2021 paper, Combining Human Predictions with Model Probabilities via Confusion Matrices and Calibration [[arxiv](https://arxiv.org/abs/2109.14591)].

The ImageNet-16H dataset is available on the [OSF](https://osf.io/2ntrf/?view_only=9ec9cacb806d4a1ea4e2f8acaada8f6c). Please also see our work [Bayesian Modeling of Human-AI Complementarity [Steyvers et al., 2022]](https://www.pnas.org/doi/10.1073/pnas.2111547119) describing this dataset in more detail.

# Project Structure

- Data for the CIFAR-10h experiments is contained in the `/cifar10h` directory. 
- `/experiments` contains various scripts for reproducing the experiments in our paper.
- `calibrators.py` implements various calibration methods.
- `combination_methods.py` implements various combination methods.
- `data_utils.py` contains useful data processing methods.

# References
Please consider citing our paper as:
```
@inproceedings{kerrigan2021combining,
  title={Combining Human Predictions with Model Probabilities via Confusion Matrices and Calibration},
  author={Kerrigan, Gavin and Smyth, Padhraic and Steyvers, Mark},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```
