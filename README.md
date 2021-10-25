# Combining Human Predictions with Model Probabilities via Confusion Matrices and Calibration

This repo contains the code for our NeurIPS 2021 paper, Combining Human Predictions with Model Probabilities via Confusion Matrices and Calibration [[arxiv](https://arxiv.org/abs/2109.14591)].

# Project Structure

- Data is contained in the `/cifar10h` directory. Note: the ImageNet-16H dataset is currently not publicly available.
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
