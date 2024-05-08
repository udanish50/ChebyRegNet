# ChebyRegNet: An Unsupervised Deep Learning Technique for Deformable Medical Image Registration

## Overview
This repository contains the implementation and supporting material for the paper "ChebyRegNet: An Unsupervised Deep Learning Technique for Deformable Medical Image Registration". Our approach introduces a novel deep learning framework that utilizes Chebyshev polynomials to enhance registration accuracy across diverse medical imaging modalities.

## Publication
- **Authors**: Muhammad Umair Danish, Mohammad Noorchenarboo, Apurva Narayan, Katarina Grolinger
- **Affiliations**: Department of Electrical and Computer Engineering and Department of Computer Science, Western University, London, Ontario, Canada
- **Contact**: [mdanish3@uwo.ca](mailto:mdanish3@uwo.ca), 

## Abstract
ChebyRegNet leverages Chebyshev polynomials within a hypernetwork structure to address the high error rates commonly seen in multi-modality medical image registrations. This innovative architecture not only enhances the precision of deformable image registration but also maintains robustness against variations in imaging conditions.

## Key Contributions
- Introduction of Chebyshev polynomials in a deep learning model to enhance medical image registration accuracy.
- Development of a Feature Extraction Network that processes deep features for the Chebyshev HyperNetwork.
- Empirical validation shows improved registration accuracy compared to traditional methods like VoxelMorph and HyperMorph, especially in challenging multi-modality scenarios.

## Repository Structure
- `src/`: Contains all source code used for implementing ChebyRegNet.
- `data/`: Sample data and links to the IXI dataset used for training and validation.
- `models/`: Pre-trained models and weights.
- `notebooks/`: Jupyter notebooks for demonstration of the methods and visualization of results.
- `results/`: Detailed results and comparative analysis with other state-of-the-art methods.

## Setup and Running
Instructions for setting up the environment, installing dependencies, and running the code are provided in the `setup.md` file.

## Usage
To replicate the results or to use ChebyRegNet on new datasets, follow the guidelines in the `usage.md` file located in the `notebooks/` directory.

## Citing
If you find ChebyRegNet useful in your research, please consider citing:
```bibtex
@article{danish2023chebyregnet,
  title={ChebyRegNet: An Unsupervised Deep Learning Technique for Deformable Medical Image Registration},
  author={Danish, Muhammad Umair and Noorchenarboo, Mohammad and Narayan, Apurva and Grolinger, Katarina},
  journal={Journal of Advanced Medical Imaging},
  year={2023}
}
