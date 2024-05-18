# ATF - Aggregated Testing Framework

This repository contains the source code for a Python framework developed for the ease of testing machine learning models. It was a group project with members - 
Vibhu Garg, Himanshu Yadav and Divyansh Gupta. 

## Overview

ATF currently supports models trained in TensorFlow and aims to extend its capabilities to other model frameworks through community contributions. It accepts a 
model, input data, expected output, and a list of attacks to evaluate the model's robustness. This is crucial for developing models that can withstand adversarial 
manipulations in real-world applications.
Model, input data and output data files are expected to be in [HDF5](https://docs.h5py.org/en/stable/) format.

## Contributions

ATF encourages contributions from the community to:
- Extend support for additional machine learning frameworks.
- Add new attack methods.
- Improve existing attack implementations.

## License

ATF is distributed under the MIT License. See LICENSE for more information.

## Acknowledgements

Attack implementations:
- [cleverhans](https://github.com/cleverhans-lab/cleverhans) for FGSM and PGD
- [foolbox](https://github.com/bethgelab/foolbox) for Carlini & Wagner, DeepFool, and Salt and Pepper Noise Attack
