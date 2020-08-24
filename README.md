# ehr_ml

ehr_ml is a python package for building models on EHR data. 

There are four main features of this library
1. Process EHR data into an efficient per-patient data format to enable further processing
2. Apply labeling functions on those patients to obtain labels
3. Apply featurization schemes on those patients to obtain a feature matrix
4. Train and evaluate models using using patient splitting and cross validation

Note: This repository requires bazel, poetry and a C++ compiler in order to function. Please read the poetry documentation for how to install/use this package.

Further documentation will be coming soon.
