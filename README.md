# RamanNet
With the advent of hyperspectral Raman imaging technology, especially the rapid and high-resolution imaging schemes, datasets with thousands to millions of spectra are now commonplace. Standard preprocessing and regression methods such as least squares approaches are time consuming and require input from highly trained operators. Here we propose a solution to this analytic bottleneck through a convolutional neural network trained fully on synthetic data and then applied to experimental measurements, including cases where complete spectral information is missing (i.e. an underdetermined model). An advantage of the model is that it combines background correction and regression into a single step, and does not require user-selected parameters. We compare our results with traditional least squares methods, including the popular asymmetric least squares (AsLS) approach. Our results demonstrate that the proposed CNN model boasts less sensitivity to parameter selection, and with a rapid processing speed, with performance equal to or better than comparison methods. The performance is validated on synthetic spectral mixtures, as well as experimentally measured single-vesicle liposome data.
# Installation
Python 3.6 is recommended.

[python](https://www.python.org/)

## Install tensorflow

[tensorflow](https://www.tensorflow.org/)

## Install dependent packages
**1.Numpy**

pip install numpy

**2.Scipy**

pip install Scipy

**3.Matplotlib**

pip install Matplotlib

**4.Pandas**

pip install pandas

# Download the model and run directly
The spectra database and models exceeded the limit; therefore we have uploaded them on Google drive.

Download at: [Google drive](https://drive.google.com/drive/folders/16bgVecnjALsifiu14kMi9xshx5iXEhnu)

**1.Generating synthetic training mixture datasets**

A simulated binary or ternary spectral mixture datasets can be generated by running the files 'generate_training_dataset_binary.py' or 'generate_training_dataset_ternary.py' respectively. A corresponding example training dataset has been uploaded to the folder named 'spectra_data'.

**2.Training the model**

Run the files "Regression_Raman_train_binary and Regression_Raman_train_ternary" to train a CNN binary model and CNN ternary model respectively. A corresponding example CNN binary model and CNN ternary model have been uploaded to the folder named 'RamanNet'. Similarly, you can train a three-component (ternary) model and four-component (quaternary model) as is the case for the experimental mixtures.

**3.Predict spectral mixtures concentrations**

Run the file 'Concentration_Raman_test_simulated.py' to predict the concentrations of individual chemicals in the simulated mixtures.

Run the file 'Concentration_Raman_test.py' to predict the concentrations of the pure biochemicals in the experimental mixture.

# Contact
Derrick Boateng: derrick756@mail.ustc.edu.cn
