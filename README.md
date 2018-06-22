Vintage Car Auctions
=====================================

### News
June 2018
* Adding a documentation to the project

May 2018

April 2018

### Introduction

This project aims to create a novel approach of estimating a dynamic auction market for vintage cars. This allows user to enter simulation data
and stock data which will be used to correlate one's private wealth to his or her bidding potential.
* **Important**: The system assumes the presence of two files in the folder simulation_data. One being :
  * avgsp.csv
  * book_total_brand.csv


### Setting: Auctions for Vintage Cars
* These cars have direct use value but can also be resold
* Explicitly treat these cars as assets: Buyers anticipate selling in the future,
wealth evolves endogenously, and return on other assets potentially influences prices of vintage cars
* Similar in spirit to Bayer et al's (Econometrica 2016) model of demand for houses:
  * "We explicitly mohttps://github.com/Chaitanyadel housing as an asset and allow each household's wealth to evolve endogenously. Households in our model anticipate selling their homes at some point in the future and thus explicitly consider the expected evolution of neighborhood amenities and housing prices when deciding where and when to purchase (or sell) their house."
* The graph below shows that average prices have fluctuated strongly in recent years and are strongly correlated with the stock market
  * The (blue) squares plot the simple average of vintage cars sold in that year
  * The (green) dots plot the year coefficients of the year fixed effect from the following regression:
  * ![equation](http://latex.codecogs.com/gif.latex?%5Ctext%7Bln%7D%28p_%7Bi%2Ct%7D%29%20%3D%20%5Cgamma_y%20&plus;%20%5Ctheta_j%20&plus;%20%5Cepsilon_%7Bi%2Ct%7D), where *i* indexes cars, *j* indexes model-year-built, *y* indexes the year
  * The (red) crosses plot the average S&P500
  * The observed correlation may be driven (among other things) by the income effect of a higher stock market and by arbitrage between the two markets that equalises returns.


 <figure>
    <img src='documentation/graph.png' alt='missing' width="450"/>
    <figcaption>Correlation of average prices with the stock market</figcaption>
</figure>

### Table of Contents
* [1. Installation and Requirements](#1-installation-and-requirements)
  * [1.1. Required Libraries](#11-required-libraries)
  * [1.2. Installation](#12-installation)
  * [1.3. Required Data](#13-required-data-pre-processing)
* [2. Running the Software](#2-running-the-software)
  * [2.1. How to run a simulation](#21-training-a-tiny-cnn---making-sure-it-works)
  * [2.2. Running it on a GPU](#22-running-it-on-a-gpu)
* [3. How it works](#3-how-it-works)
  * [3.1 Algorithm](#31-specifying-model-architecture)
  * [3.2 Mathematics](#32-training)
* [4. How to run the simulation on your data](#4-how-to-run-deepmedic-on-your-data)
* [5. Concluding](#5-concluding)
* [6. Licenses](#6-licenses)

### 1. Installation and Requirements

#### 1.1. Required Libraries
***Note for Wolfgang:
pip freeze > requirements.txt
pip install -r requirements.txt***

The system is written in python. The following libraries are required:

- [scipy](https://www.scipy.org/):  Python-based ecosystem of open-source software for mathematics, science, and engineering.
- [pandas](http://pandas.pydata.org/): Library providing high-performance, easy-to-use data structures and data analysis tools.
- [matplotlib](https://matplotlib.org/users/installing.html): Python 2D plotting library.
- [numpy](http://www.numpy.org/) : General purpose array-processing package.
- [seaborn](https://seaborn.pydata.org/) : Python visualization library based on matplotlib
- [argparse](https://pypi.org/project/argparse/) : Makes it easy to write user friendly command line interfaces.
- [bayesian-optimization](https://github.com/fmfn/BayesianOptimization): Python implementation of bayesian global optimization with gaussian processes.

***Note for Wolfgang:
https://www.parallelpython.com/
https://pymotw.com/2/multiprocessing/basics.html
https://github.com/tirthajyoti/PythonMachineLearning/tree/master/Function%20Approximation%20by%20Neural%20Network***

#### 1.2. Installation
(The below are for unix systems, but similar steps should be sufficient for Windows.)

The software cloned with:
```
git clone https://github.com/ChaitanyaBaweja/Vintage_Car_Auctions.git
```
After cloning it, all dependencies can be installed as described below.

#### Install using virtual environment (preferred)

If you do not have sudo/root privileges on a system, we suggest you install using a virtual environment.
From a *bash* shell, create a virtual environment in a folder that you wish:
```cshell
virtualenv -p python3 FOLDER_FOR_ENVS/environment     # or python3
source FOLDER_FOR_ENVS/environment/bin/activate       # If using csh, source environment/bin/activate.csh
```
Then continue with the steps below.
***Use pip install -r requirements.txt***
