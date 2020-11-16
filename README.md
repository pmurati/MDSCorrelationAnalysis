# Multidimensional Scaling for the 3d visualization of the dynamic propagation of correlations in stock's closing prices

The approach taken will make use of two python scripts `stock_data_collection_tools.py` and `corrMDS.py`. The first program is a collection of functions necessary for retrieving and processing closing prices for the German DAX index and has been adopted almost one to one from Kinsleys's YouTube Playlist on [Python Programming for Finance](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcOdF96TBtRtuQksErCEBYZ). The second one contains the class and its methods for transforming a data frame of time series objects into a 3d trajectory, as shown in the animation below.

![Animation of 3d trajectory](images/animation.gif)

In the following, I will discuss the motivation and the problem of transforming the correlation structure of time series, specifically stock prices, into a configuration of points in an abstract 3d cartesian space for a holistic visualization approach of the statistical relationship between them. I will make a few remarks on the package dependencies and I will link their respective documentation pages afterwards. A user guide is added, explaining the procedure by applying multidimensional scaling to the correlation structure of daily stock closing prices for the German DAX index for the past 10 years. I will conclude with an API reference, giving an overview of the class and its methods.

## Table of Contents

- [Introduction](#introduction)
- [Package Dependencies](#package-dependencies)
- [User Guide](#user-guide)
- [API Reference](#api-reference)

## Introduction

### Correlation matrices

Although quite easy to compute, the inference of information from correlation matrices can get messy, once we want to get a birds eye view on the correlation structure as a whole and are not only interested in pairwise dependencies. While it might be possible to get an overall feeling for the correlations of a few time series, it gets increasingly more difficult for larger collections as the number of pairwise correlations for <img src="https://latex.codecogs.com/svg.latex?\inline&space;n" title="n" /> time series grows as <img src="https://latex.codecogs.com/svg.latex?\inline&space;O(n^2)" title="O(n^2)" />. For that matter, considering the case of time dependent correlations, the problem gets even worse. A typical correlation matrix is shown in the following graphic for some stocks of the German DAX index.

![CorrMat_DAX](images/corrmat_DAX.png "Correlation matrix for the german DAX index, based on daily closing prices from 2010-2020")


### Distance metric

The following is an experimental attempt to tackle this issue by trying to find a sensible mapping from a given correlation matrix into an abstract 3d cartesian space, that provides additional information. Therefore, correlation has to be set in a relationship with an appropriate distance metric. Before choosing the metric, we have to ask ourselves in what way a specific pairwise correlation <img src="https://latex.codecogs.com/svg.latex?\rho_{ij}" title="\rho_{ij}" /> should correspond to a certain distance <img src="https://latex.codecogs.com/svg.latex?d_{\rho}(i,j)" title="d_{\rho}(i,j)" /> between two point coordinates <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{i}" title="\bold{x}_{i}" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{j}" title="\bold{x}_{j}" /></a>.

At first, it seems quite intuitive to make the assumption that time series with high positive correlation should correspond to nearby points and in the case of uncorrelated time series, the points should be far apart. The complication arises, when we deal with negative correlations. One solution is to consider only the magnitude <a href="https://www.codecogs.com/eqnedit.php?latex=\lvert\rho\rvert" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\lvert\rho\rvert" title="\lvert\rho\rvert" /></a>, thereby mapping positive and negative correlations with the same value onto the same coordinate. We would like to avoid this information loss in the resulting representation and assume the following implied manifold for the abstract space instead.   

Consider a 2d sphere embedded in 3d space as in the figure below with vectors <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{i}" title="\bold{x}_{i}" /></a>, <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{j}" title="\bold{x}_{j}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{k}" title="\bold{x}_{k}" /></a>. Let us analyze the pairwise correlations with respect to <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{i}" title="\bold{x}_{i}" /></a>.    

![CorrDist_Manifold](images/corrdist_manifold.jpg "Visualization of a possible spherical manifold")

In this representation, <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{i}" title="\bold{x}_{i}" /></a> sits at the pole of the sphere and points with high positive correlation will lie in its vicinity on the upper hemisphere. Here, spherical sections correspond to circles of constant correlation, with the great circle corresponding points of zero correlation, i.e. <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{j}" title="\bold{x}_{j}" /></a>. Now, the antipodal point of <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{i}" title="\bold{x}_{i}" /></a> given by <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{k}" title="\bold{x}_{k}" /></a> represents a perfect negative correlation and points within its vicinity on the lower represent overall negative correlation. A distance metric that enables this representation is given by

<a href="https://www.codecogs.com/eqnedit.php?latex=d_{\rho}(i,j)&space;=&space;\sqrt{2(1-\rho_{ij})}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?d_{\rho}(i,j)&space;=&space;\sqrt{2(1-\rho_{ij})}" title="d_{\rho}(i,j) = \sqrt{2(1-\rho_{ij})}" /></a>

The above choice, although sensible will not guarantee a perfect mapping onto a sphere. The resulting manifold will resemble more a general ellipsoid and the coordinates will deviate from its surface. The implementation of a gradient descent in the following section is easier to compute with the constraint of a spherical surface removed. Moreover, the assumption of the above manifold is not backed up by the data and should only serve as a guiding blueprint in the following steps.

### Gradient descent

In the following, we will define the loss function <a href="https://www.codecogs.com/eqnedit.php?latex=J" target="_blank"><img src="https://latex.codecogs.com/svg.latex?J" title="J" /></a> which is going to be minimized, given:
- an initial correlation matrix <a href="https://www.codecogs.com/eqnedit.php?latex=d_{\rho}(i,j)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?d_{\rho}(i,j)" title="d_{\rho}(i,j)" /></a> of <a href="https://www.codecogs.com/eqnedit.php?latex=n" target="_blank"><img src="https://latex.codecogs.com/svg.latex?n" title="n" /></a> time series
- a random configurations of cartesian coordinates <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{1},\cdots,\bold{x}_{n}\in\mathbb{R}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{1},\cdots,\bold{x}_{n}\in\mathbb{R}" title="\bold{x}_{1},\cdots,\bold{x}_{n}\in\mathbb{R}" /></a>, arranged in the matrix

<a href="https://www.codecogs.com/eqnedit.php?latex=\bold{X}&space;=&space;[\bold{x}_{1},\dots,\bold{x}_{n}]^{T}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{X}&space;=&space;[\bold{x}_{1},\dots,\bold{x}_{n}]^{T}" title="\bold{X} = [\bold{x}_{1},\dots,\bold{x}_{n}]^{T}" /></a>

- a number of epochs <a href="https://www.codecogs.com/eqnedit.php?latex=k\in1,\dots,K" target="_blank"><img src="https://latex.codecogs.com/svg.latex?k\in1,\dots,K" title="k\in1,\dots,K" /></a>

- the learning rate <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\lambda" title="\lambda" /></a>

Let the Euclidean distance between two vectors <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{i}" title="\bold{x}_{i}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{j}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{j}" title="\bold{x}_{j}" /></a> be given by

<a href="https://www.codecogs.com/eqnedit.php?latex=d_{\bold{x}}(i,j)&space;=&space;\sqrt{(x_{i}-x_{j})^2&space;&plus;&space;(y_{i}-y_{j})^2&space;&plus;&space;(z_{i}-z_{j})^2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?d_{\bold{x}}(i,j)&space;=&space;\sqrt{(x_{i}-x_{j})^2&space;&plus;&space;(y_{i}-y_{j})^2&space;&plus;&space;(z_{i}-z_{j})^2}" title="d_{\bold{x}}(i,j) = \sqrt{(x_{i}-x_{j})^2 + (y_{i}-y_{j})^2 + (z_{i}-z_{j})^2}" /></a>

with <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{i}&space;=&space;[x_{i},y_{i},z_{i}]^{T}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{i}&space;=&space;[x_{i},y_{i},z_{i}]^{T}" title="\bold{x}_{i} = [x_{i},y_{i},z_{i}]^{T}" /></a> and where the bold index <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}" title="\bold{x}" /></a> distinguishes this distance form the distance metric above. We want to minimize the difference in distance between our derived distance metric and the Euclidean distance of configuration vectors <a href="https://www.codecogs.com/eqnedit.php?latex=\varepsilon_{ij}&space;=&space;d_{\bold{x}}(i,j)&space;-&space;d_{\rho}(i,j)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\varepsilon_{ij}&space;=&space;d_{\bold{x}}(i,j)&space;-&space;d_{\rho}(i,j)" title="\varepsilon_{ij} = d_{\bold{x}}(i,j) - d_{\rho}(i,j)" /></a>. This allows for the definition of a convex loss function

<a href="https://www.codecogs.com/eqnedit.php?latex=J&space;=&space;\sum_{i,j}&space;\varepsilon_{ij}^2&space;=&space;\sum_{i,j}&space;(d_{\bold{x}}(i,j)&space;-&space;d_{\rho}(i,j))^2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?J&space;=&space;\sum_{i,j}&space;\varepsilon_{ij}^2&space;=&space;\sum_{i,j}&space;(d_{\bold{x}}(i,j)&space;-&space;d_{\rho}(i,j))^2" title="J = \sum_{i,j} \varepsilon_{ij}^2 = \sum_{i,j} (d_{\bold{x}}(i,j) - d_{\rho}(i,j))^2" /></a>

The gradient of this loss with respect to each vector <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{i}" title="\bold{x}_{i}" /></a> is given by

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla_{i}J&space;=&space;\sum_{i,j}&space;\nabla_{i}&space;(\varepsilon_{ij})^2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nabla_{i}J&space;=&space;\sum_{i,j}&space;\nabla_{i}&space;(\varepsilon_{ij})^2" title="\nabla_{i}J = \sum_{i,j} \nabla_{i} (\varepsilon_{ij})^2" /></a>

By the chain rule, each term in the sum can be written as

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla_{i}&space;(\varepsilon_{ij})^2&space;=&space;2&space;\varepsilon_{ij}\cdot\nabla_{i}\varepsilon_{ij}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nabla_{i}&space;(\varepsilon_{ij})^2&space;=&space;2&space;\varepsilon_{ij}\cdot\nabla_{i}\varepsilon_{ij}" title="\nabla_{i} (\varepsilon_{ij})^2 = 2 \varepsilon_{ij}\cdot\nabla_{i}\varepsilon_{ij}" /></a>

Expanding <a href="https://www.codecogs.com/eqnedit.php?latex=\varepsilon_{ij}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\varepsilon_{ij}" title="\varepsilon_{ij}" /></a> in terms of the distances and noting that <a href="https://www.codecogs.com/eqnedit.php?latex=d_{\rho}(i,j)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?d_{\rho}(i,j)" title="d_{\rho}(i,j)" /></a> is fixed, we get

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla_{i}\varepsilon_{ij}&space;=&space;\nabla_{i}&space;d_{\bold{x}}(i,j)&space;=&space;\frac{\bold{x}_{i}-\bold{x}_{j}}{d_{\bold{x}}(i,j)}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nabla_{i}\varepsilon_{ij}&space;=&space;\nabla_{i}&space;d_{\bold{x}}(i,j)&space;=&space;\frac{\bold{x}_{i}-\bold{x}_{j}}{d_{\bold{x}}(i,j)}" title="\nabla_{i}\varepsilon_{ij} = \nabla_{i} d_{\bold{x}}(i,j) = \frac{\bold{x}_{i}-\bold{x}_{j}}{d_{\bold{x}}(i,j)}" /></a>

leading to an overall expression for the loss gradient

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla_{i}J&space;=&space;2\sum_{i,j}\frac{\varepsilon_{ij}}{d_{\bold{x}}(i,j)}\cdot(\bold{x}_{i}-\bold{x}_{j})" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nabla_{i}J&space;=&space;2\sum_{i,j}\frac{\varepsilon_{ij}}{d_{\bold{x}}(i,j)}\cdot(\bold{x}_{i}-\bold{x}_{j})" title="\nabla_{i}J = 2\sum_{i,j}\frac{\varepsilon_{ij}}{d_{\bold{x}}(i,j)}\cdot(\bold{x}_{i}-\bold{x}_{j})" /></a>

For compactness let us define

<a href="https://www.codecogs.com/eqnedit.php?latex=\nabla\bold{J}&space;=&space;\begin{bmatrix}&space;(\nabla_{1}J)^{T}&space;\\&space;\vdots&space;\\(\nabla_{n}J)^{T}&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\nabla\bold{J}&space;=&space;\begin{bmatrix}&space;(\nabla_{1}J)^{T}&space;\\&space;\vdots&space;\\(\nabla_{n}J)^{T}&space;\end{bmatrix}" title="\nabla\bold{J} = \begin{bmatrix} (\nabla_{1}J)^{T} \\ \vdots \\(\nabla_{n}J)^{T} \end{bmatrix}" /></a>

Updating the coordinate matrix <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{X}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{X}" title="\bold{X}" /></a> after one epoch <a href="https://www.codecogs.com/eqnedit.php?latex=k\to&space;k&plus;1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?k\to&space;k&plus;1" title="k\to k+1" /></a> of gradient descent is thus given by

<a href="https://www.codecogs.com/eqnedit.php?latex=\bold{X}^{k&plus;1}&space;=&space;\bold{X}^{k}&space;-&space;\lambda\cdot\nabla\bold{J}^{k}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{X}^{k&plus;1}&space;=&space;\bold{X}^{k}&space;-&space;\lambda\cdot\nabla\bold{J}^{k}" title="\bold{X}^{k+1} = \bold{X}^{k} - \lambda\cdot\nabla\bold{J}^{k}" /></a>

where we added the upper index for the respective epoch. Thus after each step, the coordinates get updated and in return the distances <a href="https://www.codecogs.com/eqnedit.php?latex=d_{\bold{x}}(i,j)" target="_blank"><img src="https://latex.codecogs.com/svg.latex?d_{\bold{x}}(i,j)" title="d_{\bold{x}}(i,j)" /></a>.  

### Dynamic propagation

We are interested in correlation changes over time. This can be achieved by computing the correlation matrix within a moving window, by setting a certain window and step size. By default we will consider step sizes of one week. The choice of the window size is a delicate one, as it will have a direct impact on how smooth correlation will vary over time. In the current setting, windows of 6 months are considered. 

Let the time dependence be given by the index <a href="https://www.codecogs.com/eqnedit.php?latex=t\in&space;1,\dots,T" target="_blank"><img src="https://latex.codecogs.com/svg.latex?t\in&space;1,\dots,T" title="t\in 1,\dots,T" /></a> which should not be confused with the epoch index <a href="https://www.codecogs.com/eqnedit.php?latex=k\in&space;1,\dots,K" target="_blank"><img src="https://latex.codecogs.com/svg.latex?k\in&space;1,\dots,K" title="k\in 1,\dots,K" /></a>. Keep in mind that one whole gradient descent is performed at each time <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/svg.latex?t" title="t" /></a>. The procedure in the precious section has the downside that the initial configuration of coordinate points is initialized randomly and that the final vectors, given only the distances from the correlation matrix as an input, are unique only up to rotation. To mitigate this problem, lets assume the following procedure.   

At time <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/svg.latex?t" title="t" /></a> given the coordinate matrix <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{X}_{t}^{1}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{X}_{t}^{1}" title="\bold{X}_{t}^{1}" /></a> in the first epoch, perform the gradient descent until epoch <a href="https://www.codecogs.com/eqnedit.php?latex=K" target="_blank"><img src="https://latex.codecogs.com/svg.latex?K" title="K" /></a> (or until a reasonable stopping criterion is triggered) 

<a href="https://www.codecogs.com/eqnedit.php?latex=\bold{X}_{t}^{1}&space;\xrightarrow[descent]{gradient}&space;\bold{X}_{t}^{K}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{X}_{t}^{1}&space;\xrightarrow[descent]{gradient}&space;\bold{X}_{t}^{K}" title="\bold{X}_{t}^{1} \xrightarrow[descent]{gradient} \bold{X}_{t}^{K}" /></a>

Then, in going from <a href="https://www.codecogs.com/eqnedit.php?latex=t" target="_blank"><img src="https://latex.codecogs.com/svg.latex?t" title="t" /></a> to <a href="https://www.codecogs.com/eqnedit.php?latex=t&plus;1" target="_blank"><img src="https://latex.codecogs.com/svg.latex?t&plus;1" title="t+1" /></a>, set 

<a href="https://www.codecogs.com/eqnedit.php?latex=\bold{X}_{t&plus;1}^{1}&space;=&space;\bold{X}_{t}^{K}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{X}_{t&plus;1}^{1}&space;=&space;\bold{X}_{t}^{K}" title="\bold{X}_{t+1}^{1} = \bold{X}_{t}^{K}" /></a>

thus allowing the previous coordinates to be the input for the following optimization. Again, the quality of the resulting trajectories will depend on the choice in window size. A wide window will lead to less variability in the change of correlation and smoother trajectories than a short one.   

## Package Dependencies

The correlation analysis described in the previous section as well as the process for retrieving the time series data are implemented using the following libraries.

- [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) used for web scraping the stock ticker symbols
- [datetime](https://docs.python.org/3/library/datetime.html) basic date and time types
- [matplotlib](https://matplotlib.org/) create static and interactive visualizations
- [mpl_toolkits](https://matplotlib.org/mpl_toolkits/index.html) extensions for matplotlib, needed for updating interactive trajectory plot 
- [numpy](https://numpy.org/doc/stable/) the go-to library for mathematical calculations, used for computing the gradient 
- [pandas](https://pandas.pydata.org/pandas-docs/stable/) the go-to library for data analysis, used for the import of data tables and their manipulation
- [pandas_datareader](https://pandas-datareader.readthedocs.io/en/latest/) remote data access for OHLC stock prices from [Yahoo!Finance](https://finance.yahoo.com)
- [pickle](https://docs.python.org/3/library/pickle.html) object serialization, used for ticker symbols
- [requests](https://requests.readthedocs.io/en/master/) sending http requests 
- [tqdm](https://tqdm.github.io/) visualize progress bar
- [os](https://docs.python.org/3/library/os.html) check for existing directories when retrieving stock data

## User Guide

The usage of `stock_data_collection_tools.py` and `corrMDS.py` are displayed by considering the case of retrieving the adjusted closing prices for the German DAX index of the past 10 years. They are visualized using MDS on the correlation metric for different times, thereby obtaining interactive 3d configuration plots. This section follows the implementation, as showcased in the [testscript](Testscript.ipynb).  

### Get the data: stock_data_collection_tools 

First, import the necessary scripts and packages.

```python
from corrMDS import corrMDS
import matplotlib.pyplot as plt
import pandas as pd
import stock_data_collection_tools as stct

plt.style.use('seaborn-darkgrid') #set the plot style
```

We start by retrieving the relevant ticker symbols for the DAX index, which are then saved as a pickle file. Subsequently, this file is used to get the OHLC prices from yahaoo finance and save a csv file for each stock in the generated sub folder `dax_stock_dfs`.

```python
sdct.save_dax_tickers()
sdct.get_dax_from_yahoo()
```

The data aggregation is finished by creating a joint table of the adjusted closing prices for each stock and saving it in your root directory as *dax_joined_closes.csv*.

```python
sdct.compile_dax()
```

>>>
**NOTE**  
The functions in `stock_data_collection_tools` can be easily adjusted to retrieve other indices. For a more detailed explanation, see Kinsleys's YouTube Playlist on [Python Programming for Finance](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcOdF96TBtRtuQksErCEBYZ).
>>>

The following function allows for the visualization of the correlation matrix, as showcased in the [introduction](#correlation-matrices). 

```python
sdct.visualize_data()
```

### Correlation based trajectories: the corrMDS class

Import the joint data frame of closing prices and create an instance of the corrMDS class, taking the data frame as input.

```python
df = pd.read_csv('dax_joined_closes.csv')
stock_obj = corrMDS(df)
```

The main method of this class is [MDS_trajectory](#mds_trajectorysize_windows6-seed42-metriclinear-nmax2500-lam001), which is the wrapper function for the [gradient descent](#gradient-descent) algorithm. Based on a moving time interval, the distance metric $`d_{\rho}(i,j) = \sqrt{2(1-\rho_{ij})}`$ (see, [here](#distance-metric)) and the respective correlation matrix, it computes the coordinates in the configuration space and saves the mean and standard deviation of the stock prices for each time step.

```python
stock_obj.MDS_trajectory()
```

Finally, we can visualize the results. The following method returns an interactive slider plot consisting of
- the mean distance of the configuration from its centroid, see [mean_distance](#mean_distance)
- the 3d time dependent configuration scatter plot, where each point is scaled by the standard deviation and colored with respect to the mean value of the underlying stock price for each time step

```python
stock_obj.plot_3d_state()
```

![Animation of 3d trajectory](images/animation.gif)

As we can see, the configuration seems to lie approximately on an ellipsoidal surface for most of the time. The size of the ellipsoid, or equivalently the mean distance from the centroid, shrink at certain points in time, indicating a tendency towards overall positive correlation as a result of stock market crashs. Of course, this representation of the data depends on numerous assumptions, one of which is the moving window size. Thus, by adjusting the window size (by default it is set to 6 months) to larger values, we would expect a smoother variation of the mean distance and smoother trajectories of the 3d configuration.


## API Reference

`stock_data_collection_tools.py`
- [save_dax_tickers](#save_dax_tickers)
- [get_dax_from_yahoo](#get_dax_from_yahoo)
- [compile_dax](#compile_dax)
- [visualize_data](#visualize_datadatadax_joined_closescsv)

`corrMDS.py`
- [cartesianMDS](#cartesianmds)
- [MDS_trajectory](#mds_trajectorysize_windows6-seed42-metriclinear-nmax2500-lam001)
- [mean_distance](#mean_distance)
- [plot_3d_state](#plot_3d_state)

This section will give an overview of the two scripts `stock_data_collection_tools.py` and `corrMDS.py`, their functions, classes and methods.

### stock_data_collection_tools.py

Includes the functions for retrieving and aggregating closing prices for the German DAX index and visualize their cross-correlations based on open source data.

>>>
**NOTE**  
The following functions can easily be adjusted to receive an input for any index or list of stocks.
>>>

#### save_dax_tickers():

Collect ticker symbols listed in the German DAX from the Wikipedia page.

**Returns:**     a list with the respective ticker symbols  
**Return type:** .pickle object

#### get_dax_from_yahoo()

Create a directory and save historical OHLC stock data for each ticker symbol from the *daxtickers.pickle* object for the last 10 years. A new directory *dax_stock_dfs* is created.

**Returns:**     the OHLC prices for each ticker  
**Return type:** .csv files

#### compile_dax()

For each ticker symbol in *daxtickers.pickle*, open the respective csv file with the OHLC prices and combine their adjusted closing prices into one single data frame. 

**Returns:**     the joined closing prices  
**Return type:** data frame

#### visualize_data(data='dax_joined_closes.csv')

Visualization of cross correlation matrix for the adjusted closing prices of the German DAX index as a heatmap plot. The color scale RdYlGn has been used to indicate positive (green), negative (red) and no correlation (yellow).

**Parameters:** **data** *(str,optional)* - the joined closing prices

### class corrMDS(df)

The class corrMDS in `corrMDS.py` contains the necessary methods to visualize the propagation of pairwise correlations between stock prices in an abstract 3d space. However, the class can be used for any data frame that consists of time series data.

**Parameters:** **df** *(str)* - input the relevant time series data, timestamps as column needed

The incoming data frame is split into a data frame of values and a series for the time stamps inside the instantiated class object.

>>>
**NOTE**  
The originial data under consideration, i.e. the stock data for the DAX index, contained a couple of stocks which were not acesible for the whole period. Thus, they resulted in correlations with NA values for certain periods, making an initial straightforward implementation of gradient descent impossible. This means that the number of points in the embedded 3d space will be conserved, i.e. fixed over time. Future updates might include the possibility of varying points, allowing the visualization of stocks that can enter and exit the DAX index at any given time.
>>>

#### cartesianMDS()

The core method, applying the gradient descent algorithm for one time step. The necessary input parameters are all passed directly to the instance in the wrapper function, see [MDS_trajectory](#mds_trajectorysize_windows6-seed42-metriclinear-nmax2500-lam001). 

**Returns:** arrays for the mean values and standard deviations per time series for given window, the change in loss function and the resulting loss  
**Return type:** array, float number

The arrays of mean values and standard deviations are used later on in the 3d visualization to display additional information. 

>>>
**NOTE**  
Currently, the gradient descent implemented in this method does not have a stopping criterion and runs for a specified number of epochs, passed on from the wrapper. Initially, the long run behavior of the loss function has been of interest. Future versions might drop the array of the loss and include a suitable stopping criterion.
>>>

#### MDS_trajectory(size_windows=6, seed=42, metric='linear', Nmax=2500, lam=0.01)

The wrapper function for [cartesianMDS](#cartesianmds), applying the gradient descent at each time step for a given window size and returning the cartesian coordinates as well as additional statistics such as the mean value and the std per moving window for each time series and the loss per epoch for each moving window.

**Parameters:**
**size_windows** *(int,optional)* - set the window size in months,default is set to 6  
**seed** *(int,optional)* - set the random seed for the initialization of coordinates, default is set to 42  
**metric** *(str,optional)* - the metric used, use the default option `linear`  
**Nmax** *(int,optional)* - the maximum number of epochs, default is set to 2500  
**lam** *(float,optional)* - the learning rate, default is set to 0.01

All parameters are saved directly as variables of the instance to be used in the cartesianMDS method. 

>>>
**NOTE**  
The option to set a different metric then `linear` is outdated. The option `spherical` was implemented to compute distances as the arc along a spherical surface. However, this restriction is not fulfilled by the resulting data, since the points are distributed roughly on an ellipsoid instead.
>>>

#### mean_distance()

For each time step, compute the mean distance of cartesian coordinates from their centroid (i.e. center of mass) as well as the mean standard deviation and save these arrays as instance variables.

The mean distance <a href="https://www.codecogs.com/eqnedit.php?latex=\overline{d}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\overline{d}" title="\overline{d}" /></a> of <a href="https://www.codecogs.com/eqnedit.php?latex=n" target="_blank"><img src="https://latex.codecogs.com/svg.latex?n" title="n" /></a> coordinates <a href="https://www.codecogs.com/eqnedit.php?latex=\bold{x}_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\bold{x}_{i}" title="\bold{x}_{i}" /></a> from their centroid <a href="https://www.codecogs.com/eqnedit.php?latex=\overline{\bold{x}}&space;=&space;\frac{1}{n}&space;\sum_{i}&space;\bold{x}_{i}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\overline{\bold{x}}&space;=&space;\frac{1}{n}&space;\sum_{i}&space;\bold{x}_{i}" title="\overline{\bold{x}} = \frac{1}{n} \sum_{i} \bold{x}_{i}" /></a> is defined as

<a href="https://www.codecogs.com/eqnedit.php?latex=\overline{d}&space;=&space;\frac{1}{n}&space;\sum_{i}&space;\lvert&space;\bold{x}_{i}&space;-&space;\overline{\bold{x}}&space;\rvert" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\overline{d}&space;=&space;\frac{1}{n}&space;\sum_{i}&space;\lvert&space;\bold{x}_{i}&space;-&space;\overline{\bold{x}}&space;\rvert" title="\overline{d} = \frac{1}{n} \sum_{i} \lvert \bold{x}_{i} - \overline{\bold{x}} \rvert" /></a>


#### plot_3d_state()

The final visualization, containing a mean distance plot and a 3d scatter plot with an interactive slider option for analyzing the trajectory. The scatter plot is colored in relation to the mean value per time series per time step and the sizes of the markers are set proportional to the standard deviation of the time series at each time step.

>>>
**NOTE**  
- Currently, the colors of the markers are not updated properly.
- Future versions might include annotation of individual points when hovering over them, see [this](https://stackoverflow.com/questions/10374930/matplotlib-annotating-a-3d-scatter-plot) post on Stack Overflow. However, in the current setting I could not find an efficient way to implement this feature.
>>>
