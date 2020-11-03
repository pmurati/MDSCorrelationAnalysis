# Multidimensional Scaling for the 3d visualization of the dynamical propagation of correlations in stock's closing prices

The approach taken will make use of two python scripts `corrMDS.py` and `stock_data_collection_tools.py`. The first program contains the class and its methods for transforming a dataframe of time series objects into a 3d trajectory. The second one is a collection of functions necessary for retrieving and processing closing prices for the german DAX index and has been adopted almost one to one from Kinsleys's Youtube Playlist on [Python Programming for Finance](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcOdF96TBtRtuQksErCEBYZ).

In the following, I will discuss the motivation and the problem of transforming the correlation structure of time series, specifically stock prices, into a configuration of points in an abstract 3d cartesian space for a holistic visualization approach of the statistical relationship between them. I will make a few remarks on the package dependencies and I will link their respective documentation pages afterwards. A user guide is added, explaining the procedure by applying multidimensional scaling to the correlation structure of daily stock closing prices for the german DAX index for the past 10 years. I will conclude with an API reference, giving an overiew of the class and its methods.

## Table of Contents

- [Introduction](#introduction)
- [Package Dependencies](#package-dependencies)
- [User Guide](#user-guide)
- [API Reference](#api-reference)

## Introduction

### Correlation matrices

Although quite easy to compute, the inference of information from correlation matrices can get messy, once we want to get a birds eye view on the correlation structure as a whole and are not only interested in pairwise dependencies. While it might be possible to get an overall feeling for the correlations of a few time series, it gets increasingly more difficult for larger collections as the number of pairwise correlations for $`n`$ time series grows as $`\Omicron(n^2)`$. For that matter, considering the case of time dependent correlations, the problem gets even worse. A typicall correlation matrix is shown in the following graphic for some stocks of the german DAX index.

![CorrMat_DAX](images/corrmat_DAX.png "Correlation matrix for the german DAX index, based on daily closing prices from 2010-2020")


### Distance metric

The following is an experimental attempt to tackle this issue by trying to find a sensible mapping from a given correlation matrix into an abstract 3d cartesian space, that provides additional information. Therefore, correlation has to be set in a relationship with an appropriate distance metric. Before choosing the metric, we have to ask ourselves in what way a specific pairwise correlation $`\rho_{ij}`$ should correspond to a certain distance $`d_{\rho}(i,j)`$ between two point coordinates $`\bold{x}_{i}`$, $`\bold{x}_{j}`$.

At first, it seems quite intuitive to make the assumption that time series with high positive correlation should correspond to nearby points and in the case of uncorrelated time series, the points should be far apart. The complication arises, when we deal with negative correlations. One solution is to conider only the magnitude $`\lvert\rho\rvert`$, thereby mapping positive and negative correlations with the same value onto the same coordinate. We would like to avoid this information loss in the resulting respresentation and assume the following implied manifold for the abstract space instead.   

Consider a 2d sphere embedded in 3d space as in the figure below with vectors $`\bold{x}_{i}`$, $`\bold{x}_{j}`$ and $`\bold{x}_{k}`$. Let us analyze the pairwise correlations with respect to $`\bold{x}_{i}`$.    

![CorrDist_Manifold](images/corrdist_manifold.jpg "")
 
In this representation, $`\bold{x}_{i}`$ sits at the pole of the sphere and points with high positive correlation will lie in its vicinity on the upper hemisphere. Here, spheric sections correspond to circles of constant correlation, with the great circle corresponding points of zero correlation, i.e. $`\bold{x}_{j}`$. Now, the antipodal point of $`\bold{x}_{i}`$ given by $`\bold{x}_{k}`$ represents a perfect negative correlation and points within its vicinity on the lower represent overall negative correlation. A distance metric that enables this representation is given by

```math
d_{\rho}(i,j) = \sqrt{2(1-\rho_{ij})}
```

The above choice, although sensible will not guarantee a perfect mapping onto a sphere. The resulting manifold will resemble more a general ellipsoid and the coordinates will deviate from its surface. The implementation of a gradient descent in the following section is easier to compute with the constraint of a spherical surface removed. Moreover, the assumption of the above manifold is not backed up by the data and should only serve as a guiding blueprint in the following steps.

### Gradient descent

Consider a random configuration of $`n`$ cartesian coordinates $`\bold_{x}_{1},...,\bold_{x}_{n}\in\R^3`$ arranged in the matrix 

```math
\bold_{X} = [\bold_{x}_{1},...,\bold_{x}_{n}]
```

## Package Dependencies

## User Guide

## API Reference