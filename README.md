---
stage: Create
group: Source Code
info: "To determine the technical writer assigned to the Stage/Group associated with this page, see https://about.gitlab.com/handbook/engineering/ux/technical-writing/#designated-technical-writers"
type: reference, howto
---

# Multidimensional Scaling for the 3d visualization of the dynamical propagation of correlations in stock's closing prices

The approach taken will make use of two python scripts `corrMDS.py` and `stock_data_collection_tools.py`. The first program contains the class and its methods for transforming a dataframe of time series objects into a 3d trajectory. The second one is a collection of functions necessary for retrieving and processing closing prices for the german DAX index and has been adopted almost one to one from Kinsleys's Youtube Playlist on [Python Programming for Finance](https://www.youtube.com/playlist?list=PLQVvvaa0QuDcOdF96TBtRtuQksErCEBYZ).

In the following, I will discuss the motivation and the problem of transforming the correlation structure of time series, specifically stock prices, into a configuration of points in an abstract 3d cartesian space for a holistic visualization approach of the statistical relationship between them. I will make a few remarks on the package dependencies and I will link their respective documentation pages afterwards. A user guide is added, explaining the procedure by applying multidimensional scaling to the correlation structure to daily stock closing prices of the german DAX index for the past 10 years. I will conclude with an API reference, giving an overiew of the class and its methods.        