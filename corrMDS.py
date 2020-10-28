# -*- coding: utf-8 -*-
"""
===============================================================================
Multidimensional Scaling (MDS) for the 3d visualization of the dynamical
propagation of correlations in stock's closing prices.
===============================================================================

DESCRIPTION
-----------

The class corrMDS contains the necessary functions to visualize the propagation
of pairwise correlations between stock prices in an abstract 3d space. However,
the class can be used for any dataframe that consists of time series data.

Multidimensional Scaling (MDS) is used to visualize pairwise distances among
a set of objects into a configuration of points in an abstract Cartesian space.
Therefore, a first mapping of correlations into a well defined distance metric
is used, in order to apply MDS. The key assumption is that objects with high
pairwise correlation (i.e close to 1) will be transformed into points in space
that are close together, points with high negative correlation (i.e. close to
-1) will be far apart and uncorrelated points (around 0) will be somewhere
inbetween. This can be achieved by the transformation/distance metric 

    d_ij = sqrt(2* (1 - rho_ij))
    
where rho_ij stands for the correlation between objects i,j and d_ij for the 
resulting distance.     

NOTE: there is ambiguity in the choice of the metric as well as the resulting 
configuration in 3d space. The later results form the fact that MDS is 
implemented in a stochastic way and is unique only up to rotational invariance.


With the distance metric defined as above, the optimal configuration can be
computed by means of a gradient descent algorithm, starting with a unform
random configuration of points in Cartesian 3d space. The cost function is
given by the squared sum of differences between d_ij and the distances of the
resulting cartesian coordiantes.
By using a moving window, one can analyze changes in the pairwise correlation
over time and thus, observe that the points in 3d trace out trajectories. 
Therefore, at each time step, the resulting cartesian coordinates from the
preceding gradient descent can be used as inputs for the following one,
ensuring a sensible path in the output space.       

NOTE: a termination condition is not implemented yet. For the moment, the
algorithm runs through a fixed loop.


The class corrMDS consists of the following methods:
    __init__:
        initialize the instance, input the dataframe of time series data
    cartesianMDS:
        the core method, applying the gradient descent based on the distance
        metric and an input matrix of coordiantes
    MDS_trajectory:
        the wrapper function for cartesianMDS, applying the gradient descent
        at each time step and returning the cartesian coordinates as well as
        additional statistics
    mean_distance:
        compute the mean distance of points from their centroid
    plot_3d_state:
        the final visualization, containing a mean distance plot and a
        3d scatter plot with an interactive slider option for analyszing the
        trajectory.
"""

import matplotlib as mlt
import matplotlib.cm as cmx
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider #interactive slider
from mpl_toolkits.mplot3d.art3d import juggle_axes #update 3d axes 
import numpy as np
import pandas as pd
from tqdm import tqdm #progress bar


class corrMDS:
    '''
    Class for computing and visualizing pairwise correlations of time series
    data in an abstract 3d Cartesian space.
    
    Attributes
    ----------
    df : dataframe
        Input the relevant time series data, timestamps as column needed
        
    Methods
    -------
    cartesianMDS():
        the core method, applying the gradient descent based on the distance
        metric and an input matrix of coordiantes
        
    MDS_trajectory(size_windows=6,
                   seed=42,
                   metric='linear',
                   Nmax=2500,
                   lam=0.01):
        the wrapper function for cartesianMDS, applying the gradient descent
        at each time step for a given window size and returning the cartesian
        coordinates as well as additional statistics
        
    mean_distance():
        compute the mean distance of points from their centroid
        
    plot_3d_state():
        the final visualization, containing a mean distance plot and a
        3d scatter plot with an interactive slider option for analyszing the
        trajectory    
        
    '''
    
    def __init__(self,df):
        
        '''
        Extract the date column from the input dataframe
        
        Parameters
        ----------
        df : dataframe
            Input the relevant time series data, timestamps as column needed
        '''
        
        self.df = df.copy()
        
        self.date_range=pd.to_datetime(df['Date'].values)
        self.df.drop(['Date'],axis=1,inplace=True)
        #NOTE: the following dropna command has been implemented since the
        #original data under consideration, i.e. the stock data for the DAX
        #index, contained a couple of stocks which were not accesible for the
        #whole time period. Thus, they resulted in correlations with NA values
        #for certain periods, making an initial straight forward implementation
        #of gradient descent impossible.This means that the number of points
        #in the embedded 3d space will be concerved, i.e. fixed over time.
        #Future updates might include the possiblity of varying points,
        #allowing the visualization of stocks that enter and exit the DAX index
        #at any given time.
        self.df.dropna(axis=1,how='any',inplace=True)
        
        
    def cartesianMDS(self):
        '''
        The core method, applying the gradient descent algorithm for one 
        time step. For details on the method, see the github documentation
        under docs/.
        
        The necessary input parameters are all passed directly to the instance
        in the wrapper function, see MDS_trajectory.
                        
        Returns
        -------
        df_mean : array
            array of mean values per time series for given window
            
        df_std : array
            array of standard deviations per time series for given window
            
        eps_array : array
            the change in loss function
            
        eps : float
            the resulting loss 
        
        '''
        
        #specification of metric will be removed in the future
        metric=self.metric
        Nmax=self.Nmax #number of epochs
        lam=self.lam #learning rate
        
        #define the moving window, i.e. which data to use, based on the
        #index (self.idx) and the window size (self.size_windows)
        df_m = self.df[self.date_range.month.isin(self.windows[1,self.idx:(self.idx+self.size_windows)]) &
                       self.date_range.year.isin(self.windows[0,self.idx:(self.idx+self.size_windows)])]

        df_mean = df_m.mean() #mean for the window
        df_std = df_m.std() #std for the window

        df_m_corr = df_m.corr() #correlation matrix for the window

        corr_dist = df_m_corr.copy(deep=True)

        #transform correlation into distance
        if metric=='linear':
            for i in corr_dist.columns:
                for j in corr_dist.index:

                    if i==j:
                        corr_dist.loc[j,i]=0 # zero distance (X=Y => d[X,Y]=0) 
                    else:
                        corr_dist.loc[j,i]=np.sqrt(2*(1-df_m_corr.loc[j,i]))

        #outdated
        if metric=='spheric':
            for i in corr_dist.columns:
                for j in corr_dist.index:

                    if i==j:
                        corr_dist.loc[j,i]=0
                    else:
                        corr_dist.loc[j,i]=np.sqrt(2*(1-np.cos(np.pi/2 *(1-df_m_corr.loc[j,i]))))

        #initialize array for loss as a function of epoch
        eps_array=np.zeros(Nmax)
        
        #run gradient descent for Nmax epochs
        #the initial (random) configuration of points in Cartesian coordinates
        #is passed to the instance via self.X
        for nn in (range(Nmax)):
            #compute distance of cartesian coordinates
            d_ij=np.sqrt(
                (np.outer(np.ones(len(self.X[:,0])),self.X[:,0])-np.outer(self.X[:,0],np.ones(len(self.X[:,0]))) )**2 +
                (np.outer(np.ones(len(self.X[:,1])),self.X[:,1])-np.outer(self.X[:,1],np.ones(len(self.X[:,1]))) )**2 +
                (np.outer(np.ones(len(self.X[:,2])),self.X[:,2])-np.outer(self.X[:,2],np.ones(len(self.X[:,2]))) )**2

            )

            #the difference in distances
            eps_ij = d_ij - corr_dist.values
            #the loss
            eps_array[nn]=np.sum(eps_ij**2)        
            
            #to avoid divergences in the computation of the gradient
            np.fill_diagonal(eps_ij,0)
            np.fill_diagonal(d_ij,1)

            #the partial derivative of coordinate distances by x1, x2, x3  
            delta_d_ij_x1= (np.outer(np.ones(len(self.X[:,0])),self.X[:,0])-np.outer(self.X[:,0],np.ones(len(self.X[:,0]))) )/d_ij
            delta_d_ij_x2= (np.outer(np.ones(len(self.X[:,1])),self.X[:,1])-np.outer(self.X[:,1],np.ones(len(self.X[:,1]))) )/d_ij
            delta_d_ij_x3= (np.outer(np.ones(len(self.X[:,2])),self.X[:,2])-np.outer(self.X[:,2],np.ones(len(self.X[:,2]))) )/d_ij

            #to avoid divergences in the computation of the gradient
            np.fill_diagonal(delta_d_ij_x1,0)
            np.fill_diagonal(delta_d_ij_x2,0)
            np.fill_diagonal(delta_d_ij_x3,0)

            #the complete partial derivative of the loss by x1, x2, x3
            #NOTE: the partial derivative of loss by coordinate distance
            #      equals eps_ij
            delta_eps_x1=np.sum(eps_ij*delta_d_ij_x1,axis=1)
            delta_eps_x2=np.sum(eps_ij*delta_d_ij_x2,axis=1)
            delta_eps_x3=np.sum(eps_ij*delta_d_ij_x3,axis=1)


            delta_eps=np.array([delta_eps_x1,delta_eps_x2,delta_eps_x3]).T
            
            #update coordinates
            self.X += +lam*delta_eps


        return(df_mean, df_std, eps_array, eps_array[-1]) 
    
    
    def MDS_trajectory(self, size_windows=6, seed=42, metric='linear', Nmax=2500, lam=0.01):
        '''
        The wrapper function for cartesianMDS, applying the gradient descent
        at each time step for a given window size and returning the cartesian
        coordinates as well as additional statistics such as the mean value and
        the std per moving window for each time series and the loss per epoch 
        for each moving window.
        
        Inputs
        ------
        size_windows : int (optional)
            Set the window size in months. Default is set to 6.
            
        seed : int (optional)
            Set the random seed for the initialization of coordinates. Default
            is set to 42.
        
        metric : str (optional)
            The metric used. Use the default option 'linear'. See documentation
            for more details.
            
        Nmax : int (optional)
            The maximum number of epochs. Default is set to 2500
            
        lam : float (optional)
            The learning rate, default is set to 0.01
        
        '''        
        
        self.metric=metric
        self.Nmax=Nmax
        self.lam=lam
        
        #extract month and year to specify the moving window to use
        win = np.zeros((2,len(self.date_range)))
        win[1,:] = list(self.date_range.month)
        win[0,:] = list(self.date_range.year)

        windows = np.unique(win,axis=1)
        total_windows = windows.shape[1]

        np.random.seed(seed)
        #initialize random coordiantes to start gradient descent with
        X=np.random.uniform(-0.5,0.5,(self.df.shape[1],3))
                
        self.windows=windows
        self.size_windows=size_windows
        self.X=X
        
        #initialize dictionaries containing the relevant attributes for each
        #moving window
        res_pos={} #position vectors
        res_mean={} #mean values
        res_std={} #std
        res_eps={} #final loss 
        res_eps_array={} #loss per epoch
        
        for idx in tqdm(range(total_windows-size_windows)):
            self.idx=idx
            mm,ss,ee_array,ee=self.cartesianMDS()
            
            res_pos[idx]=self.X.copy()
            res_mean[idx]=mm.copy()
            res_std[idx]=ss.copy()
            res_eps_array[idx]=ee_array.copy()
            res_eps[idx]=ee.copy()
            #set the final position vectors as new inputs for the gradient
            #descent in the next time step
            self.X=res_pos[idx]

        
        self.pos = res_pos
        self.mean = res_mean
        self.std = res_std
        self.eps_array = res_eps_array
        self.eps = res_eps
    
    
    def mean_distance(self):
        '''
        For each time step compute the mean distance of cartesian coordinates
        from their centroid (i.e. center of mass) as well as the mean standard
        deviation.
        '''
        mean_dist = np.zeros(len(self.pos))
        mean_dist_std = np.zeros(len(self.pos))
        
        for tt in range(len(self.pos)):
            mean_dist[tt] = np.sqrt(((self.pos[tt]-self.pos[tt].mean(axis=0))**2).sum(axis=1)).mean()
            mean_dist_std[tt] = (np.sqrt(((self.pos[tt]-self.pos[tt].mean(axis=0))**2).sum(axis=1))).std()
        
        self.mean_dist = mean_dist
        self.mean_dist_std = mean_dist_std
    
    
    def plot_3d_state(self):
        '''
        The final visualization, containing a mean distance plot and a
        3d scatter plot with an interactive slider option for analyszing the
        trajectory. The scatter plot is colored in relation to the mean value
        per time series per time step and the sizes of the markers are set
        proportional to the standard deviation of the time series at each time 
        step.         
        '''
        
        #compute the mean distance
        self.mean_distance()
        
        #define the array of time steps for the title of the plot
        time_array = np.empty(self.windows.shape[1],dtype='object')
        
        for idx in range(self.windows.shape[1]):
            if int(self.windows[1,idx])//10 == 0:
                time_array[idx] = '0'+str(int(self.windows[1,idx]))+'-'+str(int(self.windows[0,idx]))
            else:
                time_array[idx] = str(int(self.windows[1,idx]))+'-'+str(int(self.windows[0,idx]))
                
        #set min, max and initial values for the slider        
        tt_min = 0
        tt_max = len(self.pos.keys())-1
        tt_init = 0
        
        #initialize figure with two subplots
        fig = plt.figure(figsize=(12,6))
        gridspec.GridSpec(2,3)
        
        #first plot: mean distance from centroid
        ax_mean = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=2)
        plt.plot(self.mean_dist,color='black',linewidth=0.8)
        #create a marker, marking the current time step of the slider
        marker_pos, = plt.plot(tt_init,self.mean_dist[tt_init],'ro')
        ax_mean.set_title('mean distance from centroid')  
        
        #second plot: 3d scatter plot of abstract 3d coordinates
        ax_3d=plt.subplot2grid((2,3), (0,1), colspan=2, rowspan=3,projection='3d')
        ax_3d.set_xlim([-1,1])
        ax_3d.set_ylim([-1,1])
        ax_3d.set_zlim([-1,1])
        ax_3d.set_title('Correlation at {}'.format(time_array[tt_init]))
        
        #set color scale for mean values
        cm=plt.cm.get_cmap('inferno', 7)
        cNorm = mlt.colors.Normalize(vmin=min(self.df.mean()), vmax=max(self.df.mean()))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        #scatter plot for the initial value of the slider
        sc=ax_3d.scatter(self.pos[tt_init][:,0],self.pos[tt_init][:,1],self.pos[tt_init][:,2],
                         marker='s',
                         s=25*self.std[tt_init],
                         c=self.mean[tt_init],
                         cmap=cm,
                         edgecolors='green',linewidth=0.4
                        )
        
        cbar = fig.colorbar(scalarMap)
        cbar.set_label('mean stock price [€]', rotation=90)
        
        #define slider
        slider_ax = plt.axes([0.3, 0.05, 0.5, 0.05])
        tt_slider = Slider(slider_ax, 'time:', tt_min, tt_max, valinit=tt_init,valstep=1)
        
        #define update function for interaction with the plot (via slider)
        def update(tt):
            tt=int(tt)
            
            #update marker position
            marker_pos.set_data(tt,self.mean_dist[tt])
            #update coordinate positions
            sc._offsets3d = juggle_axes(self.pos[tt][:,0],self.pos[tt][:,1],self.pos[tt][:,2], 'z')
            #update marker sizes of coordinates
            sc._sizes3d = 25*self.std[tt]
            #update title
            ax_3d.set_title('Correlation at {}'.format(time_array[tt]))
            #update color of markers: not working properly!
            sc._faceolor3d=(self.mean[tt]) 
            
            fig.canvas.draw_idle()
        
        tt_slider.on_changed(update)    

        plt.show()
        
        return(tt_slider)