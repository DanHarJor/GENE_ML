import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

import sys
import os
sys.path.append('/home/djdaniel/GENE_UQ/GENE_ML')
from GENE_ML.gene_ml.uncertianty_quantification.distributions.gaussian import Gaussian
from GENE_ML.gene_ml.uncertianty_quantification.distributions.uniform import Uniform
from GENE_ML.gene_ml.uncertianty_quantification.montecarlo import Montecarlo



class MaxOfManyGaussians():
    def __init__(self, num_dim, num_gaussians, bounds=None, mean_bounds=None, std_bounds=None, seed=42):
        self.num_dim = num_dim
        self.num_gaussians = num_gaussians
        if type(bounds) == type(None):
            self.bounds = np.repeat((0,1),num_dim)
        else: self.bounds = np.array(bounds)
        if type(std_bounds) == type(None):
            self.std_bounds = np.array([(b[0], b[0]+(b[1]-b[0])/2 ) for b in bounds])
        else: self.std_bounds = np.array(std_bounds)
        if type(mean_bounds) == type(None):
            self.mean_bounds=self.bounds
        else: self.mean_bounds = mean_bounds
        
        self.rg = np.random.default_rng(seed=seed)
        self.gaussians = self.generate_gaussians()

    def generate_gaussians(self):
        gaussians = []
        # Generate multiple Gaussians
        for _ in range(self.num_gaussians):
            mean = []
            for b in self.mean_bounds:
                mean.append(self.rg.uniform(*b, 1)[0])
            mean = np.array(mean)
            cov_bounds = self.std_bounds**2
            # cov = self.rg.uniform(*cov_bounds, (self.num_dim, self.num_dim))
            # cov = np.dot(cov, cov.T)  # Ensure the covariance matrix is positive semi-definite
            print('CB',cov_bounds)
            cov = np.diag(self.rg.uniform(*cov_bounds, (self.num_dim,self.num_dim)))
            gaussians.append(multivariate_normal(mean, cov))
        return gaussians

    def evaluate(self, pos):
        Z = []
        for g in self.gaussians:
            Z.append(g.pdf(pos))
        Z = np.array(Z)
        Zmax = np.max(np.stack(Z), axis = 0)

        #set out of bounds to 0
        for i in range(self.num_dim):
            b1 = self.bounds[i][0]
            b2 = self.bounds[i][1]
            if isinstance(Zmax, np.ndarray):
                p = pos[...,i]
                Zmax[(p < b1) | (p > b2)] = 0
            elif isinstance(Zmax, int):
                if pos[i] > b1 or pos[i] < b2: 
                    Zmax = 0
        return Zmax
    
    def plot_slices(self, grid_size=200, nominals=None):
        print('PLOT SLICES')
        if type(nominals) == type(None):
            nominals = [np.mean(b) for b in self.bounds]
        
        for i, b in enumerate(self.bounds):
            p = np.stack([nominals for i in range(grid_size)])
            print('d', p.shape)
            x = np.linspace(b[0],b[1], grid_size)
            p[:,i] = x
            y_true = self.evaluate(p)
            fig = plt.figure()
            plt.plot(x,y_true, color ='black', label='True Function')
            plt.legend()
            plt.xlabel(f'dimension {i}')
            plt.ylabel('function value')
            fig.show()

    def plot_matrix_contour(self):
        w=2
        h=2
        figure, AX = plt.subplots(self.num_dim, self.num_dim, figsize=(w*self.num_dim, h*self.num_dim), sharex=True, sharey=True)
        for i in range(self.num_dim):
            for j in range(self.num_dim):
                if j>=i:
                    figure.delaxes(AX[i,j])
                    # break
                else:
                    self.plot_2D_of_many(which2=(j,i),ax=AX[i,j], style='contour', grid_size=50)
                    if j==0:
                        AX[i,j].set_ylabel(f'{i}')
                    if i==self.num_dim-1:
                        AX[i,j].set_xlabel(f'{j}')
        figure.show()

        
    def plot_2D_of_many(self, which2, extra=0, plot_bounds=None, nominals=None, grid_size=100, style='3D', ax=None):
        # which2 is a sequence that specifies which dimensions to plot, the rest are kept nominal, example which2 = (0,2) to plot the 1st and 3rd dimensions. 
        if plot_bounds == None:
            plot_bounds = self.bounds
        if type(nominals) == type(None):
            nominals = [np.mean(b) for b in self.bounds]
        
        xlow, xhigh = plot_bounds[which2[0]][0]-extra, plot_bounds[which2[0]][1]+extra
        ylow, yhigh = plot_bounds[which2[1]][0]-extra, plot_bounds[which2[1]][1]+extra
        x = np.linspace(xlow, xhigh, grid_size)
        y = np.linspace(ylow, yhigh, grid_size)
        X, Y = np.meshgrid(x, y)
        
        arrays2stack = []
        for i, n in enumerate(nominals):
            arrays2stack.append(np.full_like(X,n))
        arrays2stack[which2[0]] = X
        arrays2stack[which2[1]] = Y
        pos = np.dstack(arrays2stack)
        
        # print(pos.shape)
        Z = np.zeros(shape=(grid_size,grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                p = nominals
                p[which2[0]] = X[i,j]
                p[which2[1]] = Y[i,j]
                Z[i,j] = self.evaluate(p)

        if style == '3D':
            if type(ax) == type(None):
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')    
            ax.plot_surface(X, Y, Z, cmap='viridis')
        else:
            if type(ax) == type(None):
                fig = plt.figure(figsize=(2,2), dpi=200)
                ax = fig.add_subplot(111)
            ax.contour(X,Y,Z, cmap='viridis')
        
        if type(ax) == type(None):
            ax.set_xlabel(f'{which2[0]}')
            ax.set_ylabel(f'{which2[1]}')
            fig.show()
            

    def plot_2d_gaussians(self, grid_size=100, onlyContour=False, plot_bounds=None, extra=0):
        if plot_bounds == None:
            plot_bounds = self.bounds
        if self.num_dim != 2:
            raise ValueError('Daniel Says: n_dim must equil 2')
        xlow, xhigh = plot_bounds[0][0]-extra, plot_bounds[0][1]+extra
        ylow, yhigh = plot_bounds[1][0]-extra, plot_bounds[1][1]+extra
        x = np.linspace(xlow, xhigh, grid_size)
        y = np.linspace(ylow, yhigh, grid_size)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
                
        
        Zmax = self.evaluate(pos)
        
        z = []
        yy = 0.5
        x_at_y = [(xi, yy) for xi in x]
        for g in self.gaussians:
            z.append(g.pdf(x_at_y))
        zmax = np.max(np.stack(z), axis = 0)
        #slice
        if not onlyContour:
            plt.figure()
            plt.plot(x, zmax)
            plt.show()
            
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_surface(X, Y, Zmax, cmap='viridis')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Function Value')
            # ax.set_title('2D Surface of Multimodal Multivariate Gaussian Distribution')
            ax.view_init(elev=30, azim=30-90)

        fig = plt.figure(figsize=(4,4))
        plt.contour(X,Y,Zmax)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.xlabel('X')
        plt.ylabel('Y')

    # def do_UQ(self, n_samples, n_bins):
    #     #This assums each dimensions is centered around a nominal value and the bounds are uncertanties.
    #     bounds = np.array(self.bounds)
    #     nominals = np.mean(bounds, axis=1)
    #     print('NOM',nominals)
    #     distributions = []
    #     for bound in bounds:
    #         distributions.append(Uniform(bound))
    #     montecarlo = Montecarlo(self.evaluate, distributions)
    #     montecarlo.compute_output_distribution(n_samples)
    #     montecarlo.plot_output_distribution(n_bins)


if __name__ == '__main__':
    seed = 10
    mmg = MaxOfManyGaussians(num_dim=2, num_gaussians=5, bounds=[(0.4,0.6), (0.4,0.6)], std_bounds=(0.005,0.008), seed=seed)
    # mmg = MaxOfManyGaussians(2, 5, [(0,1), (0,1)], std_bounds=(0.1,0.1), rg=rg)

    mmg.plot_2d_gaussians()

    # max_value, max_pos = generate_2d_gaussians(num_gaussians, grid_size)
    # print(f"Maximum Gaussian value: {max_value} at position {max_pos}")

