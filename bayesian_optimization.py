import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
from bayesian_optimization_util import plot_approximation, plot_acquisition
from bayesian_optimization_util import plot_convergence
from acquisition_functions import propose_location, expected_improvement

bounds = np.array([[-5.0, 6.0]])
noise = 0.2

def f(X, noise=noise):
    #return -np.sin(3*X) - X**2 + 0.7*X + noise * np.random.randn(*X.shape)
    f = np.poly1d([1, -2, -28, 28, 12, -26, 100])

    # Return the value of the polynomial
    return f(X) * -0.05

X_init = np.array([[-2.0], [5.5]])
Y_init = f(X_init)


# Dense grid of points within bounds
X = np.arange(bounds[:, 0], bounds[:, 1], 0.01).reshape(-1, 1)


# Noise-free objective function values at X 
Y = f(X,0)

# Plot optimization objective with noise level 
#plt.plot(X, Y, 'y--', lw=2, label='Noise-free objective')
#plt.plot(X, f(X), 'bx', lw=1, alpha=0.1, label='Noisy samples')
#plt.plot(X_init, Y_init, 'kx', mew=3, label='Initial samples')
#plt.legend()


# Gaussian process with Mat??rn kernel as surrogate model
m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2)

# Initialize samples
X_sample = X_init
Y_sample = Y_init

# Number of iterations
n_iter = 10

plt.figure(figsize=(12, n_iter * 3))
plt.subplots_adjust(hspace=0.4)

for i in range(n_iter):
    # Update Gaussian process with existing samples
    gpr.fit(X_sample, Y_sample)

    # Obtain next sampling point from the acquisition function (expected_improvement)
    X_next = propose_location(expected_improvement, X_sample, Y_sample, gpr, bounds)
    
    # Obtain next noisy sample from the objective function
    Y_next = f(X_next, noise)
    
    # Plot samples, surrogate function, noise-free objective and next sampling location
    plt.subplot(n_iter, 2, 2 * i + 1)
    plot_approximation(gpr, X, Y, X_sample, Y_sample, X_next, show_legend=i==0)
    plt.title(f'Iteration {i+1}')

    plt.subplot(n_iter, 2, 2 * i + 2)
    plot_acquisition(X, expected_improvement(X, X_sample, Y_sample, gpr), X_next, show_legend=i==0)
    
    # Add sample to previous samples
    X_sample = np.vstack((X_sample, X_next))
    Y_sample = np.vstack((Y_sample, Y_next))

plot_convergence(X_sample, Y_sample)
plt.show()