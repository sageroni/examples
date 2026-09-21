import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from optimization_test_functions import himmelblaus_function


def plot_2d_surface(function, x_lims, y_lims, n=100):
    '''
    Utility function for plotting a 2d surface
    :param function: the function defining the surface
    :param x_lims: a tuple of min/max x values to plot
    :param y_lims: a tuple of min/max y values to plot
    :param n: number of points between min and max values to plot
    '''
    x = np.linspace(x_lims[0], x_lims[1], n)
    y = np.linspace(y_lims[0], y_lims[1], n)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            Z[i, j] = function([float(X[i, j]), float(Y[i, j])])

    # Create a 2D flat layout
    fig, ax = plt.subplots()

    # plot a filled contour mapping (100 levels provides a smooth gradient)
    contour = ax.contourf(X, Y, Z, 100, cmap="plasma")
    ax.contour(X, Y, Z)
    fig.colorbar(contour, ax=ax)


plot_2d_surface(himmelblaus_function, [-5, 5], [-5, 5])


test_pts = []

def obj_fun_wrapper(input_vector):
    test_pts.append(input_vector)
    return himmelblaus_function(input_vector)

result = minimize(obj_fun_wrapper, x0=[-2, -2], method='CG')

# it used 157 diffeent test points when it was flip flopping the test points with the triangle.

# something we can do to check the way it is flip flopping is to create our own objective function
print(result)

for pt in test_pts:
    plt.scatter(pt[0], pt[1], color='black', s=10)

plt.show()