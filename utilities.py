
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from math import floor, ceil


def rootfinding1(func, x_min=-100, x_max=100, max_iter=100, func_arg=[], random_search=True):
    '''
    Solve an equation by bisection method.
    
    Parameters:
        func: function
            a function to solve
        x_min, x_max: int or float
            minimum and maximum of range of x where you would like to find a solution
        max_iter: int (> 0)
            number of iterations to find a solution
        func_arg: [](None) or list of argments
            argments of func except for variable x
        random_search: boolean
            whether to initialize x at the first iteration
    Returns:
        med: float
            optimal value of x
    
    Example:
        >>>import numpy as np
        >>>from utilities import rootfinding1

        >>>f = lambda x: np.sqrt(x) - 5
        >>>rootfinding1(f)
        25.000000000000007
    '''
    if random_search:
        x = np.random.random(max_iter)*(x_max - x_min) + x_min
        y = func(*([x]+func_arg))
        try:
            posi = x[y > 0][0]
            nega = x[y < 0][0]
        except:
            return 'Solution was not found'
    else:
        if func(*([x_min]+func_arg)) < 0:
            nega, posi = x_min, x_max
        else:
            nega, posi = x_max, x_min
    
    for i in range(max_iter):
        med = (posi + nega) / 2
        y_m = func([med]+func_arg)
        
        if y_m > 0:
            posi = med
        else:
            nega = med
    
    return med



def rootfinding2(func, x_min=0, x_max=1, max_iter=100, h=.0001, func_arg=[]):
    '''
    Solve an equation by Newton method.
    
    Parameters:
        func: function
            a function to solve
        x_min, x_max: int or float
            minimum and maximum of range of x where you would like to find a solution
        max_iter: int (> 0)
            number of iterations to find a solution
        h: float, (0<h<1)
            parameter of differentiation. The smaller h is, the more accurate derivative you will get. However, extremely small h (e.g. 1e-20) can 
            lead to unstable operation. 0.1 < h < 1e-10 is recommended.
        func_arg: [](None) or list of argments
            argments of func except for variable x
    Returns:
        a: float
            optimal value of x
    
    Example:
    # computing square root 2
    >>>import numpy as np
    >>>from utilities import rootfinding2

    >>>f = lambda x: x**2 - 2
    >>>rootfinding2(f)
    1.4142135623730951
    '''
    a = np.random.random()*(x_max - x_min) + x_min
    
    for i in range(max_iter):
        der = differentiate(func, a, h=h, func_arg=func_arg)
        y = func(*([a]+func_arg))
        a = a - y/der
    
    return a



def rootfinding3(func, x_min=False, x_max=False, max_iter=100, func_arg=[]):
    '''
    Solve an equation by secant method.
    
    Parameters:
        func: function
            a function to solve
        x_min, x_max: int or float
            minimum and maximum of range of x where you would like to find a solution
        max_iter: int (> 0)
            number of iterations to find a solution
        func_arg: [](None) or list of argments
            argments of func except for variable x
    Returns:
        a: float
            optimal value of x
    '''
    a,b = np.random.random(2)
    
    for i in range(max_iter):
        f_a, f_b = func(*([a]+func_arg)), func(*([b]+func_arg))
        
        num = a - b
        denom = f_a - f_b
        if denom == 0:
            return a
        
        a, b = a - f_a*num / denom, a
        
    return a



def rootfindingNR(func_list, init, h=.001, max_iter=10):
    '''
    Solve linear or non-linear equation system (set of equations) by Newton-Raphson Method
    
    Parameters:
        func_list: list or tuple
            set of equations to solve
        init: array-like
            set of initial values of iteration. 0 should be avoid because inverse jacobian explodes
        h: float
            parameter of differentiation to compute jacobian matrix
        max_iter: int
            number of iterations to solve
    
    Returns:
        x: array-like
            solution of the system
    
    Example:
    
    >>>import numpy as np
    >>>from utilities import rootfindingNR

    >>>f1 = lambda x1, x2: x1**2 - 2
    >>>f2 = lambda x1, x2: x1*x2

    >>>rootfindingNR([f1, f2], [1,1])
    
    array([1.41421356, 0.        ])
    '''
    x = init
    for i in range(max_iter):
        x = x - inv(jacobian(func_list, x, h=h)).dot(np.array([f(*x) for f in func_list]))
    return x



def comb(a, b):
    '''
    Computes combination (binomial coefficient) of a and b (combinations to choose b out of a i.e. a!/(b!(a-b)!))
    
    Parameters:
        a: int
            total number of states
        b: int, 0 <= b <= a
            number of choise out of total nubmer
    
    Returns:
        out: int
            number of combinations of b out of a
    
    Example:
    >>>from utilities import comb
    >>>comb(5, 2)
    10.0
    '''
    if b == 0:
        return 1
    elif b == 1:
        return a
    else:
        return (a/b)*comb(a-1, b-1)



def factorial(k):
    '''
    Returns k factorial: k! = 1 * 2 * ... * k
    '''
    if k == 1 or k == 0:
        return 1
    else:
        return k*factorial(k-1)


def stirling_log(n, log=True):
    if log==True:
        return np.log(np.sqrt(2*np.pi*n)) + n*np.log(n/np.e)
    else:
        return np.sqrt(2*np.pi*n)*(n/np.e)**n


def differentiate(func, a, h=.001, order=1, func_arg=[], float128=False):
    '''
    Computes k-th derivative of the given function

    Parameters:
        func: function
            A function to differentiate
        a: float or int
            A value at which you would like to take a derivative
        h: float
            Parameter of differentiation. The smaller h is, the more accurate value you take. However, exremely small h
            (e.g. h=1e-20) may cause unstable computation. Moderate value (e.g. 0.1 < h < 1e-10) is recommended.
        order: int, >0
            The order of derivative
        func_arg: list
            Argments of the function except for variable.
        float128: boolean
            If float128=True, you will take slightly more accurate value of higher-order derivative.

    Returns:
        out: float
            Value of derivative

    Example:
    >>>import numpy as np
    >>>from utilities import differentiate

    # compute derivateive of sin(x) i.e. cos(x) at 4 points :[0, np.pi/3, np.pi/2, np.pi]
    >>>a = np.array([0, np.pi/3, np.pi/2, np.pi])
    >>>print(differentiate(np.sin, a))
    [ 9.99999833e-01  4.99566904e-01 -4.99999958e-04 -9.99999833e-01]

    # smaller h: more accurate
    >>>print(differentiate(np.sin, a, h=1e-10))
    [ 1.          0.50000004  0.         -1.00000008]
    '''
    if float128:
        a, h = np.float128(a), np.float128(h)

    if order == 0:
        return func(*([a]+func_arg))

    if order == 1:
        return (func(*([a+h]+func_arg)) - func(*([a]+func_arg)))/h

    else:
        return (differentiate(func, a+h, h=h, order=order-1, func_arg=func_arg) - differentiate(func, a, h=h, order=order-1, func_arg=func_arg))/h


def partial_diff(func, func_arg, variable=0, h=.001, order=1, var_order=False):
    '''
    Computes partial derivative of the given function
    
    Parameters:
        func: function
            A function to take its partial derivative
        func_arg: list
            All argments of the given function including variable
        variable: int
            The position (index) of variable to take derivative of all argments
        h: float
            Parameter of differentiation. The smaller h is, the more accurate value you take. However, exremely small h
        (e.g. h=1e-20) may cause unstable computation. Moderate value (e.g. 0.1 < h < 1e-10) is recommended.
        order: int, >0
            The order of derivative
        var_order: False or list
            Order of variables to take its derivatives
    
    Returns:
        out: float
            Partial derivatives of the given function and designated variable
    
    Example:
    >>>import numpy as np
    >>>from utilities import partial_diff

    # function: sin(x) + x*y + log(y)
    >>>func = lambda x, y: np.sin(x) + x*y + np.log(y)

    # take partial derivative at (x, y) = (1, 1)
    >>>func_arg = [1, 1]

    # partial f/partial x = cos(x) + y = cos(1) + 1
    >>>print(partial_diff(func, func_arg, variable=0))
    1.5398814803602168

    # partial f/partial y = x + 1/y = 1 + 1/1
    >>>print(partial_diff(func, func_arg, variable=1))
    1.9995003330832706
    '''
    if var_order:
        variable=var_order[-1]
    func_arg2 = func_arg.copy()
    func_arg2[variable] += h
    if order==1:
        return (func(*func_arg2) - func(*func_arg))/h
    else:
        if var_order:
            return (partial_diff(func, func_arg2, variable=var_order[0], h=h, order=order-1, var_order=var_order[:-1])
                    - partial_diff(func, func_arg, variable=var_order[0], h=h, order=order-1, var_order=var_order[:-1]))/h
        else:
            return (partial_diff(func, func_arg2, variable, h=h, order=order-1) - partial_diff(func, func_arg, variable, h=h, order=order-1))/h


def grad(func, func_arg, h=.001):
    '''
    Compute gradient of the given function

    Parameters:
        func: function
            A function to take its gradient
        func_arg: list
            All argments of the given function including variable
        h: float
            Parameter of differentiation. The smaller h is, the more accurate value you take. However, exremely small h
        (e.g. h=1e-20) may cause unstable computation. Moderate value (e.g. 0.1 < h < 1e-10) is recommended.

    Returns:
        out:
            Gradient of the given function

    Example:
    >>>import numpy as np
    >>>from utilities import grad

    >>>func = lambda x, y, z: x**2 + x*y + 3*z
    >>>func_arg = [3, 4, 5]
    >>>grad(func, func_arg)
    array([10.001,  3.   ,  3.   ])
    '''
    
    variables = list(range(len(func_arg)))
    gradient = []
    for i in variables:
        gradient.append(partial_diff(func, func_arg, variable=i, h=h, order=1))
    gradient = np.array(gradient)
    return gradient



def jacobian(func_list, init, determinant=False, h=.001):
    '''
    Compute Jacobian matrix of the given function and variables

    Parameters:
        func_list: list
            list of functions to compute jacobian matrix
        init: array-like
            list of values of variables at which you would like to compute jacobian matrix
        determinant: boolean
            If True, jacobian function computes det (determinant) of the Jacobian matrix. If False, just computes the
            Jacobian matrix
        h: float
            Parameter of differentiation. The smaller h is, the more accurate value you take. However, exremely small h
        (e.g. h=1e-20) may cause unstable computation. Moderate value (e.g. 0.1 < h < 1e-10) is recommended.

    Returns:
        out:
            Jacobian matrix (or its determinant) of the given function and variable

    Example:

        >>>import numpy as np
        >>>from utilities import jacobian

        # Coordinate transformation from polar coordinate to Cartesian coordinate
        # (x , y) = f(r , θ) = (r cos θ, r sin θ)
        >>>r_0 = 2
        >>>theta_0 = np.pi / 4

        >>>f1 = lambda r, theta: r*np.cos(theta)
        >>>f2 = lambda r, theta: r*np.sin(theta)

        >>>jacobian([f1, f2], [r_0, theta_0])

        array([[ 0.70710678, -1.41492043],
           [ 0.70710678,  1.41350622]])

    '''
    variables = list(range(len(func_list)))
    J = np.zeros([len(variables), len(func_list)])
    for i in range(len(func_list)):
        J[i, :] = grad(func_list[i], init, h=h)
    if determinant and J.shape[0] == J.shape[1]:
        return det(J)
    else:
        return J



def hessian(func, func_arg, h=.001):
    '''
    Computes Hessian matrix of the given function and variables

    Parameters:
        func: function
            A function to compute its Hessian matrix

        func_arg: list
            List of values at which you would like to compute Hessian matrix.

        h: float
            Parameter of differentiation. The smaller h is, the more accurate value you take. However, exremely small h
        (e.g. h=1e-20) may cause unstable computation. Moderate value (e.g. 0.1 < h < 1e-10) is recommended.

    Returns:
        H: array
            A Hessian matrix of the given function and variables

    Example:
        >>>import numpy as np
        >>>from utilities import hessian

        >>>func = lambda x, y, z: x**2*y*z + 3*y**3
        >>>func_arg = [2, 3, 4] 

        >>>hessian(func, func_arg)

        array([[24.        , 16.004     , 12.00299999],
           [16.004     , 54.01799999,  3.99999999],
           [12.00299999,  3.99999999,  0.        ]])

    '''
    variables = list(range(len(func_arg)))
    H = np.zeros([len(variables), len(variables)])
    for i in variables:
        for j in variables:
            H[i, j] = partial_diff(func, func_arg, variable=0, h=.001, order=2, var_order=[i, j])
    return H



def integral1(func, lower, upper, h, func_arg=[]):
    n = int((upper - lower) // h)
    sum_ = 0
    for i in range(n):
        sum_ += func(*([lower + i*h]+func_arg)) * h
    
    return sum_



def integral2(func, lower, upper, h, func_arg=[]):
    n = int((upper - lower) // h)
    h_ = (upper - lower) - n*h
    edges = np.array([lower + i*h for i in range(n)] + [upper])
    edges1, edges2 = edges[:-1], edges[1:]
    main = ((func(*([edges2[:-1]]+func_arg)) + func(*([edges1[:-1]]+func_arg)))*h*.5).sum()
    margin = (func(*([edges2[-1]]+func_arg)) + func(*([edges1[-1]]+func_arg)))*h_*.5
    
    return main + margin



def integral3(func, lower, upper, h, func_arg=[]):
    n = np.floor((upper - lower) // h).astype(int)
    
    if n % 2 ==1:
        n = n+1
    edges = np.array([lower + i*h for i in range(n+1)])
    edges1 = edges[[2*i-2 for i in range(1, int(n/2)+1)]]
    edges2 = edges[[2*i-1 for i in range(1, int(n/2)+1)]]
    edges3 = edges[[2*i for i in range(1, int(n/2)+1)]]
    sum_ = (func(*([edges1]+func_arg)) + 4*func(*([edges2]+func_arg)) + func(*([edges3]+func_arg))).sum()
    
    return h/3*sum_



def integral4(func, lower, upper, h, func_arg=[]):
    n = np.floor((upper - lower) // h).astype(int)
    
    if n % 3 == 2:
        n = n+1
    if n % 3 == 1:
        n = n-1
    edges = np.array([lower + i*h for i in range(n+1)])
    edges1 = edges[[i for i in range(1, n-1) if i % 3 != 0]]
    edges2 = edges[[3*i for i in range(1, int(n/3)-1)]]
    
    term1 = func(*([lower]+func_arg))
    term2 = func(*([edges1]+func_arg)).sum()
    term3 = func(*([edges2]+func_arg)).sum()
    term4 = func(*([upper]+func_arg))
    
    return 3*h/8*(term1 + 3*term2 + 2*term3 + term4)



def integral_MonteCarlo(func, lower, upper, h=False, n_sample=1e+4, rv='uniform', func_arg=[]):
    rvs = np.random.random(int(n_sample))*(upper-lower) + lower
    return (upper - lower)*func(*([rvs]+func_arg)).mean()



def integral(func, lower, upper, h, func_arg=[], algo='Simpson1', n_sample=1e+4, rv='uniform'):
    '''
    Computes integral of the given function
    
    Parameters:
        func: function
            A function to integrate
        lower: int or float. lower < upper
            The minimum of range of integration
        upper: int or float. upper > lower
            The maximum of range of integration
        h = int or float
            The width of increment of integration. The smaller h is, the more accurate you will get but the longer computing
            time becomes.
        func_arg: list
            Argments of the function except for variable.
        
        algo: str
            Algorithm (as shown below) of computing integration.
            
            Algorithms:
                'ordinary':　Approximate area of integral by set of rectangles.
                'trapezoidal': Approximate area of integral by set of trapezoids.
                'Simpson1': Approximate area of integral by Simpson's rule.
                'Simpson2': Approximate area of integral by Composite Simpson's rule.
                'MonteCarlo': Approximate area of integral by Monte-Carlo simulation.
                
        n_sample: int, > 0 (optional, only when use MonteCarlo algorithm)
        
        rv: str (optional, only when use MonteCarlo algorithm)
    
    Returns:
        out: float
            The integral of the given function.
    
    Example:
    
    >>>import numpy as np
    >>>from utilities import integral

    >>>integral(np.sin, 0, np.pi, h=.001)
    
    1.9999999170344631
    '''
    if algo=='MonteCarlo':
        return  integral_MonteCarlo(func, lower, upper, h, n_sample, rv, func_arg)
    elif algo=='ordinary':
        return integral1(func, lower, upper, h, func_arg)
    elif algo=='trapezoidal':
        return integral2(func, lower, upper, h, func_arg)
    elif algo=='Simpson1':
        return integral3(func, lower, upper, h, func_arg)
    elif algo=='Simpson2':
        return integral4(func, lower, upper, h, func_arg)



def taylor(func, x, a, limit=5, h=.0001, func_arg=[], float128=False):
    '''
    Compute Taylor series (approximation) of the given function around point a i.e.
    f(x) = f(a) + f^1(a)/1! * (x - a) + f^2(a) / 2! * (x - a)**2 + ...
    
    Parameters:
        func: function
            A function to expand to Talor Series
        x: int, float or array-like
            Values of variable of function
        a: int or float
            A value around which you would like to compute Talor series
        limit: int, >0
            Number of terms of Taylor series. Number over 5 is not recommed because higher order derivative may be
            inaccurate.
        h: float
            Parameter of differentiation. The smaller h is, the more accurate value you take. However, exremely small h
            (e.g. h=1e-20) may cause unstable computation. Moderate value (e.g. 0.1 < h < 1e-10) is recommended.
        func_arg: list
            Argments of the function except for variable.
        float128: boolean
            If float128=True, you will take slightly more accurate value of higher-order derivative.
    
    Returns:
        out:
            Sum of Taylor series
    
    Example:
    >>>import numpy as np
    >>>from utilities import taylor

    >>>x = np.array([0.1, 0.2, 0.3])

    # exact value
    >>>print(np.exp(x))
    [1.10517092 1.22140276 1.34985881]

    # approximated value by Talor series
    >>>print(taylor(np.exp, x, 0, limit=5))
    [1.1051906853812845, 1.2216415117303563, 1.3510187017625732]
    '''
    
    y_list = []
    for x_ in x:
        y = func(*([a]+func_arg))
        for i in range(1, limit):
            term= differentiate(func, a, h=h, order=i, func_arg=func_arg, float128=float128)/factorial(i)*(x_-a)**i
            y += term
        
        y_list.append(y)
    
    return y_list



def convolution1d(func1, func2, t, lower=-100, upper=100, h=.01, func1_arg=[], func2_arg=[]):
    '''
    Compute convolution of two functions
    
    Parameters:
        func1, func2: function
            Two function to compute their convolution
        
        t: int, float or array-like
            Value(s) around which you would like to compute convolution
        
        lower, higher: int or float, lower < higher
            minimum and maximum values of range of integration in convolution
        
        h: float
            parameter of integration (width of increment) in convolution
        
        func1_arg, func2_arg: list
            Argments of two functions other than variable
    
    Returns:
        out:
            Value of convolution
    
    Examples:
    
    >>>import numpy as np
    >>>from utilities import convolution1d

    >>>t = np.array([0, 0.1, 0.2, 0.3, 0.4])
    >>>convolution1d(np.sin, np.cos, t)
    array([8.78081522e-03, 1.00354045e+01, 1.99617578e+01, 2.96886598e+01,
       3.91189225e+01])
    '''
    
    FUNC = lambda tau, t: func1(*([tau]+func1_arg))*func2(*([t - tau]+func2_arg))
    return integral(FUNC, lower=lower, upper=upper, h=h, func_arg=[t], algo='ordinary')



def Fourier_series(func, n, lower=-np.pi, upper=np.pi, h=.01, func_arg=[]):
    '''
    Compute coefficient of Fourier series (approximation) of the given function
    
    Parameters:
        func: function
            A function to compute its Fourier series
        n: int, >0
            Number of coefficeints. Total number of coefficients is 2n + 1 (sin, cos, and constant term)
        lower,upper: int or float
            Minimum and maximum of range of integration
        h: float
            parameter of integration (width of increment) in convolution
        func_arg: list
            Argments of function other than variable
    
    Returns:
        out: list
            List of coefficients. First element of the list is coefficient of constant term, the 2k - th (k is integer) element
            is coefficient of cos(kx) term and the 2k+1 -th (k is integer, k>1) element is coefficient of sin(kx) term.
    '''
    c = 1/(2*np.pi)*integral(func, lower, upper, h, func_arg)
    coef = [c]
    for i in range(1, n+1):
        func1 = lambda x: func(*([x]+func_arg))*np.cos(i*x)
        func2 = lambda x: func(*([x]+func_arg))*np.sin(i*x)
        
        coef.append(1/np.pi*integral(func1, lower, upper, h, func_arg))
        coef.append(1/np.pi*integral(func2, lower, upper, h, func_arg))
        
    return coef



def Fourier_transform(func, xi, lower=-100, upper=100, h=.01, func_arg=[], const=False, inverse=False):
    '''
    Compute Fourier transform of the given function.
    
    Parameters:
        func: function
            A function to transform
        xi: int, float or array-like
            Frequencies to which you would like to transform the original data
        lower, upper: int or float, lower < upper
            Minimum and maximum of range of integration in Fourier transform
        h: float
            parameter of integration (width of increment) in convolution
        func_arg: list
            Argments of function other than variable
        const: False, int or float
            Constant term in exponential of e. If False, default value (-2*pi in FT or 2*pi in inverse FT)
        inverse: boolean
            If true, ordinary Fourier transform is operated. If false, inverse Fourier transform is done.
    
    Returns:
        out: float or array-like
    '''
    if not const and not inverse:
        const = -2*np.pi
    elif not const and inverse:
        const = 2*np.pi
    func2 = lambda x, xi: func(*([x]+func_arg))*np.e**(const*1j*x*xi) 
    return integral(func2, lower=lower, upper=upper, h=h, func_arg=[xi], algo='ordinary')



def LU(A):
    '''
    Compute L (lower triangular) matrix and U (upper triangular) matrices of LU factorization
    
    Paramters:
        A: np.matrix or np.array (2 dimention)
            Matrix to factrize
    
    Returns:
        L, U: np.array(2 dimention)
            L and U matrices
    
    Example:
    
    >>>import numpy as np
    >>>from utilities import LU

    >>>A = np.array([[2,3,4], [5,1,3], [1, 8, 9]])

    >>>L, U = LU(A)
    >>>print(L)
    [[ 2.   0.   0. ]
     [ 5.  -6.5  0. ]
     [ 1.   6.5  0. ]]

    >>>print(U)
    [[1.         1.5        2.        ]
     [0.         1.         1.07692308]
     [0.         0.         1.        ]]
    '''
    
    L, U = np.zeros(A.shape), np.identity(A.shape[0])
    n_iter = A.shape[0]
    for i in range(n_iter):
        if i ==n_iter - 1:
            a = A[0,0]
            L[i, i] = a
        else:
            a = A[0,0]
            b = A[0, 1:]/a 
            c = A[1:, 0]
            d = A[1:,1:]
            
            L[i, i] = a
            U[i, i+1:] = b
            L[i+1:, i] = c
            A = d - c.reshape(len(c), 1).dot(b.reshape(1, len(b)))
    
    return L, U



def invLU(L):
    diagonal = []
    for i in range(len(L)):
        diagonal.append(L[i, i])
    diagonal = np.array(diagonal)
    
    Dinv = np.diag(1/diagonal)
    N = Dinv.dot(L) - np.eye(len(L))
    
    INinv = np.eye(len(L))
    for i in range(1, len(L)):
        INinv += (-1)**i*ndot(N, i)
    
    return INinv.dot(Dinv)



def inv(A):
    '''
    Compute inverse matrix of the given matrix using LU factorization
    
    Parameters:
        A: np.matrix or np.array(2-dimention)
            Matrix to take its inverse
            
    Returns:
        invA: np.array
            Inverse of the given matrix
    
    Example:
    
    >>>import numpy as np
    >>>from utilities import inv

    >>>A = np.array([[2,3,4], [5,1,3], [1, 8, 0]])

    >>>print(inv(A))
    
    [[-0.20512821  0.27350427  0.04273504]
     [ 0.02564103 -0.03418803  0.11965812]
     [ 0.33333333 -0.11111111 -0.11111111]]
    '''
    
    L, U = LU(A)
    return invLU(U).dot(invLU(L))


def inner(v1, v2, conjugate=False):
    '''
    Compute inner product of two vectors
    
    Parameters:
        v1, v2: array-like, list or tuple
            Two vectors to compute their inner product
        
        conjugate: boolean
            If true, compute inner product of one vector and complex conjugate of the other vector
    
    Returns:
        out: float or complex
            Inner product of the two vectors
    '''
    v1, v2 = np.array(v1), np.array(v2)
    if conjugate:
        return conj(v1).T.dot(v2) 
    else:
        return v1.T.dot(v2)

def norm2(x):
    x = np.array(x)
    return np.sqrt((x**2).sum())

def norm1(x):
    x = np.array(x)
    return abs(x).sum()

def norm_inner(v, conjugate=False):
    return np.sqrt(inner(v, v, conjugate=conjugate))

def norm_f(A):
    AA = A.T.dot(A)
    return np.sqrt(trace(AA))

def norm_p(x, p):
    x = np.array(x)
    return ((x**p).sum())**(1/p)

def norm_max(x):
    return x.max()

def norm(x, norm_type='L2', conjugate=False, p=2):
    '''
    Compute various kinds of norm of the given vector or matrix.
    
    Parameters:
        x: array-like
            Vector or matrix to compute its norm
        
        norm_type: str
            Type of norm. You can choose type from the followings.
            
            'L1': L1 norm of Manhattan distance
            'L2': L2 norm or Euclidean norm
            'inner': Same as L2 norm but computes norm using inner product
            'p': p norm. You need to specify parameter p
            'max': Max norm
            'Frobenius': Frobenius norm of matrix
        
        conjugate: boolean (optional)
            You need to specify only when you choose inner product norm. 
            If true, compute inner product of one vector and complex conjugate of the other vector
        
        p: int, > 0 (optional)
            Parameter p of p-norm
            
    Returns:
        out: float
            Norm of the given matrix or vector.
    
    Example:
    
    >>>import numpy as np
    >>>from utilities import norm

    >>>v = np.array([1, 0.5, -2])

    >>>print(norm(v, norm_type='L1'))
    3.5
    
    >>>print(norm(v, norm_type='L2'))
    2.29128784747792
    
    >>>print(norm(v, norm_type='inner'))
    2.29128784747792
    
    >>>print(norm(v, norm_type='p', p=4))
    2.032406925412472
    
    >>>print(norm(v, norm_type='max'))
    1.0
    
    >>>A = np.array([[1,2,3], [4,5,6], [7, 8, 9]])
    >>>print(norm(A, norm_type='Frobenius'))
    16.881943016134134
    '''
    
    if norm_type == 'L2':
        return norm2(x)
    elif norm_type =='L1':
        return norm1(x)
    elif norm_type =='inner':
        return norm_inner(x)
    elif norm_type == 'p':
        return norm_p(x, p)
    elif norm_type == 'max':
        return norm_max(x)
    elif norm_type == 'Frobenius':
        return norm_f(x)


def round_complex(c, n=0):
    '''
    Round off the given complex number to the n-th decimal place
    
    Parameters:
        c: complex
            Complex number to round off

        n: int, >= 0
            The order of decimal point to which you would like to round off
    
    Returns:
        out: complex
            Complex number rounded off
            
    Example:
    
    >>>import numpy as np
    >>>from utilities import round_complex

    >>>c = np.random.random() + np.random.random()*1j
    
    >>>print(c)
    (0.46649001556310365+0.9117226928612181j)

    >>>print(round_complex(c, 2))
    (0.47+0.91j)
    '''
    c = np.array(c)
    if (c.imag == 0).all():
        return c.real.round(n)
    else:
        return c.real.round(n) + c.imag.round(n)*1j


def conj(c):
    '''
    Returns complex conjugate of the given complex number c
    '''
    return c.real-c.imag*1j


def H(A):
    '''
    Returns conjugate transpose of the given matrix A. It is similar to method of np.matrix .H but H() works for 
    np.array as well.
    '''
    return conj(A.T)



def det2(A):
    A = np.array(A)
    return A[0,0]*A[1,1] - A[0,1]*A[1,0]



def det3(A):
    A = np.array(A)
    A = np.hstack([A, A])
    plus = 0
    minus=0
    for i in range(3):
        plus += A[0, i]*A[1, i+1]*A[2, i+2]
        minus += A[0, 5-i]*A[1, 4-i]*A[2, 3-i]
    return plus-minus



def cofactor(A, k, l):
    A = np.array(A)
    cofac = []
    A[k, :] = float('inf')
    A[:, l] = float('inf')
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] != float('inf'):
                cofac.append(A[i, j])
    
    cofac = np.array(cofac).reshape([A.shape[0]-1, A.shape[1]-1])
    return cofac



def det(A):
    '''
    Compute determinant of the given matrix A
    '''
    
    if len(A) == 2:
        return det2(A)
    elif len(A) == 3:
        return det3(A)
    else:
        sum_ = 0
        for i in range(A.shape[0]):
            sum_ += (-1)**i*A[i, 0]*det(cofactor(A, i, 0))
        return sum_



def diagonalize(A):
    '''
    Diagonalize (eigenvalue decomposition) of the given matrix A i.e. A = PLP^(-1)
    
    Parameter:
        A: np.matrix or np.array(2D)
    
    Returns
        Each returns correspond PLP^(-1) = A i.e.
        
        P: np.array(2D)
            Matrix where column vectors are eigenvectors of the given matrix
        
        L: np.array(2D)
            Dinagonal matrix where diagonal entries are eigenvalues of the given matrix
        
        invP: np.array(2D)
            Inverse of matrix P
    
    Example:
    
    >>>import numpy as np
    >>>from utilities import diagonalize

    >>>A = np.array([[3, 3, 4], [0, 1, 8], [2, 5, 9]])

    >>>P, L, invP = diagonalize(A)

    >>>print(P.dot(L).dot(invP))
    
    [[ 3.00000000e+00  3.00000000e+00  4.00000000e+00]
     [-1.75773426e-15  1.00000000e+00  8.00000000e+00]
     [ 2.00000000e+00  5.00000000e+00  9.00000000e+00]]
    '''
    w, v = np.linalg.eig(A)
    L = np.diag(w)
    return v, L, inv(v)



def ndot(A, n):
    '''
    Compute the n-th power of the given matrix A (n times dot-product of A)
    '''
    A_ = A.copy()
    for i in range(n-1):
        A = A.dot(A_)
    return A



def trace(A):
    '''
    Compute trace of the given matrix A
    '''
    diag = [A[i, i] for i in range(len(A))]
    return sum(diag)



def pdf2cdf(func, a, lower=-100, h=.01, func_arg=[]):
    '''
    Compute CDF (cumulative density fucntion) of the given PDF (probability density fucntion) by integration
    
    Parameter:
        func: function
            PDF of distribution to compute its CDF
        
        a: int, float or array-like
            (Set of) point(s) of density at which you would like to compute CDF
        
        lower: int or float
            Lower bound of CDF
        
        h: float
            parameter of integration (width of increment)
            
        func_arg: list
            Argments of function other than variable
    
    Returns:
        out: int, float or array-like
            Values of culumative distribution function corresponding points a
    
    Example:
    
    >>>import numpy as np
    >>>from utilities import normal_pdf, pdf2cdf

    # transform PDF of normal distribution to CDF
    >>>x = np.linspace(-3, 3, 10).reshape(-1, 1)

    >>>cdf = []
    >>>for x_ in x:
    >>>    cdf.append(pdf2cdf(normal_pdf, x_, func_arg=[0, 1])) # Normal distribution with mean 0 and std 1

    >>>print(cdf)
    [0.0012848430600854487, 0.009513290489145006, 0.04696643116255136, 0.15505190576100758, 
    0.3650466450152271, 0.627409976601133, 0.8376889725074514, 0.9510384504458084, 
    0.9899642789011661, 0.9985821660316603]
    '''
    return integral(func, lower=lower, upper=a, h=h, func_arg=func_arg, algo='ordinary')



def normal_pdf(x, mu=0, sig=1):
    '''
    Compute PDF of normal distribution corresponding random variable x
    
    Parameters:
        x: int, float or array-like
            Data points following a normal distribution
        
        mu: int or float
            Mean of normal distribution
        
        sig: int or float
            Standart deviation (square root of variance) of normal distribution
    
    Returns:
        out: float or array-like
            Probability density of each random variable x
    
    Example:
    
    >>>import numpy as np
    >>>from utilities import normal_pdf

    >>>x = np.linspace(-3, 3, 10)
    >>>normal_pdf(x)
    array([0.00443185, 0.02622189, 0.09947714, 0.24197072, 0.37738323,
       0.37738323, 0.24197072, 0.09947714, 0.02622189, 0.00443185])
    '''
    return 1/np.sqrt(2*np.pi* sig**2) * np.exp(-(x-mu)**2/(2*sig**2))



def normal_cdf(x, mu=0, sigma=1):
    '''
    Compute values of CDF (cumulative density function) of x of normal distribution with mean (mu) and standard
    deviation (sigma)
    '''
    return pdf2cdf(func=normal_pdf, a=x, func_arg=[mu, sigma])



def normal_sampling(size, mu=0, sigma=1):
    '''
    Samples data points following a normal distribution by Box-Muller method
    
    Paramers:
        size: int or array-like
            Size of sampled data points
        
        mu: int, float
            Mean of normal distribution
        
        sigma: int, float
            Standard deviation of normal distribution
        
    Returns:
        out: float or array-like
            Sampled data points
    '''
    x = np.random.random(size)
    y = np.random.random(size)
    z1 = np.sqrt(-2*np.log(x))*np.cos(2*np.pi*y)
    z2 = np.sqrt(-2*np.log(x))*np.sin(2*np.pi*y)
    
    return z1*sigma + mu



def comb_stirling(n, k):
    return stirling_log(n, log=False)/(stirling_log(n-k, log=False)*stirling_log(k, log=False))



def binom_pmf(k, n, p, q=None, stirling=False):
    '''
    Compute PMF (probability mass fucntion) of binamial distribution which represents the probability of getting
    exactly k successes in n independent Bernoulli trials.
    
    Parameters:
        k: int, 0 <= k <= n
            The number of success out of n trials
        n: int
            The number of trials
        p: float, 0 <= p <= 1
            The probability of success
        q: float, 0<= q <= 1, p+q <= 1
            The probability of unsuccess
        stirling: boolean
            If true, Stirling's formula is used to compute factorial. If n and/or k are large, using Stirling's formula leads to
            much less computing time in exchange for relatively subtle approximation error.
    
    Returns
        prob: float
            The probability of getting exactly k successes in n independent Bernoulli trials. 
    
    Example:
    
    >>>import numpy as np
    >>>from utilities import binom_pmf

    >>>binom_pmf(3, 10, .2)
    0.2013265920000001
        
    '''
    if q==None:
        q = 1 - p
    
    if stirling:
        return comb_stirling(n, k)*p**k*q**(n-k)
    
    return comb(n, k)*p**k*q**(n-k)



def binom_sampling(n, p, sample_n=1000):
    '''
    Sample data points following a binomial distribution
    
    Parameters:
        n: int
            The number of trials
        p: float, 0 <= p <= 1
            The probability of success
        sample_n: int
            The number of samples
    
    Returns:
        out: float or array
            Sampled data points
    
    Example:
    >>>import numpy as np
    >>>from utilities import binom_sampling

    >>>binom_sampling(5, .3, 10)
    [0, 2, 3, 2, 1, 3, 1, 1, 2, 1]
    '''
    out = []
    for i in range(sample_n):
        temp = np.random.random(n)
        out.append((temp <= p).sum())
    
    return out



def poisson_pmf(lam, k):
    '''
    Compute probability mass function of Poisson distribution with paramter lambda
    
    Parameters:
        lam: float or int
            Poisson parameter lambda i.e. average number of occurance of the event per unit time or "density"
        k: int
            Number of occurance of the event
    
    Returns:
        p: Probability that the event occurs k times per unit time
    
    Example:
    >>>from utilities import poisson_pmf
    >>>poisson_pmf(1.5, 2)
    0.25102143016698353
    '''
    k = np.array(k)
    if k.shape==():
        return (lam**k/factorial(k) * np.exp(-lam))
    else:
        output = []
        for i in range(len(k)):
            output.append(lam**k[i]/factorial(k[i]) * np.exp(-lam))
        
        return output


def poisson_sampling(n, p, sample_n=1000):
    '''
    Sample data point following a poisson distribution with lambda = n * p
    
    Parameter:
        n: int
            The total number of population per unit time
        p: float, 0 <= p <= 1
            Probability of occurance of event per one element of population
        sample_n: int
            The number of samples
    
    Returns:
        out: int or array-like
            Data points following a poisson distribution
    
    Example:
    >>>from utilities import poisson_sampling
    >>>poisson_sampling(100, .05, sample_n=5)
    [3, 5, 2, 6, 3]
    '''
    k_list = []
    for i in range(sample_n):
        temp = np.random.random(n)
        k_list.append((temp<=p).sum())
    
    return k_list


def geometric_pmf(k, p):
    '''
    Compute probability mass function of geometric distribution with parameter k and p i.e.
    The probability distribution of the number x of Bernoulli trials need to get one success
    
    Parameter:
        k: int, k>0
            The number of trials to get the first success
        p: float, 0 < p <= 1
            The probability of success. The probability of unsuccess will be 1-p
    
    Returns:
        out: int or array-like
            The probability of the given k and p
    
    Exapmle:
    >>>from utilities import geometric_pmf
    >>>geometric_pmf(3, .2)
    0.12800000000000003
    '''
    return p*(1-p)**(k-1)


def geometric_sampling(p, sample_n=1000):
    '''
    Sample data points following a geometric distribution
    
    Parameters:
        p: float, 0< p<=1
            The probability of success.
        sample_n: int
            The number of samples
    
    Returns:
        out: int or array-like
            Data points following a geometric distribution
    
    Example:
    >>>import numpy as np
    >>>from utilities import geometric_sampling

    >>>geometric_sampling(.3, sample_n=10)
    
    [6, 6, 1, 3, 6, 9, 2, 1, 2, 1]
    '''
    out = []
    p = float(p)
    for i in range(sample_n):
        temp = np.random.random(round(10/p))
        out.append((temp<=p).argmax() + 1) 
    
    return out


def uniform_pdf(x, a, b):
    '''
    Returns probability density function (PDF) of uniform distribution in an interval [a, b]
    
    Parameters:
        x: int, float or array-like
            The values at which you would like to compute density
        
        a, b: int or float, a < b
            Minimum and maximum of the interval of uniform distribution
    
    Returns:
        density: float or array-like
            Density of the given x
            
    Examples:
    >>>import numpy as np
    >>>from utilities import uniform_pdf

    >>>x = [.5, 1.5, 2.5]
    
    >>>uniform_pdf(x, 0, 2)
    
    array([0.5, 0.5, 0. ])
    '''
    x = np.array(x)
    return ((x >= a) & (x <= b)).astype(float)/(b-a)


def pwlaw(k, alpha):
    '''
    Returns probability density function of power law distribution: f = A*k^(-alpha)
    
    Parameters:
        k: array-like, k > 0
            constant
        
        alpha: int or float, alpha > 0
            power parameter
    
    Returns:
        f: array-like
            density of the given k
    
    Example:
    >>>import numpy as np
    >>>from utilities import pwlaw

    >>>x = np.array([.5, 1, 1.5])
    >>>alpha = 2.5

    >>>pwlaw(k, alpha)
    array([0.1843235 , 0.0325841 , 0.01182436])
    '''
    output = []
    for i in range(len(k)):
        output.append(k[i] ** (-alpha))
    
    output = np.array(output)
    A = sum((1/output) ** alpha)
    output = output * A
    
    return output


def exp_pdf(x, l):
    '''
    Returns probability density function of exponential distribution with parameter l
    
    Parameters:
        x: int, float or array-like
            Data point
        
        l (lambda): int or float
            Paramtter of exponential distribution
    
    Returns:
        f: intk, float or array-like
            Density corresponding x
    
    Example:
    >>>import numpy as np
    >>>from utilities import exp_pdf

    >>>x = np.array([.2, .5, 1])

    >>>exp_pdf(x, 1)
    
    array([0.81873075, 0.60653066, 0.36787944])
    '''
    x = np.array(x)
    x = np.append(x, 0)
    y = l*np.e**(-l*x)
    y[x<0] = 0
    y = y[:-1]
        
    return y


def inner_func(func1, func2, a=-np.pi, b=np.pi):
    '''
    Computes innter product of the given two functions
    
    Parameters:
        func1, func2: function
            Two functions to compute their inner product
        
        a, b: float or int, a < b
            Minimum and maximum of range of integration
    
    Returns:
        inner_product: float
            Inner product of the given two functions
    
    Example:
    
    >>>import numpy as np
    >>>from utilities import inner_func

    >>>print(inner_func(np.sin, np.cos))
    (3.318621222661022e-07+0j)

    >>>print(inner_func(np.cos, np.cos))
    (3.1424073462299624+0j)
    '''
    
    func = lambda x: conj(func1(x))*func2(x)
    return integral(func=func, lower=a, upper=b, h=.001)
