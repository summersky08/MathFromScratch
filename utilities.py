
# coding: utf-8

# In[2]:

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:

from math import floor, ceil


# In[ ]:

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


# In[ ]:

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


# In[ ]:

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
def rootfinding3(func, x_min=False, x_max=False, max_iter=100, func_arg=[]):
    
    a,b = np.random.random(2)
    
    for i in range(max_iter):
        f_a, f_b = func(*([a]+func_arg)), func(*([b]+func_arg))
        
        num = a - b
        denom = f_a - f_b
        if denom == 0:
            return a
        
        a, b = a - f_a*num / denom, a
        
    return a


# In[ ]:

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


# In[ ]:

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


# In[ ]:

def factorial(k):
    '''
    Returns k factorial: k! = 1 * 2 * ... * k
    '''
    if k == 1 or k == 0:
        return 1
    else:
        return k*factorial(k-1)


# In[3]:

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
        return (differentiate(func, a+h, h=h, order=order-1, func_arg=func_arg)                     - differentiate(func, a, h=h, order=order-1, func_arg=func_arg))/h


# In[9]:

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


# In[4]:

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


# In[15]:

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
        J[i, :] = grad(func_list[i], init, variables=variables, h=h)
    if determinant and J.shape[0] == J.shape[1]:
        return det(J)
    else:
        return J


# In[ ]:

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


# In[3]:

def integral1(func, lower, upper, h, func_arg=[]):
    n = int((upper - lower) // h)
    sum_ = 0
    for i in range(n):
        sum_ += func(*([lower + i*h]+func_arg)) * h
    
    return sum_


# In[4]:

def integral2(func, lower, upper, h, func_arg=[]):
    n = int((upper - lower) // h)
    h_ = (upper - lower) - n*h
    edges = np.array([lower + i*h for i in range(n)] + [upper])
    edges1, edges2 = edges[:-1], edges[1:]
    main = ((func(*([edges2[:-1]]+func_arg)) + func(*([edges1[:-1]]+func_arg)))*h*.5).sum()
    margin = (func(*([edges2[-1]]+func_arg)) + func(*([edges1[-1]]+func_arg)))*h_*.5
    
    return main + margin


# In[26]:

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


# In[6]:

def integral4(func, lower, upper, h, func_arg=[]):
    n = np.floor((upper - lower) // h).astype(int)
    # 面倒なのでマージンは計算しない
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


# In[7]:

def integral_MonteCarlo(func, lower, upper, h=False, n_sample=1e+4, rv='uniform', func_arg=[]):
    rvs = np.random.random(int(n_sample))*(upper-lower) + lower
    return (upper - lower)*func(*([rvs]+func_arg)).mean()


# In[8]:

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


# In[ ]:

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


# In[24]:

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


# In[ ]:

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


# In[ ]:

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

