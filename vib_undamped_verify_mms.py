import sympy as sym
import matplotlib.pyplot as plt
import numpy as np
from nose.tools import assert_almost_equal

V, t, I, w, dt, b, a= sym.symbols('V t I w dt b a')  # global symbols
f = None  # global variable for the source term in the ODE

def ode_source_term(u):
    """Return the terms in the ODE that the source term
    must balance, here u'' + w**2*u.
    u is symbolic Python function of t."""
    return sym.diff(u(t), t, t) + w**2*u(t)

def residual_discrete_eq(u):
    """Return the residual of the discrete eq. with u inserted."""
    R = DtDt(u,dt)+w**2*u(t)-f 
    return sym.simplify(R)

def residual_discrete_eq_step1(u):
    """Return the residual of the discrete eq. at the first
    step with u inserted."""
    R = sym.Rational(1,2)*(ode_source_term(u)-w**2*u(t))*dt**2 + u(t) + dt*V -u(t+dt)
    R = R.subs(t,0) # must set " t = 0 "
    #--- nope! # in the scheme step 0 to 1 is no different than step i to i+1, so we can use t to t+dt (?)
    return sym.simplify(R)

def DtDt(u, dt):
    """Return 2nd-order finite difference for u_tt.
    u is a symbolic Python function of t.
    """
    return (u(t+dt)-2*u(t)+u(t-dt))/dt**2

def solver(I, w, dt, T, V, f):
    """
    Solve u'' + w**2*u = 0 for t in (0,T], u(0)=I and u'(0)=0,
    by a central finite difference method with time step dt.
    """
    dt = float(dt)
    Nt = int(round(T/dt)) # 100000
    u = np.zeros(Nt+1)
    t = np.linspace(0, Nt*dt, Nt+1)

    u[0] = I
    u[1] = u[0] + dt*V + 0.5*(f(t[0]) - w**2*u[0])*dt**2#compute first step by 1'st order difference
    for n in range(1, Nt):
        u[n+1] = (f(t[n])-w**2*u[n])*dt**2 + 2*u[n]-u[n-1]
    return u, t

def visualize(u, t, I, w, b):
    plt.plot(t, u, 'r--o')
    #t_fine = np.linspace(0, t[-1], 1001)  # very fine mesh for u_e
    u_e = f_numerical(b, V, I, t) #lamda function
    plt.hold('on')
    plt.plot(t, u_e(t), 'b-')
    plt.legend(['numerical', 'exact'], loc='upper left')
    plt.xlabel('t')
    plt.ylabel('u')
    dt = t[1] - t[0]
    plt.title('dt=%g' % dt)
    umin = 1.2*u.min();  umax = 1.2*u.max()
    plt.axis([t[0], t[-1], umin, umax])
    plt.show()
    #plt.savefig('tmp1.png');  plt.savefig('tmp1.pdf')


def main(u):
    """
    Given some chosen solution u (as a function of t, implemented
    as a Python function), use the method of manufactured solutions
    to compute the source term f, and check if u also solves
    the discrete equations.
    """
    print '=== Testing exact solution: %s ===' % u
    print "Initial conditions u(0)=%s, u'(0)=%s:" \
          % (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0))

    # Method of manufactured solution requires fitting f
    global f  # source term in the ODE
    f = sym.simplify(ode_source_term(u))

    # Residual in discrete equations (should be 0)
    print 'residual step1:', residual_discrete_eq_step1(u)
    print 'residual:', residual_discrete_eq(u)
    

def linear():
    main(lambda t: V*t + I)

def quadratic():
    main(lambda t: b*t**2 + V*t + I)

def cubic():
    main(lambda t: a*t**3 + b*t**2 + V*t + I) # gives -a*dt**3 on step 1

def f_numerical(b, V, I, t):# got "TypeError: 'Add' object is not callable" without lamda? 
   return lambda t: b*t**2 + V*t + I

def u_exact(t, I, w, V):# temporary function, delete later
    #return I*np.cos(w*t)
    return 1 + (V*t+I)*w**2

def error_plot():
    """
    function that plots the deviation of exact numerical and the numerical 
    scheme for a range of dt's.
    """

    global b,V,I,w,dt,f,t
    n=100
    b = 2.2
    V = 2
    I = 1
    w = 2.*np.pi
    dt_array = np.linspace(0.0005,0.3,n) # store dt values
    eps_array = np.zeros(n) #store deviation
    num_periods = 5
    P = 2.*np.pi/w    #  one period
    T = P*num_periods

    f = ode_source_term(f_numerical(b, V, I, t))  
    f_ = sym.lambdify(t,f)

    for i in range(0,n):
        u_num, t_num = solver(I=I, w=w, dt=dt_array[i], T=T, V=V, f=f_)

        u_analytic = f_numerical(b, V, I, t_num)
        eps_array[i] = np.abs(u_num - u_analytic(t_num)).max()

    plt.plot(dt_array,eps_array)
    plt.xlabel('dt')
    plt.ylabel('deviation')
    plt.title('deviation between numerical and analytical')
    umin = 1.2*eps_array.min();  umax = 1.2*eps_array.max()
    plt.axis([dt_array[0], dt_array[-1], umin, umax])
    plt.show()

def vizualize_accumulation():
    """
    function that plots the accumulation of error with inncreasing T and fixed dt.
    """

    global b,V,I,w,dt,f,t
    n = 10
    b = 2.2
    V = 2
    I = 1
    w = 2.*np.pi
    dt = 0.05
    eps_array = np.zeros(n) #store deviation
    num_periods = 5
    P = 2.*np.pi/w    #  one period
    T = np.linspace(1,P*num_periods,n)

    f = ode_source_term(f_numerical(b, V, I, t))  
    f_ = sym.lambdify(t,f)

    for i in range(0,n):
        u_num, t_num = solver(I=I, w=w, dt=dt, T=T[i], V=V, f=f_)

        u_analytic = f_numerical(b, V, I, t_num)
        eps_array[i] = np.abs(u_num - u_analytic(t_num)).max()

    plt.plot(T,eps_array)
    plt.xlabel('dt')
    plt.ylabel('deviation')
    plt.title('Accumulation of error with increase in T')
    umin = 1.2*eps_array.min();  umax = 1.2*eps_array.max()
    plt.axis([T[0], T[-1], umin, umax])
    plt.show()
    

def nose_test():
    """
    function that tests if the difference between the numerical annalytical
    quadratic solution and the fitted solution of u is not too large.   
    """

    global b,V,I,w,dt,f,t
    b = 2.2
    V = 2
    I = 1
    w = 2.*np.pi
    dt = 0.05
    num_periods = 5
    P = 2.*np.pi/w    #  one period
    T = P*num_periods

    f = ode_source_term(f_numerical(b, V, I, t))  
    f_ = sym.lambdify(t,f)

    u_num, t_num = solver(I, w, dt, T, V, f=f_)

    u_analytic = f_numerical(b, V, I, t_num)
    eps = np.abs(u_num - u_analytic(t_num)).max()
    print  "machine precicion = %.2e" %(np.finfo(float).eps) # get machine precision
    assert_almost_equal(eps, 0, delta=1E-10)
    print 'Error in computing a quadratic solution:', eps
    
    visualize(u_num, t_num, I, w, b)

if __name__ == '__main__':
    #cubic()
    #quadratic()
    linear()
    #error_plot() #hardcode Nt in solver() to use this
    nose_test()
    vizualize_accumulation()
    





