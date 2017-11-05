
#!/usr/bin/env python
"""
INF5620 2016, Exercise 13

1D wave equation with homogeneous Neumann conditions::
  u, x, t, cpu = solver(q,f,I, L,T,Nx,V,u_exact=None,user_action=None, \
version='scalar', method = 'sum_approx')

Function solver solves the wave equation
   u_tt = c**2*u_xx + f(x,t) on
(0,L) with du/dn=0 on x=0 and x = L.

c**2 = q(x) is a variable velocity coefficient

The program is written with harcoded calls to function 
compute_convergence_rate_A/B that calls solver() to solve the wave equation.
These calls are foung in __main__

compute_convergence_rate_A/B contains information on specific q(x) functions
and solver takes the argument "method" to specify how the neumann boundary
conditions are imposed on the system.

The goal of this exercize is to analyze different scemes for imposing neumann 
conditions, and how they affect the convergence rates

"""
#import time, glob, shutil, os
import numpy as np
import matplotlib.pyplot as plt


def solver(q,f,I, L,T,Nx,V,u_exact=None,user_action=None, version='scalar', method = 'sum_approx'): #implement vectorized?
    """
    Solve u_tt=c^2*u_xx + f on (0,L)x(0,T].
    u(0,t)=U_0(t) or du/dn=0 (U_0=None), u(L,t)=U_L(t) or du/dn=0 (u_L=None).
    """
    x = np.linspace(0, L, Nx+1)       # Mesh points in space
    dx = L/float(Nx)
    dt = dx*0.9/max(map(q,x))**0.5    #stability criteria
    Nt = int(round(T/dt))
    t = np.linspace(0, Nt*dt, Nt+1)   # Mesh points in time
    C2= (dt/dx)**2; dt2 = dt*dt            # Help variables in the scheme


    # Wrap user-given f, I, V, U_0, U_L if None or 0
    if f is None or f == 0:
        f = (lambda x, t: 0) if version == 'scalar' else \
            lambda x, t: np.zeros(x.shape)
    if I is None or I == 0:
        I = (lambda x: 0) if version == 'scalar' else \
            lambda x: np.zeros(x.shape)
    if V is None or V == 0:
        V = (lambda x: 0) if version == 'scalar' else \
            lambda x: np.zeros(x.shape)

    u   = np.zeros(Nx+1)   # Solution array at new time level
    u_1 = np.zeros(Nx+1)   # Solution at 1 time level back
    u_2 = np.zeros(Nx+1)   # Solution at 2 time levels back

    Ix = range(0, Nx+1)
    It = range(0, Nt+1)

    if method == 'original_approx':
        q1 = np.array(map(q,x)) # is q a function?
        def approx(i, side):
            if side == 0:
                ip = i+1
                im = ip  # i-1 -> i+1
                return 0.5*(q1[i] + q1[ip])*(u_1[ip] - u_1[i]) - 0.5*(q1[i] + q1[im])*(u_1[i] - u_1[im])
            if side == 1:
                im = i-1
                ip = im  # i+1 -> i-1
                return 0.5*(q1[i] + q1[ip])*(u_1[ip] - u_1[i]) - 0.5*(q1[i] + q1[im])*(u_1[i] - u_1[im])
    if method == 'sum_approx':
        q1 = np.array(map(q,x)) # is q a function?
        def approx(i, side):
            if side == 0:
                ip = i+1
                im = ip  # i-1 -> i+1
                return 0.5*(q1[i] + q1[ip])*(u_1[ip] - u_1[i]) - 0.5*(q1[i] + q1[im])*(u_1[i] - u_1[im])
            if side == 1:
                return 2.0*q1[i]*(u_1[i-1] - u_1[i])
    if method == 'half_approx':
        q2 = np.array(map(q,x-(dx/2.))) # is q a function?
        q1 = np.array(map(q,x)) # is q a function?
        def approx(i, side):
            if side == 0:
                ip = i+1
                im = ip  # i-1 -> i+1
                return 0.5*(q1[i] + q1[i+1])*(u_1[i+1] - u_1[i]) - 0.5*(q1[i] + q1[im])*(u_1[i] - u_1[im])
            elif side == 1:
                return 2.0*q2[i]*(u_1[i-1] - u_1[i])
    if method == 'one_sided_approx':
        q1 = np.array(map(q,x-(dx/2.))) # is q a function?
        def approx(i, side):
            if side == 1:
                return -0.5*(q1[i]+q1[i-1])*(u_1[i] - u_1[i-1])
            elif side == 0:
                return 0.5*(q1[i]+q1[i-1])*(u_1[i+1] - u_1[i])
    if method == 'shifted_approx':
        q1 = np.array(map(q,x+(dx/2.))) # is q a function? shift q in scheme
        def approx(i, side):
            if side == 0:
                ip = i+1
                im = ip  # i-1 -> i+1
                return 0.5*(q1[i] + q1[ip])*(u_1[ip] - u_1[i]) - 0.5*(q1[i] + q1[im])*(u_1[i] - u_1[im])
            elif side == 1:
                return q1[i-1]*(u_1[i-1] - u_1[i])
        

    import time;  t0 = time.clock()  # CPU time measurement

    # Load initial condition into u_1
    for i in Ix:
        u_1[i] = I(x[i])

    # Special formula for the first step

    #compute internal points
    for i in Ix[1:-1]:
        u[i] = u_1[i] + dt*V(x[i]) + \
        0.5*C2*(0.5*(q1[i] + q1[i+1])*(u_1[i+1] - u_1[i]) - \
                0.5*(q1[i] + q1[i-1])*(u_1[i] - u_1[i-1])) + \
        0.5*dt2*f(x[i], t[0])

    #compute boundary spatial points with neumann cond. 
    i = Ix[0]
    # Set boundary values (x=0: i-1 -> i+1 since u[i-1]=u[i+1]
    # when du/dn = 0, on x=L: i+1 -> i-1 since u[i+1]=u[i-1])
    u[i] = u_1[i] + dt*V(x[i]) + 0.5*C2*(approx(i, side = 0)) + 0.5*dt2*f(x[i], t[0])

    i = Ix[-1]
    u[i] = u_1[i] + dt*V(x[i]) + 0.5*C2*(approx(i, side = 1)) + 0.5*dt2*f(x[i], t[0])

    # Update data structures for next step
    #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
    u_2, u_1, u = u_1, u, u_2

    for n in It[1:-1]:
        # Update all inner points
        for i in Ix[1:-1]:
            u[i] = - u_2[i] + 2*u_1[i] + \
                    C2*(0.5*(q1[i] + q1[i+1])*(u_1[i+1] - u_1[i])  - \
                        0.5*(q1[i] + q1[i-1])*(u_1[i] - u_1[i-1])) + \
            dt2*f(x[i], t[n])

        # Insert boundary conditions at each timestep
        i = Ix[0]
        # Set boundary values
        # x=0: i-1 -> i+1 since u[i-1]=u[i+1] when du/dn=0
        # x=L: i+1 -> i-1 since u[i+1]=u[i-1] when du/dn=0
        u[i] = - u_2[i] + 2*u_1[i] + C2*(approx(i, side = 0)) + dt2*f(x[i], t[n])

        i = Ix[-1]
        u[i] = - u_2[i] + 2*u_1[i] + C2*(approx(i, side = 1)) + dt2*f(x[i], t[n])

        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2

    # Important to correct the mathematically wrong u=u_2 above
    # before returning u
    u = u_1
    cpu_time = time.clock() - t0
    #print "cpu time = %5.5f" % cpu_time          # print time usage

    if u_exact is not None:
        e = abs(u_2-u_exact(x,t[-1]))	#error of method
        e = sum(e**2)*dx

    return u, x, t, cpu_time, e, dt

def compute_convergence_rate_A(method):
    """
    compute convergence rate in a simple test case for q = 1+(x-L/2)**4
    uses sympy to fit f, and uses that f in an analytic solution.
    """
    import math
    import sympy
    from sympy import cos
    from sympy import sin
    from sympy.utilities.lambdify import lambdify

    w,x, L ,t= sympy.symbols("w x L t")
    pi = math.pi	

    u = lambda x,t : cos(w*t)*cos(pi*x/L)
    q = lambda x : 1 + ( x -L/2.0)**4

    def source_term(u,q):
        return sympy.simplify(u(x,t).diff(t, t) - (u(x,t).diff(x)*q(x)).diff(x) )

    w=1
    L=2
    T=2
    Nx=30

    f =  sympy.lambdify((x,t), source_term(u,q)  ,'numpy')
    q = sympy.lambdify(x,q(x) ,'numpy')
    I = sympy.lambdify(x, u(x,t).subs(t,0),'numpy')
    exact_solution = sympy.lambdify((x,t), u(x,t),'numpy')

    def convegence_rates(h,E):
        from math import log
        return [log(E[i]/E[i-1])/log(h[i]/h[i-1]) for i in range(1, len(h))]

    #if u==None: raise ValueError
    E = []
    h=[]
    for nx in [ Nx*i for i in range(1,10)]:
	
        u_num, x, t, cpu_time, e, dt =solver(q,f,I, L,T,nx,V=0,u_exact=exact_solution,user_action=None, version='scalar', method = method) #hardcoded method
        E.append(e)
        h.append(dt)
        #print "done"
	
	
    r =  convegence_rates(h,E)
    return r

def compute_convergence_rate_B():
    """
    compute convergence rate in a simple test case for 1 + cos(pi*x/L)
    uses sympy to fit f, and uses that f in an analytic solution.
    """
    import math
    import sympy
    from sympy import cos
    from sympy import sin
    from sympy.utilities.lambdify import lambdify

    w,x, L ,t= sympy.symbols("w x L t")
    pi = math.pi	

    u = lambda x,t : cos(w*t)*cos(pi*x/L)
    q = lambda x : 1 + cos(pi*x/L) # new q for task B

    def source_term(u,q):
        return sympy.simplify(u(x,t).diff(t, t) - (u(x,t).diff(x)*q(x)).diff(x) )

    w=1
    L=2
    T=2
    Nx=30

    f = sympy.lambdify((x,t), source_term(u,q)  ,'numpy')
    q = sympy.lambdify(x,q(x) ,'numpy')
    I = sympy.lambdify(x, u(x,t).subs(t,0),'numpy')
    exact_solution = sympy.lambdify((x,t), u(x,t),'numpy')

    def convegence_rates(h,E):
        from math import log
        return [log(E[i]/E[i-1])/log(h[i]/h[i-1]) for i in range(1, len(h))]

    #if u==None: raise ValueError
    E = []
    h = []
    for nx in [ Nx*i for i in range(1,10)]:
	
        u_num, x, t, cpu_time, e, dt =solver(q,f,I, L,T,nx,V=0,u_exact=exact_solution,user_action=None, version='scalar',  method = 'half_approx')
        E.append(e)
        h.append(dt)
        #print "done"
	
	
    r =  convegence_rates(h,E)
    return r

def test_constant():
    """
    simple test with constant solution
    """
    import math
    import numpy as np


    exact_solution = lambda x : 2.345
    q = lambda x : 4.245
    f =  lambda x,t : 0
    I = lambda x : 2.345
    V = 0
    L = 2
    T=1
    Nx = 50

    
    def assert_if_error(u,x,t):
        e = abs(u-exact_solution(x)).max()
        np.testing.assert_almost_equal(e, 0, decimal=13)

    u, x, t, cpu_time, e, dt = solver(q,f,I, L,T,Nx,V,u_exact=None,user_action=None, version='scalar')
    assert_if_error(u,x,t)
    #plt.plot(x,u)
    #plt.show()

if __name__ == '__main__':
    #pass
    V = 0 #testing neumann conditions, nn V
    conv_rate = compute_convergence_rate_A(method='original_approx')
    print "convergence rate original scheme = %5.5f" % conv_rate[-1]

    conv_rate = compute_convergence_rate_A(method='sum_approx')
    print "convergence rate task A = %5.5f" % conv_rate[-1]

    conv_rate = compute_convergence_rate_B()
    print "convergence rate task B = %5.5f" % conv_rate[-1]

    conv_rate = compute_convergence_rate_A(method='one_sided_approx')
    print "convergence rate task C using q from A = %5.5f" % conv_rate[-1]

    conv_rate = compute_convergence_rate_A(method='shifted_approx')
    print "convergence rate task D using q from A = %5.5f" % conv_rate[-1]


    #test_constant()

