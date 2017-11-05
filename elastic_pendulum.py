import numpy as np
import matplotlib.pyplot as mpl
#from nose.tools import assert_almost_equal

def crank_nichols(epsilon, T, beta, Theta, Nt):
    """ solver for the elastic pendulm scaled mobel
        using Crank nicholson scheme

        ~~~Not Working!~~~~
    """

    dt = float(T/Nt)            # precision decided by T and Nt
   
    x = np.zeros(Nt+1)           # array of x[n] values
    y = np.zeros(Nt+1)
    v_x = np.zeros(Nt+1)
    v_y = np.zeros(Nt+1)

    t = np.linspace(0, T, Nt+1)  # time mesh

    x[0] = (1+epsilon)*np.sin(Theta)                  # assign initial condition
    y[0] = 1-(1+epsilon)*np.cos(Theta)
    v_x[0] = 0
    v_y[0] = 0
    for i in range(0, Nt):    # n=0,1,...,Nt-1
        L = np.sqrt(x[i]**2+(y[i]-1)**2) # decides forces
        a_x = -(beta/(1-beta))*(1-(beta/L))*x[i] #decided by forces
        a_y = -(beta/(1-beta))*(1-(beta/L))*(y[i]-1)-beta
        v_x[i+1] = (1 - (1-0.5)*a_x*dt)/(1 + 0.5*dt*a_x)*v_x[i]               # integrate once
        v_y[i+1] = (1 - (1-0.5)*a_y*dt)/(1 + 0.5*dt*a_y)*v_y[i]
        x[i+1] = (1 - (1-0.5)*v_x[i]*dt)/(1 + 0.5*dt*v_x[i])*x[i]              # integrate twice
        y[i+1] = (1 - (1-0.5)*v_y[i]*dt)/(1 + 0.5*dt*v_y[i])*y[i]
    return x, y, t

def forward_euler(epsilon, T, beta, Theta, Nt):
    """ solver for the elastic pendulm scaled mobel
        using Forward Euler(actually euler-chromer) scheme
    """
    
    dt = float(T/Nt)            # precision decided by T and Nt
   
    x = np.zeros(Nt+1)           # array of x[n] values
    y = np.zeros(Nt+1)
    v_x = np.zeros(Nt+1)
    v_y = np.zeros(Nt+1)

    t = np.linspace(0, T, Nt+1)  # time mesh

    x[0] = (1+epsilon)*np.sin(Theta)                  # assign initial condition
    y[0] = 1-(1+epsilon)*np.cos(Theta)
    v_x[0] = 0
    v_y[0] = 0
    for i in range(0, Nt):    # n=0,1,...,Nt-1
        L = np.sqrt(x[i]**2+(y[i]-1)**2) # decides forces
        a_x = -(beta/(1-beta))*(1-(beta/L))*x[i] #decided by forces
        a_y = -(beta/(1-beta))*(1-(beta/L))*(y[i]-1)-beta
        v_x[i+1] = v_x[i] + a_x*dt               # integrate once
        v_y[i+1] = v_y[i] + a_y*dt
        x[i+1] = x[i] + v_x[i+1]*dt              # integrate twice
        y[i+1] = y[i] + v_y[i+1]*dt
    return x, y, t

def forward_euler_test(epsilon, T, beta, Theta, Nt):
    """ solver for the elastic pendulm scaled mobel
        using Forward Euler(actually euler-chromer) scheme

        test case implementing the 2'nd order ODE analog scheme.
        (does not calculate v(t), only r(t). )
    """

    
    dt = float(T/Nt)            # precision decided by T and Nt
   
    x = np.zeros(Nt+1)           # array of x[n] values
    y = np.zeros(Nt+1)
    v_x = np.zeros(Nt+1)
    v_y = np.zeros(Nt+1)

    t = np.linspace(0, T, Nt+1)  # time mesh

    x[0] = (1+epsilon)*np.sin(Theta)                  # assign initial condition
    y[0] = 1-(1+epsilon)*np.cos(Theta)
    v_x[0] = 0
    v_y[0] = 0
    L = np.sqrt(x[0]**2+(y[0]-1)**2) # initial L
    a_x = -(beta/(1-beta))*(1-(beta/L))*x[0] # initial a
    a_y = -(beta/(1-beta))*(1-(beta/L))*(y[0]-1)-beta
    x[1] = 2*x[0]-(x[0]-dt*v_x[0])+a_x*dt**2
    y[1] = 2*y[0]-(y[0]-dt*v_y[0])+a_y*dt**2
    for i in range(1, Nt):    # n=0,1,...,Nt-1
        L = np.sqrt(x[i]**2+(y[i]-1)**2) # decides forces
        a_x = -(beta/(1-beta))*(1-(beta/L))*x[i] #decided by forces
        a_y = -(beta/(1-beta))*(1-(beta/L))*(y[i]-1)-beta
        
        x[i+1] = 2*x[i]-x[i-1]+a_x*dt**2              # solve as 2'nd order ODE
        y[i+1] = 2*y[i]-y[i-1]+a_y*dt**2
    return x, y, t


def a_x(x, y, beta):
    L = np.sqrt(x**2+(y-1)**2)
    return -(beta/(1-beta))*(1-(beta/L))*x
def a_y(x, y, beta):
    L = np.sqrt(x**2+(y-1)**2)
    return -(beta/(1-beta))*(1-(beta/L))*(y-1)-beta
    
def r_k4_ks_x(dt,x,y,v_x,v_y,beta):
    """
    computes K's in RK4 method
    """
    x1 = x
    y1 = y
    v_x1 = v_x
    v_y1 = v_y
    a_x1 = a_x(x1, y1, beta)
    a_y1 = a_y(x1, y1, beta)

    x2 = x + 0.5*v_x1*dt
    y2 = y + 0.5*v_y1*dt
    v_x2 = v_x + 0.5*a_x1*dt
    v_y2 = v_y + 0.5*a_y1*dt
    a_x2 = a_x(x2, y2, beta)
    a_y2 = a_y(x2, y2, beta)

    x3 = x + 0.5*v_x2*dt
    y3 = y + 0.5*v_y2*dt
    v_x3 = v_x + 0.5*a_x2*dt
    v_y3 = v_y + 0.5*a_y2*dt
    a_x3 = a_x(x3, y3, beta)
    a_y3 = a_y(x3, y3, beta)

    x4 = x + v_x3*dt
    y4 = y + v_y3*dt
    v_x4 = v_x + a_x3*dt
    v_y4 = v_y + a_y3*dt
    a_x4 = a_x(x4, y4, beta)
    a_y4 = a_y(x4, y4, beta)

    xf = x + (dt/6.0)*(v_x1 + 2*v_x2 + 2*v_x3 + v_x4)
    yf = y + (dt/6.0)*(v_y1 + 2*v_y2 + 2*v_y3 + v_y4)
    v_xf = v_x + (dt/6.0)*(a_x1 + 2*a_x2 + 2*a_x3 + a_x4)
    v_yf = v_y + (dt/6.0)*(a_x1 + 2*a_x2 + 2*a_x3 + a_x4)

    return xf, yf, v_xf, v_yf
    
def runge_kutta4(epsilon, T, beta, Theta, Nt):
    """ solver for the elastic pendulm scaled mobel
        atempt on implementing RK-4 method, but 
        currently not working
    """
    
    dt = float(T/Nt)            # precision decided by T and Nt
   
    x = np.zeros(Nt+1)           # array of x[n] values
    y = np.zeros(Nt+1)
    v_x = np.zeros(Nt+1)
    v_y = np.zeros(Nt+1)

    t = np.linspace(0, T, Nt+1)  # time mesh

    x[0] = (1+epsilon)*np.sin(Theta)                  # assign initial condition
    y[0] = 1-(1+epsilon)*np.cos(Theta)
    v_x[0] = 0
    v_y[0] = 0
    for i in range(0, Nt):    # n=0,1,...,Nt-1
       x[i+1], y[i+1], v_x[i+1], v_y[i+1] = r_k4_ks_x(dt,x[i],y[i],v_x[i],v_y[i],beta)

    return x, y, t

def simulate(
    beta=0.9,                 # dimensionless parameter
    Theta=2,                 # initial angle in degrees
    epsilon=0,                # initial stretch of wire
    num_periods=6,            # simulate for num_periods
    time_steps_per_period=6000, # time step resolution set high for Forward Euler
    plot=False,                # make plots or not
    ):

    Nt = num_periods*time_steps_per_period    # total steps
    T = num_periods*2*np.pi                 # decided by classical pendulum period

    # choose method
    #x, y, t = forward_euler(epsilon, T, beta, Theta, Nt)
    #x, y, t = forward_euler_test(epsilon, T, beta, Theta, Nt)
    #x, y, t = crank_nichols(epsilon, T, beta, Theta, Nt)    #not working
    x, y, t = runge_kutta4(epsilon, T, beta, Theta, Nt)
    theta_t = np.arctan(x/(1-y))             # compute theta array

    if plot == True:     #make plots of x vs y, and theta vs t
        mpl.plot(x,y)
        mpl.gca().set_aspect('equal')
        mpl.xlabel('x(t)')
        mpl.ylabel('y(t)')
        mpl.title('x(t) vs y(t) beta = %2.2f, Theta_0 = %2.2f, epsilon = %2.2f' %(beta,Theta,epsilon))
        mpl.show()
        if Theta < 10:
            mpl.plot(t,theta_t)
            mpl.hold('on')
            mpl.plot(t,np.cos(t))
            mpl.legend(['numerical', 'exact'], loc='upper left')
            mpl.xlabel('t')
            mpl.ylabel('angle(t)')
            mpl.title('Angle of pendulum vs Classical angle')
            mpl.show()
        else:
            mpl.plot(t,theta_t)
            mpl.title('Angle of pendulum for large initial angles')
            mpl.ylabel('angle(t)')
            mpl.xlabel('t')
            mpl.show()
    return x, y, theta_t, t

def zero_stability():
    """
    Test to see if zero values as initial conditions give only zero values 
    as time progresses. Serious problem in numerical scheme if this test fails.
    """
    x, y, theta_t, t = simulate(Theta=0, epsilon=0,time_steps_per_period=60)
    if np.sum(x) == np.sum(y) == np.sum(theta_t) == 0.0:
        print("zero_stability test success!!")
    else:
        print("Warning: The simulation for zero values is not equal zero!")

    #Maybe use nose.tools?
    #assert_almost_equal(np.sum(x), 0.0, delta=1E-10)
    #print 'The simulation for zero values is not equal zero!', np.sum(x)

def vertical_test(beta):
    """
    Test that checks if the pendulum ocillates according to the analytically 
    derived equation Theta(t) = epsilon*np.cos(omega*t) with 
    omega = sqrt(beta/(1-beta). And epsilon is the initial stretch on the pendulum

    The ocillations in this test are only vertical so this is in escence a regular 
    spring problem which is a harmonic oscillator. 
    """

    x, y, theta_t, t = simulate(beta = beta,Theta=0, epsilon=0.2,num_periods=3, time_steps_per_period=6000) # time steps might be to large
    y_exact = -0.2*np.cos(np.sqrt(beta/(1-beta))*t)
    mpl.plot(t,y_exact, 'r-')
    mpl.hold('on')
    mpl.plot(t,y, 'b-')
    mpl.legend(['exact', 'numerical'], loc='upper right')
    mpl.xlabel('t')
    mpl.ylabel('y(t)')
    mpl.title('y(t) vs y_exact(t) for vertical case')
    mpl.show()

    err = float(sum(abs(y-y_exact)))  # error is a sum of differences of y(i) - y_exact(i)
    if err > 2: #2 is accepded accumulated error. maybe to large?
        print("Too large error in the vertical test! Accumulated error = %2.5f" % err )
    else:
        print("vertical test success! Accumulated error = %3.5f" % err )
    
def demo(beta, Theta):#this is not necessary but asked for in exercise.
    """demo function that calls simulate() with some variables pre defined. """
    simulate(beta = beta, Theta = Theta, epsilon=0,time_steps_per_period=600, plot=True)
    
    
if __name__ == '__main__':
    zero_stability()
    #simulate(Theta=1, epsilon=0,time_steps_per_period=6000, plot=True)
    vertical_test(0.5)
    demo(beta = 0.5, Theta = 1)
