#!/usr/bin/env python
"""
2D wave equation solved by finite differences::
  dt, cpu_time, u = solver(I, V, f, q, Lx, Ly, Nx, Ny, dt, T,b,
                    user_action=None, version='scalar',
                    dt_safety_factor=1, bc = None, animate = True):


Solve the 2D wave equation u_tt + b*ut = u_xx + u_yy + f(x,t) on (0,L) 
can specify u=c on the boundary and initial condition du/dt=0.
the system has reflecting boundarys du/dn = 0 if the boundary condition
u = None is set.
Nx and Ny are the total number of mesh cells in the x and y
directions. The mesh points are numbered as (0,0), (1,0), (2,0),
..., (Nx,0), (0,1), (1,1), ..., (Nx, Ny).
dt is the time step. If dt<=0, an optimal time step is used.
T is the stop time for the simulation.
I, V, f are functions: I(x,y), V(x,y), f(x,y,t). V and f
can be specified as None or 0, resulting in V=0 and f=0.
user_action: function of (u, x, y, t, n) called at each time
level (x and y are one-dimensional coordinate vectors).
This function allows the calling code to plot the solution,
compute errors, etc.

The vectorized version is not implemented and working yet.
"""
import time, sys
from scitools.std import *

def solver(I, V, f, q, Lx, Ly, Nx, Ny, dt, T,b,
            user_action=None, version='scalar',
            dt_safety_factor=1, bc = None, animate = True):

    from mpl_toolkits.mplot3d import axes3d
    import matplotlib.pyplot as plt
    import numpy as np



    if version == 'vectorized':
        advance = advance_vectorized
    elif version == 'scalar':
        advance = advance_scalar
    else:
        print 'version error, not scalar or vectorized!'
        sys.exit(1)

    # Allow f and V to be None or 0
    if f is None or f == 0:
        f = (lambda x, y, t: 0) if version == 'scalar' else \
            lambda x, y, t: zeros((x.shape[0], y.shape[1]))
        # or simpler: x*y*0
    if V is None or V == 0:
        V = (lambda x, y: 0) if version == 'scalar' else \
            lambda x, y: zeros((x.shape[0], y.shape[1]))
    if q is None or q == 0:
        q = (lambda x, y: 0) if version == 'scalar' else \
            lambda x, y: zeros((x.shape[0], y.shape[1]))


    x = linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = linspace(0, Ly, Ny+1)  # mesh points in y dir
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    xv = x[:,newaxis]          # for vectorized function evaluations
    yv = y[newaxis,:]
    try:
        max_q = max(q(x,y))
    except:
        max_q = q(x,y)
    #print sqrt(max_q)
    stability_limit = (1/sqrt(float(max_q)))*(1/sqrt(1/dx**2 + 1/dy**2))
    if dt <= 0:                # max time step?
        safety_factor = -dt    # use negative dt as safety factor
        dt = 0.5*safety_factor*stability_limit
    elif dt > stability_limit:
        print 'error: dt=%g exceeds the stability limit %g' % \
              (dt, stability_limit)
    Nt = int(round(T/float(dt)))
    t = linspace(0, Nt*dt, Nt+1)    # mesh points in time
    Cx2 = (dt/dx)**2;  Cy2 = (dt/dy)**2    # help variables
    dt2 = dt**2
    print dx
    print dt

    


    order = 'Fortran' if version == 'f77' else 'C'
    u   = zeros((Nx+1,Ny+1), order=order)   # solution array
    u_1 = zeros((Nx+1,Ny+1), order=order)   # solution at t-dt
    u_2 = zeros((Nx+1,Ny+1), order=order)   # solution at t-2*dt
    f_a = zeros((Nx+1,Ny+1), order=order)   # for compiled loops

    Ix = range(0, u.shape[0])
    Iy = range(0, u.shape[1])
    It = range(0, t.shape[0])

    import time; t0 = time.clock()          # for measuring CPU time

    # Load initial condition into u_1
    if version == 'scalar':
        for i in Ix:
            for j in Iy:
                u_1[i,j] = I(x[i], y[j])
    else: # use vectorized version
        u_1[:,:] = I(xv, yv)

    if user_action is not None:
        user_action(u_1, x, xv, y, yv, t, 0)

    # Special formula for first time step
    n = 0
    # First step requires a special formula, use either the scalar
    # or vectorized version (the impact of more efficient loops than
    # in advance_vectorized is small as this is only one step)
    if version == 'scalar':
        u = advance_scalar(
            u, u_1, u_2, f, x, y, t, n,
            Cx2, Cy2, dt2,q,b, V, step1=True)

    else:
        f_a[:,:] = f(xv, yv, t[n])  # precompute, size as u
        V_a = V(xv, yv)
        u = advance_vectorized(
            u, u_1, u_2, f_a,
            Cx2, Cy2, dt2, V=V_a, step1=True)

    if user_action is not None:
        user_action(u, x, xv, y, yv, t, 1)

    # Update data structures for next step
    #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
    u_2, u_1, u = u_1, u, u_2

    if animate:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            wframe = None

    for n in It[1:-1]:
        if version == 'scalar':
            # use f(x,y,t) function
            u = advance(u, u_1, u_2, f, x, y, t, n, Cx2, Cy2, dt2,q,b, bc=bc)
        else:
            f_a[:,:] = f(xv, yv, t[n])  # precompute, size as u
            u = advance(u, u_1, u_2, f_a, Cx2, Cy2, dt2)
        if animate:

            xs = x
            ys = y
            X, Y = np.meshgrid(xs, ys)
            Z = u

            oldcol = wframe

            wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)

            # Remove old line collection before drawing
            if oldcol is not None:
                ax.collections.remove(oldcol)

            plt.pause(.001)



        # removed fortran version

        if user_action is not None:
            if user_action(u, x, xv, y, yv, t, n+1):
                break

        # Update data structures for next step
        #u_2[:] = u_1;  u_1[:] = u  # safe, but slower
        u_2, u_1, u = u_1, u, u_2

    # Important to set u = u_1 if u is to be returned!
    t1 = time.clock()
    # dt might be computed in this function so return the value
    return dt, t1 - t0,u_1



def advance_scalar(u, u_1, u_2, f, x, y, t, n, Cx2, Cy2, dt2,q,b,
                   V=None, step1=False, bc = None):

    Ix = range(0, u.shape[0]);  Iy = range(0, u.shape[1])

    dt = sqrt(dt2)
    #b = 1.
    damping_c = 0.5*b*dt 

    Nx = len(u[0])-1
    Ny = len(u[1])-1
    #q = lambda x, y : 1.*x/Nx # hardcode to zero while testing
    q1 = zeros((Nx+1,Ny+1))
    for i in Ix:
        for j in Iy:
            q1[i,j] = q(i,j) #np.array(map(q,x,y))
    


    def approx_x(i,j, side):
        if side == 0:
            ip = i+1
            im = ip  # i-1 -> i+1
            return 0.5*(q1[i,j] + q1[i,j])*(u_1[ip,j] - u_1[i,j]) -\
                0.5*(q1[i,j] + q1[im,j])*(u_1[i,j] - u_1[im,j])
        if side == 1:
            im = i-1
            ip = im  # i+1 -> i-1
            return 0.5*(q1[i,j] + q1[ip,j])*(u_1[ip,j] - u_1[i,j]) - \
                0.5*(q1[i,j] + q1[im,j])*(u_1[i,j] - u_1[im,j])
    def approx_y(i,j, side):
        if side == 0:
            jp = j+1
            jm = jp  # i-1 -> i+1
            return 0.5*(q1[i,j] + q1[i,jp])*(u_1[i,jp] - u_1[i,j]) -\
                0.5*(q1[i,j] + q1[i,jm])*(u_1[i,j] - u_1[i,jm])
        if side == 1:
            jm = j-1
            jp = jm  # i+1 -> i-1
            return 0.5*(q1[i,j] + q1[i,jp])*(u_1[i,jp] - u_1[i,j]) - \
                0.5*(q1[i,j] + q1[i,jm])*(u_1[i,j] - u_1[i,jm])

    
    if step1:
        dt = sqrt(dt2)  # save
        for i in Ix[1:-1]:
            for j in Iy[1:-1]:#initial internal points
                u[i,j] = u_1[i,j] + dt*V(x[i],y[i]) + \
                        0.5*Cx2*(0.5*(q1[i,j] + q1[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - \
                        0.5*(q1[i,j] + q1[i-1,j])*(u_1[i,j] - u_1[i-1,j])) + \
                        0.5*Cy2*(0.5*(q1[i,j] + q1[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - \
                        0.5*(q1[i,j] + q1[i,j-1])*(u_1[i,j] - u_1[i,j-1])) + \
                0.5*dt2*f(x[i], y[j], t[0])
        i = Ix[0]
        for j in Iy[1:-1]: #neumann w. variable coefficient special in boundary
            u[i,j] = u_1[i,j] + dt*V(x[i],y[i]) + \
                    0.5*Cx2*(approx_x(i,j, side = 0)) + \
                    0.5*Cy2*(0.5*(q1[i,j] + q1[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - \
                    0.5*(q1[i,j] + q1[i,j-1])*(u_1[i,j] - u_1[i,j-1])) + \
            0.5*dt2*f(x[i], y[j], t[0]) #approx_x must have running j
        j = Iy[0]
        for i in Ix[1:-1]: #neumann w. variable coefficient special in boundary
            u[i,j] = u_1[i,j] + dt*V(x[i],y[i]) + \
                    0.5*Cx2*(0.5*(q1[i,j] + q1[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - \
                    0.5*(q1[i,j] + q1[i-1,j])*(u_1[i,j] - u_1[i-1,j])) + \
                    0.5*Cy2*(approx_y(i,j, side = 0)) + \
            0.5*dt2*f(x[i], y[j], t[0]) #approx_y must have running i
        i = Ix[-1]
        for j in Iy[1:-1]: #neumann w. variable coefficient special in boundary
            u[i,j] = u_1[i,j] + dt*V(x[i],y[i]) + \
                0.5*Cx2*(approx_x(i,j, side = 1)) + \
                0.5*Cy2*(0.5*(q1[i,j] + q1[i,j+1])*(u_1[i,j+1] - u_1[i,j]) - \
                0.5*(q1[i,j] + q1[i,j-1])*(u_1[i,j] - u_1[i,j-1])) + \
            0.5*dt2*f(x[i], y[j], t[0]) #approx_x must have running j
        j = Iy[-1]
        for i in Ix[1:-1]: #neumann w. variable coefficient special in boundary
            u[i,j] = u_1[i,j] + dt*V(x[i],y[i]) + \
                0.5*Cx2*(0.5*(q1[i,j] + q1[i+1,j])*(u_1[i+1,j] - u_1[i,j]) - \
                0.5*(q1[i,j] + q1[i-1,j])*(u_1[i,j] - u_1[i-1,j])) + \
                0.5*Cy2*(approx_y(i,j, side = 1)) + \
            0.5*dt2*f(x[i], y[j], t[0]) #approx_y must have running i
        #corners
        u[0,0] = u_1[0,0] + dt*V(x[0],y[0]) + \
                0.5*Cx2*(approx_x(0,0, side = 0)) + \
                0.5*Cy2*(approx_y(0,0, side = 0)) + \
            0.5*dt2*f(x[0], y[0], t[0])
        u[0,Ny] = u_1[0,Ny] + dt*V(x[0],y[Ny]) + \
                0.5*Cx2*(approx_x(0,Ny, side = 0)) + \
                0.5*Cy2*(approx_y(0,Ny, side = 1)) + \
            0.5*dt2*f(x[0], y[Ny], t[0])
        u[Nx,0] = u_1[Nx,0] + dt*V(x[Nx],y[0]) + \
                0.5*Cx2*(approx_x(Nx,0, side = 1)) + \
                0.5*Cy2*(approx_y(Nx,0, side = 0)) + \
            0.5*dt2*f(x[Nx], y[0], t[0])
        u[Nx,Ny] = u_1[Nx,Ny] + dt*V(x[Nx],y[Ny]) + \
                0.5*Cx2*(approx_x(Nx,Ny, side = 1)) + \
                0.5*Cy2*(approx_y(Nx,Ny, side = 1)) + \
            0.5*dt2*f(x[Nx], y[Ny], t[0])


    else:
        for i in Ix[1:-1]:
            for j in Iy[1:-1]:
                u[i,j] = (1./(1+damping_c))*( (damping_c-1)*u_2[i,j] + 2*u_1[i,j] + \
                    Cx2*(0.5*(q1[i,j] + q1[i+1,j])*(u_1[i+1,j] - u_1[i,j])  - \
                    0.5*(q1[i,j] + q1[i-1,j])*(u_1[i,j] - u_1[i-1,j])) + \
                    Cy2*(0.5*(q1[i,j] + q1[i,j+1])*(u_1[i,j+1] - u_1[i,j])  - \
                    0.5*(q1[i,j] + q1[i,j-1])*(u_1[i,j] - u_1[i,j-1])) + \
                dt2*f(x[i], y[j], t[n]) )

        # Insert boundary conditions at each timestep
        i = Ix[0]
        for j in Iy[1:-1]:
            u[i,j] = -u_2[i,j] + 2*u_1[i,j] + \
                Cx2*(approx_x(i,j, side = 0)) - \
                Cy2*(0.5*(q1[i,j] + q1[i,j+1])*(u_1[i,j+1] - u_1[i,j])  - \
                0.5*(q1[i,j] + q1[i,j-1])*(u_1[i,j] - u_1[i,j-1])) + \
            dt2*f(x[i], y[j], t[n])
        j = Iy[0]
        for i in Ix[1:-1]:
            u[i,j] = -u_2[i,j] + 2*u_1[i,j] + \
                Cx2*(0.5*(q1[i,j] + q1[i+1,j])*(u_1[i+1,j] - u_1[i,j])  - \
                0.5*(q1[i,j] + q1[i-1,j])*(u_1[i,j] - u_1[i-1,j])) - \
                Cy2*(approx_y(i,j, side = 0)) + \
            dt2*f(x[i], y[j], t[n])
        i = Ix[-1]
        for j in Iy[1:-1]:
            u[i,j] = -u_2[i,j] + 2*u_1[i,j] + \
                Cx2*(approx_x(i,j, side = 1)) - \
                Cy2*(0.5*(q1[i,j] + q1[i,j+1])*(u_1[i,j+1] - u_1[i,j])  - \
                0.5*(q1[i,j] + q1[i,j-1])*(u_1[i,j] - u_1[i,j-1])) + \
            dt2*f(x[i], y[j], t[n])
        j = Iy[-1]
        for i in Ix[1:-1]: 
            u[i,j] = -u_2[i,j] + 2*u_1[i,j] + \
                Cx2*(0.5*(q1[i,j] + q1[i+1,j])*(u_1[i+1,j] - u_1[i,j])  - \
                0.5*(q1[i,j] + q1[i-1,j])*(u_1[i,j] - u_1[i-1,j])) - \
                Cy2*(approx_y(i,j, side = 1)) + \
            dt2*f(x[i], y[j], t[n])

        # corners
        u[0,0] = -u_2[0,0] + 2*u_1[0,0] + \
                Cx2*(approx_x(0,0, side = 0)) - \
                Cy2*(approx_y(0,0, side = 0)) + \
            dt2*f(x[0], y[0], t[n])
        u[Nx,0] = -u_2[Nx,0] + 2*u_1[Nx,0] + \
                Cx2*(approx_x(Nx,0, side = 1)) - \
                Cy2*(approx_y(Nx,0, side = 0)) + \
            dt2*f(x[Nx], y[0], t[n])
        u[0,Ny] = -u_2[0,Ny] + 2*u_1[0,Ny] + \
                Cx2*(approx_x(0,Ny, side = 0)) - \
                Cy2*(approx_y(0,Ny, side = 1)) + \
            dt2*f(x[0], y[Ny], t[n])
        u[Nx,Ny] = -u_2[Nx,Ny] + 2*u_1[Nx,Ny] + \
                Cx2*(approx_x(Nx,Ny, side = 1)) - \
                Cy2*(approx_y(Nx,Ny, side = 1)) + \
            dt2*f(x[Nx], y[Ny], t[n])

    # Boundary condition u=0
    if bc is not None:
        
        j = Iy[0]
        for i in Ix: u[i,j] = bc
        j = Iy[-1]
        for i in Ix: u[i,j] = bc
        i = Ix[0]
        for j in Iy: u[i,j] = bc
        i = Ix[-1]
        for j in Iy: u[i,j] = bc

    return u

def advance_vectorized(u, u_1, u_2, f_a, Cx2, Cy2, dt2,
                       V=None, step1=False):
    if step1:
        dt = sqrt(dt2)  # save
        Cx2 = 0.5*Cx2;  Cy2 = 0.5*Cy2; dt2 = 0.5*dt2  # redefine
        D1 = 1;  D2 = 0
    else:
        D1 = 2;  D2 = 1
    u_xx = u_1[:-2,1:-1] - 2*u_1[1:-1,1:-1] + u_1[2:,1:-1]
    u_yy = u_1[1:-1,:-2] - 2*u_1[1:-1,1:-1] + u_1[1:-1,2:]
    u[1:-1,1:-1] = D1*u_1[1:-1,1:-1] - D2*u_2[1:-1,1:-1] + \
                   Cx2*u_xx + Cy2*u_yy + dt2*f_a[1:-1,1:-1]
    if step1:
        u[1:-1,1:-1] += dt*V[1:-1, 1:-1]
    # Boundary condition u=0
    j = 0
    u[:,j] = 0
    j = u.shape[1]-1
    u[:,j] = 0
    i = 0
    u[i,:] = 0
    i = u.shape[0]-1
    u[i,:] = 0
    return u



def gaussian(plot_method=2, version='vectorized', save_plot=True):
    """
    Initial Gaussian bell in the middle of the domain.
    plot_method=1 applies mesh function, =2 means surf, =0 means no plot.
    """
    # Clean up plot files
    for name in glob('tmp_*.png'):
        os.remove(name)

    b =0
    Lx = 10
    Ly = 10
    #c = 1.0

    q = lambda x, y : 1.*x/Nx

    def I(x, y):
        """Gaussian peak at (Lx/2, Ly/2)."""
        return exp(-0.5*(x-Lx/2.0)**2 - 0.5*(y-Ly/2.0)**2)


    def plot_u(u, x, xv, y, yv, t, n):
        if t[n] == 0:
            time.sleep(2)
        if plot_method == 1:
            mesh(x, y, u, title='t=%g' % t[n], zlim=[-1,1],
                 caxis=[-1,4])
        elif plot_method == 2:
            surfc(xv, yv, u, title='t=%g' % t[n], zlim=[-1, 1],
                  colorbar=True, colormap=hot(), caxis=[-1,1],
                  shading='flat')
        elif plot_method == 3:
            print 'Experimental 3D matplotlib...under development...'
            #plt.clf()
            ax = fig.add_subplot(111, projection='3d')
            u_surf = ax.plot_surface(xv, yv, u, alpha=0.3)
            #ax.contourf(xv, yv, u, zdir='z', offset=-100, cmap=cm.coolwarm)
            #ax.set_zlim(-1, 1)
            # Remove old surface before drawing
            if u_surf is not None:
                ax.collections.remove(u_surf)
            plt.draw()
            time.sleep(1)
        if plot_method > 0:
            time.sleep(0) # pause between frames
            if save_plot:
                filename = 'tmp_%04d.png' % n
                savefig(filename)  # time consuming!

    Nx = 40; Ny = 40; T = 20
    dt, cpu = solver(I, None, None, q, Lx, Ly, Nx, Ny, -1, T,b,
                     user_action=plot_u, version=version, bc = 0)


def test_constant(plot_method=None, save_plot=False):
    """
    simple test with constant solution
    """
    import math
    import numpy as np

    # Clean up plot files
    for name in glob('tmp_*.png'):
        os.remove(name)

    def plot_u(u, x, xv, y, yv, t, n):
        if t[n] == 0:
            time.sleep(2)
        if plot_method == 1:
            mesh(x, y, u, title='t=%g' % t[n], zlim=[-1,1],
                 caxis=[-1,1])
        elif plot_method == 2:
            surfc(xv, yv, u, title='t=%g' % t[n], zlim=[-1, 4],
                  colorbar=True, colormap=hot(), caxis=[-1,4],
                  shading='flat')
            show()
        elif plot_method == 3:
            print 'Experimental 3D matplotlib...under development...'
            #plt.clf()
            ax = fig.add_subplot(111, projection='3d')
            u_surf = ax.plot_surface(xv, yv, u, alpha=0.3)
            #ax.contourf(xv, yv, u, zdir='z', offset=-100, cmap=cm.coolwarm)
            #ax.set_zlim(-1, 1)
            # Remove old surface before drawing
            if u_surf is not None:
                ax.collections.remove(u_surf)
            plt.draw()
            time.sleep(1)
        if plot_method > 0:
            time.sleep(0.1) # pause between frames
            if save_plot:
                filename = 'tmp_%04d.png' % n
                savefig(filename)  # time consuming!


    exact_solution = lambda x,y : 2.345
    q = lambda x,y : 4.245
    #f =  lambda x,y,t : 0
    I = lambda x,y : 2.345
    V = 0
    Lx = 2
    Ly = 2
    T=1
    Nx = 50
    Ny = 50
    b = 0

    
    def assert_if_error(u,x,y):
        e = abs(u-exact_solution(x,y)).max()
        np.testing.assert_almost_equal(e, 0, decimal=13)

    dt, cpu, u = solver(I, None, None, q, Lx, Ly, Nx, Ny, -1, T,b,
                     user_action=plot_u, version='scalar')     #set user_action=plot_u for visual
    x = linspace(0, Lx, Nx+1)  # mesh points in x dir
    y = linspace(0, Ly, Ny+1)  # mesh points in y dir
    assert_if_error(u,x,y)
    #plt.plot(x,u)
    #plt.show()

def pulse(C=1,            # aximum Courant number
          Nx=40,
          Ny=40,         # spatial resolution
          animate=True,
          version='scalar',
          T=2,            # end time
          loc='left',     # location of initial condition
          pulse_tp='plug',  # pulse/init.cond. type
          slowness_factor=2, # wave vel. in right medium
          skip_frame=1,      # skip frames in animations
          sigma=0.1,        # width measure of the pulse
          plot_method=2, 
          save_plot=False
          ):
    """
    Various peaked-shaped initial conditions on [0,1].
    Wave velocity is decreased by the slowness_factor inside
    medium. The loc parameter can be 'center' or 'left',
    depending on where the initial pulse is to be located.
    The sigma parameter governs the width of the pulse.
    """



    # Use scaled parameters: L=1 for domain length, c_0=1
    # for wave velocity outside the domain.
    Lx = 1.0
    Ly = 1.0
    b = 0
    q = lambda x,y :(0.025/0.0030997558903)
    V =0
    f = 0
    
    if loc == 'center':
        xc = Lx/2.
    elif loc == 'left':
        xc = 0

    if pulse_tp in ('gaussian','Gaussian'):
        def I(x,y):
            return np.exp(-0.5*((x-xc)/sigma)**2)
    elif pulse_tp == 'plug':
        def I(x,y):
            return 0 if abs(x-xc) > sigma else 1
    elif pulse_tp == 'cosinehat':
        def I(x,y):
            # One period of a cosine
            w = 2
            a = w*sigma
            return 0.5*(1 + np.cos(np.pi*(x-xc)/a)) \
                   if xc - a <= x <= xc + a else 0

    elif pulse_tp == 'half-cosinehat':
        def I(x,y):
            # Half a period of a cosine
            w = 4
            a = w*sigma
            return np.cos(np.pi*(x-xc)/a) \
                   if xc - 0.5*a <= x <= xc + 0.5*a else 0
    else:
        raise ValueError('Wrong pulse_tp="%s"' % pulse_tp)

# Clean up plot files
    for name in glob('tmp_*.png'):
        os.remove(name)

    def plot_u(u, x, xv, y, yv, t, n):
        if t[n] == 0:
            time.sleep(2)

        elif plot_method == 2:
            surfc(xv, yv, u, title='t=%g' % t[n], zlim=[-1, 4],
                  colorbar=True, colormap=hot(), caxis=[-1,4],
                  shading='flat')
            #show()

    class Action:
        """Store last solution."""
        def __call__(self, u, x, xv, y, yv, n):
            if n == len(t)-1:
                self.u = u.copy()
                self.x = x.copy()
                self.xv = xv.copy()
                self.y = y.copy()
                self.yv = yv.copy()
                self.t = t[n]

    action = Action()


    #umin=-0.5; umax=1.5*I(xc)
    #casename = '%s_Nx%s_sf%s' % \
    #           (pulse_tp, Nx, slowness_factor)
    #action = PlotMediumAndSolution(
    #    medium, casename=casename, umin=umin, umax=umax,
    #    skip_frame=skip_frame, screen_movie=animate,
    #    backend=None, filename='tmpdata')

    # Choose the stability limit with given Nx, worst case c
    # (lower C will then use this dt, but smaller Nx)
    #dt = (L/Nx)/c_0
    if save_plot:
        solver(I, V, f, q, Lx, Ly, Nx, Ny, -1, T,b,
            user_action=plot_u, version='scalar', bc = None)
    #action.make_movie_file()
    #action.file_close()
    else:
        dt, cpu, u = solver(I, V, f, q, Lx, Ly, Nx, Ny, -1, T,b,
            user_action=action, version='scalar', bc = None)
        tol = 1E-13
        u_0 = np.array([I(x_,y_) for x_,y_ in action.x,action.y])
        diff = np.abs(u - u_0).max()
        assert diff < tol


if __name__ == '__main__':
    #test_quadratic()
    test_constant()
    pulse(plot_method=None, save_plot=True)
    gaussian(plot_method=None, version='scalar', save_plot=False)


