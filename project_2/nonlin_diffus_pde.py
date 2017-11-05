from fenics import *
import numpy



# Define boundary condition
#u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)
# Neumann bc is default?


def boundary(x, on_boundary): # not needed?
	return on_boundary


def solve_nonlin_diffus(N=16, degree=1, I=Expression("0.5"), 
			dimension = 1, rho = 0.5, f = Constant(0.0), 
			plot_status = True,task_d=False,task_e=False,task_f=False):
	"""
	Solves the equation
	
	\rho u_t = \nabla\cdot (\alpha (u)\nabla u) + f(\vec x,t)
	
	The parameters given to the solver are
	N: defines mesh refinery
	dregree: of elements P1, P2....
	I: function that specifies initial condition
	dimension: in space, 1D, 2D, 3D
	rho: constant on expression
	f: function in PDE, can use Expresison("C++ syntax")
	the rest are specific cases

	The solver has hardcoded T = 1. and dt is decided by the 
	mesh spacing. The mesh is always unity.
	"""
	
	divisions=[N for i in range(dimension)]
	#dimension = len(divisions)
	domain_type = [UnitIntervalMesh, UnitSquareMesh, UnitCubeMesh]
	mesh = domain_type[dimension-1](*divisions)
	V = FunctionSpace(mesh, 'Lagrange', degree)

	if task_f:
		f = Expression("-rho*pow(x[0],3)/3.0 + rho*pow(x[0],2)/2.0 + 8*pow(t,3)*pow(x[0],7)/9.0 - 28*pow(t,3)*pow(x[0],6)/9.0 + 7*pow(t,3)*pow(x[0],5)/2.0 - 5*pow(t,3)*pow(x[0],4)/4.0 + 2*t*x[0] - t", t=0, rho=rho)
		u_exact = Expression("t*pow(x[0],2)*(0.5-x[0]/3.0)", t=0)

	# Create mesh and define function space
	u1 = interpolate(I, V)

	T = 1.0
	t = 0
	dt = (1.0/(divisions[0]-1))**2 # dt = dx**2 = dy**2 in two dimensions
	#print dt
	

	# Define variational problem
	u = TrialFunction(V)
	v = TestFunction(V)
	
	a = rho*dot(u, v)*dx + dt*dot(alpha(u1)*grad(u), grad(v))*dx
	L = rho*dot(u1, v)*dx + dt*dot(f, v)*dx

	# Compute solution
	u = Function(V)
	solve(a == L, u)

	u = Function(V)
	u = interpolate(I, V)
	
	if plot_status:
		figure = plot(u)
		figure.set_min_max(-1, 1)
	if task_e:
		u_exact = Expression("exp(-pi*pi*t)*cos(pi*x[0])", t=0)

	# Effectively one iteration picard method inside time loop
	t = dt # first step already calculated
	err_list = [] # task_f
	t_list = []
	while t<T:
		solve(a==L, u)
		if plot_status:
			figure.plot(u)
		u1.assign(u)
		t += dt
		if task_d:
			print u.vector().array()
		if task_e:
			if (t > 0.05):		
				u_exact.t = t
				u_e = interpolate(u_exact, V) # discrete points in V
				if plot_status:
					figure.plot(u_e)
				e = u_e.vector().array()-u.vector().array()
				E = numpy.sqrt(numpy.sum(e**2)/u.vector().array().size)
				return E/dt
				break
		if task_f:	
			u_exact.t = t
			u_e = interpolate(u_exact, V) # discrete points in V
			if plot_status:
				figure.plot(u_e)
				#interactive()
			e = u_e.vector().array()-u.vector().array()
			E = numpy.sqrt(numpy.sum(e**2)/u.vector().array().size)
			err_list.append(E/dt)
			t_list.append(t)
			if (t > T-2*dt):
				return err_list, t_list
				break		
			
	
	

if __name__ == '__main__':
	"""
	set the specific task to True for the solver to solve it.
	If no task is set to True, nothing is computed and there are no printouts.
	Set one and one to True to see the correct printouts.
	"""

	task_d = False#True
	task_e = False#True
	task_f = True
	demo = False
	
	if task_d:
		alpha = lambda u : 1.
		solve_nonlin_diffus(plot_status = False)
		solve_nonlin_diffus(degree=2, dimension=2, 
		plot_status = False,task_d=task_d,task_e=False)
	if task_e:
		alpha = lambda u : 1.
		error1 = solve_nonlin_diffus(N=32, degree=1, I=Expression("cos(pi*x[0])"), 
			dimension = 2, rho = 1.0, f = Constant(0.0), 
			plot_status=True,task_d=False,task_e=task_e)
		error2 =solve_nonlin_diffus(N=32, degree=1, I=Expression("cos(pi*x[0])"), 
			dimension = 2, rho = 1.0, f = Constant(0.0), 
			plot_status=True,task_d=False,task_e=task_e)
		error3 =solve_nonlin_diffus(N=64, degree=1, I=Expression("cos(pi*x[0])"), 
			dimension = 2, rho = 1.0, f = Constant(0.0), 
			plot_status=True,task_d=False,task_e=task_e)
		print"L2 norm for N = 16,32,64 is %.6f, %.6f, %.6f" %(error1, error2, error3)
	if task_f:
		import matplotlib.pyplot as mpl
		alpha = lambda u : 1. + u**2
		
		err_list, t_list = solve_nonlin_diffus(N=16, degree=1, I=Expression("0"), 
			dimension = 1, rho = 1., 
			plot_status = False,task_f=task_f)
		mpl.plot(t_list, err_list)
		mpl.xlabel("time")
		mpl.ylabel("error")
		mpl.show()

	if demo:
		alpha = lambda u : 0.01
		solve_nonlin_diffus(N=8, degree=1, I=Expression("exp(-0.5/pow(sigma, 2)*(x[0]*x[0]+x[1]*x[1]))", sigma=0.2), 
			dimension = 2, rho = 0.5, f = Expression("(x[0])*t", t=0), 
			plot_status = True)
		





