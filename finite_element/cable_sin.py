import numpy as np


def model1():
	"""
	using sympy as in lecturenotes
	Solve -u'' = f(x), u(0)=0, u(L)=D."""
        import sympy as sym
	x, c_0, c_1, = sym.symbols('x c_0 c_1')
	# Integrate twice
	u_x = sym.integrate(1, (x, 0, x)) + c_0 # RHS = 1
	u = sym.integrate(u_x, (x, 0, x)) + c_1
	# Set up 2 equations from the 2 boundary conditions and solve
	# with respect to the integration constants c_0, c_1
	r = sym.solve([u.subs(x, 0)-0,  # x=0 condition
    	          u.subs(x,2)-0],  # x=1 condition
    	          [c_0, c_1])       # unknowns
	# Substitute the integration constants in the solution
	u = u.subs(c_0, r[c_0]).subs(c_1, r[c_1])
	u = sym.simplify(sym.expand(u))
	#print u
	return sym.lambdify(x,u)



def compute_coeff_LS(printout=True):
	"""

	"""
	import sympy as sym

	i, j = sym.symbols('i j', integer=True)
	x, L = sym.symbols('x L')
	f = 1
	a = -8/(sym.pi**2*(2*i+1)**2)
	c_i = a*sym.integrate(f*sym.sin((2*i+1)*sym.pi*x/2), (x, 0, 1))
	c_i = sym.simplify(c_i)
	c = sym.lambdify(i,c_i)
	if printout==True:
		print("least squares c_i/c_i+1 ratio:")
		print("coeff: %s" % c_i)
		for i in range(0,10):
			print c(i+1)/c(i)
	return c

def compute_max_error(N=0):
	"""
	"""
	exact = model1()
	c = compute_coeff_LS(printout=False)
	print("erorrs: ------------")
	for i in range(0,10):
		error = c(i)*np.sin((2*i+1)*(np.pi/2))-exact(1)
		print(abs(error))
	
def vizualize():
	import matplotlib.pyplot as plt
	x = np.linspace(0,1,101)
	c = compute_coeff_LS(printout=False)
	exact = model1()
	FEM_0 = c(0)*np.sin((1)*((np.pi*x)/2.))
	FEM_1 = c(0)*np.sin((1)*((np.pi*x)/2.))+c(1)*np.sin((2)*((np.pi*x)/2.))
	FEM_20 = c(0)*np.sin((1)*((np.pi*x)/2.))
	for i in range(1,20):
		FEM_20 += c(i)*np.sin((2*i+1)*((np.pi*x)/2.))
	plt.plot(x,FEM_0, label = "N=0")
	plt.hold('on')
	plt.plot(x,FEM_1, label = "N=1")
	plt.hold('on')
	plt.plot(x,FEM_20, label = "N=20")
	plt.hold('on')
	plt.plot(x,exact(x), label = "exact")
	plt.legend()
	plt.show()

def compute_coeff_LS_d(printout=True):
	"""

	"""
	import sympy as sym

	i, j = sym.symbols('i j', integer=True)
	x, L = sym.symbols('x L')
	f = 1
	a = -8/(sym.pi**2*(i+1)**2)
	c_i = a*sym.integrate(f*sym.sin((i+1)*sym.pi*x/2), (x, 0, 1))
	c_i = sym.simplify(c_i)
	c = sym.lambdify(i,c_i)
	if printout==True:
		print("least squares c_i/c_i+1 ratio:")
		print("coeff: %s" % c_i)
		for i in range(0,10):
			print c(i+1)/c(i)
	return c


def vizualize_d():
	import matplotlib.pyplot as plt
	x = np.linspace(0,1,101)
	c = compute_coeff_LS(printout=False)
	exact = model1()
	FEM_20 = c(0)*np.sin((1)*((np.pi*x)/2.))
	for i in range(1,20):
		if i % 2 == 0:       # even i
            		FEM_20 += - c(i)*np.sin((i+1)*((np.pi*x)/2.))
        	elif (i-1) % 4 == 0:   # 1, 5, 9, 13, 17
            		FEM_20 += - 2*c(i)*np.sin((i+1)*((np.pi*x)/2.))
        	else:
            		FEM_20 += 0

	plt.plot(x,FEM_20, label = "N=20")
	plt.hold('on')
	plt.plot(x,exact(x), label = "exact")
	plt.legend()
	plt.show()

def vizualize_e():
	import matplotlib.pyplot as plt
	x = np.linspace(0,2,101)
	c = compute_coeff_LS(printout=False)
	exact = model1()
	FEM_20 = c(0)*np.sin((1)*((np.pi*x)/2.))
	for i in range(1,20):
		if i % 2 == 0:       # even i
            		FEM_20 += - c(i)*np.sin((i+1)*((np.pi*x)/2.))
        	elif (i-1) % 4 == 0:   # 1, 5, 9, 13, 17
            		FEM_20 += - 2*c(i)*np.sin((i+1)*((np.pi*x)/2.))
        	else:
            		FEM_20 += 0

	plt.plot(x,FEM_20, label = "N=20")
	plt.hold('on')
	plt.plot(x,exact(x), label = "exact")
	plt.legend()
	plt.show()



if __name__ == '__main__':
	compute_coeff_LS()
	compute_max_error()
	vizualize()
	vizualize_d()
	vizualize_e()
	
