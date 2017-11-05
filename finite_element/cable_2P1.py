
import sympy as sym
import numpy as np

def solver(integrand_lhs, integrand_rhs, psi, Omega,
           boundary_lhs=None, boundary_rhs=None, symbolic=True):
    N = len(psi[0]) - 1
    A = sym.zeros(N+1, N+1)
    b = sym.zeros(N+1, 1)
    x = sym.Symbol('x')
    for i in range(N+1):
        for j in range(i, N+1):
            integrand = integrand_lhs(psi, i, j)
            if symbolic:
                I = sym.integrate(integrand, (x, Omega[0], Omega[1]))
                if isinstance(I, sym.Integral):
                    symbolic = False  # force num.int. hereafter
            if not symbolic:
                integrand = sym.lambdify([x], integrand)
                I = sym.mpmath.quad(integrand, [Omega[0], Omega[1]])
            if boundary_lhs is not None:
                I += boundary_lhs(psi, i, j)
            A[i,j] = A[j,i] = I
        integrand = integrand_rhs(psi, i)
        if symbolic:
            I = sym.integrate(integrand, (x, Omega[0], Omega[1]))
            if isinstance(I, sym.Integral):
                symbolic = False
        if not symbolic:
            integrand = sym.lambdify([x], integrand)
            I = sym.mpmath.quad(integrand, [Omega[0], Omega[1]])
        if boundary_rhs is not None:
            I += boundary_rhs(psi, i)
        b[i,0] = I
    c = A.LUsolve(b)
    u = sum(c[i,0]*psi[0][i] for i in range(len(psi[0])))
    return u, c

def vizualize(u):
	import matplotlib.pyplot as plt
	x = np.linspace(0,2,101)
	b = -1	
	u_ = u(x,b)

	plt.plot(x,u_)
	plt.legend()
	plt.show()


if __name__ == '__main__':

	x, b = sym.symbols('x b')
	f = b
	B = 1 - x**3
	dBdx = sym.diff(B, x)

	# Compute basis functions and their derivatives
	N = 2
	psi = {0: [x**(i+1)*(1-x) for i in range(N+1)]}
	psi[1] = [sym.diff(psi_i, x) for psi_i in psi[0]]

	def integrand_lhs(psi, i, j):
    		return psi[1][i]*psi[1][j]

	def integrand_rhs(psi, i):
    		return f*psi[0][i] - dBdx*psi[1][i]

	Omega = [0, 1]

	u_bar, c = solver(integrand_lhs, integrand_rhs, psi, Omega, symbolic=True)
	u = B + u_bar
	print 'solution u:', sym.simplify(sym.expand(u))
	
	u = sym.simplify(sym.expand(u))
	u = sym.lambdify((x,b),u)

	vizualize(u)




