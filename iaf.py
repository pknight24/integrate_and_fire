from sympy import *
import numpy as np
import matplotlib.pyplot as plt

R, C, ur, t = symbols("R C ur t") # R =: resistance, C =: capacitance
                                  # ur =: membrane resting potential
                                  # t =: time (independent var)

u = symbols("u", cls=Function) # u =: the potential function for which we solve

# these are our parameter values

Rval = 2.600
Cval = 4.9

urval = -70e-3


class I(Function):
    
    @classmethod
    def eval(cls, x):
        return (0.0005 * x) # we define I to be a constant function


dudt = u(t).diff(t)

model = Eq(R * C * dudt, -(u(t) - ur) + R * I(t)) # construct the model symbolically

model = model.subs([(R, Rval), (C, Cval), (ur, urval)]) # then substitute our parameters

sol = dsolve(model, u(t), ics={u(0) : urval}) # sol =:  the solution to the integrate and fire linear equation


print(sol)
eq = sol.rhs

class Leaky():
    
    theta = -0.057 # our threshold for the leaky model
    t0 = 0 # the last time that a "spike" ocurred
    
    def __init__(self):
        self._build_func()

    def run(self, x):
        res = self.f(x)
        if res >= (self.theta - 0.0001):
            res = urval
            self.t0 = x
            self._build_func()
        return res

    def _build_func(self):
        plug = eq.subs(t, t - self.t0)
        self.f = lambdify(t, plug, "math")

leak = Leaky()

ts = np.linspace(0, 1000, 1000) # the time points where the model will be evaluated

results = []

for i in ts:
    results.append(leak.run(i))
results = np.array(results)

plt.plot(ts, results)
plt.xlabel("time (ms)")
plt.ylabel("Membrane potential (V)")
plt.show()
