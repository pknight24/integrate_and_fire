from sympy import *
import numpy as np

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
        return 0.05 # we define I to be a constant function


dudt = u(t).diff(t)

model = Eq(R * C * dudt, -(u(t) - ur) + R * I(t)) # construct the model symbolically

model = model.subs([(R, Rval), (C, Cval), (ur, urval)]) # then substitute our parameters

sol = dsolve(model, u(t), ics={u(0) : urval}) # sol =:  the solution to the integrate and fire linear equation


print(sol)


ut = lambdify(t, sol.rhs, "numpy") # the actual solution the the model, converted to a numeric function

theta = -55e-3 # our threshold value for the leaky model


ts = np.linspace(0, 1000, 10000) # the time points where the model will be evaluated

# TODO: 
#   integrate the threshold functionality

