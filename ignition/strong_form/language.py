"""Defines the strong_form language"""

from sympy import Symbol

class Unknowns(Symbol):
    """Represents an unknown quantitity"""
    def __new__(cls, name, dim=1, space="L2"):
        obj = Symbol.__new__(cls, name)
        obj.dim = dim
        obj.space = space
        return obj

class Space(object):
    """Represents a function space"""
    pass

class Domain(object):
    """Represents a domain"""
    pass

class Function(object):
    pass

class Variable(object):
    pass

class Operator(object):
    pass
