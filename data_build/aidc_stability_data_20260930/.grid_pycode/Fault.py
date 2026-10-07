from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "4e521475b2d23f83d1ef21de1d40f2f9"

# empty f_update

def g_update(bf, gf, u, uf, v):
    return (gf*u*uf*v**2, -bf*u*uf*v**2,)


def gy_update(bf, gf, u, uf, v):
    return (2*gf*u*uf*v, -2*bf*u*uf*v)


def gf_svc(rf, xf):
    return (real(rf) - imag(xf))/((real(rf) - imag(xf))**2 + (real(xf) + imag(rf))**2)


def bf_svc(rf, xf):
    return (-real(xf) - imag(rf))/((real(rf) - imag(xf))**2 + (real(xf) + imag(rf))**2)


def uf_svc():
    return 0


# empty sns_update

f_args = []

g_args = ['bf', 'gf', 'u', 'uf', 'v']

j_args = {'gy': ['bf', 'gf', 'u', 'uf', 'v']}

s_args = OrderedDict([('gf', ['rf', 'xf']), ('bf', ['rf', 'xf']), ('uf', [])])

sns_args = []

ia_args = OrderedDict()

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 1])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [1, 1])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0])])

j_names = ['gy']

init_seq = ['a', 'v']

need_diag_eps = []
