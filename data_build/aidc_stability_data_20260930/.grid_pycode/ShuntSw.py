from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "7c080939009709e6f0e7b320e40904bb"

# empty f_update

def g_update(beff, geff, u, v):
    return (geff*u*v**2, -beff*u*v**2,)


def gy_update(beff, geff, u, v):
    return (2*geff*u*v, -2*beff*u*v)


def vlo_svc(dv, vref):
    return -dv + vref


def vup_svc(dv, vref):
    return dv + vref


# empty sns_update

f_args = []

g_args = ['beff', 'geff', 'u', 'v']

j_args = {'gy': ['beff', 'geff', 'u', 'v']}

s_args = OrderedDict([('vlo', ['dv', 'vref']), ('vup', ['dv', 'vref'])])

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
