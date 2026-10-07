from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "dedbd8763c120a2f4a3c1d00f7a00f48"

def f_update(Idc, R, u, vC):
    return (-u*(Idc - vC/R),)


def g_update(Idc, u, v1, v2, vC):
    return (Idc*(1 - u) + u*(-v1 + v2 + vC), -Idc, Idc,)


def fx_update(R, u):
    return (u/R,)


def fy_update(u):
    return (-u,)


def gx_update(u):
    return (u,)


def gy_update(u):
    return (1 - u, -u, u, -1, 1)


def vC_ia(v1, v2):
    return v1 - v2


def Idc_ia(R, v1, v2):
    return (-v1 + v2)/R


# empty sns_update

f_args = ['Idc', 'R', 'u', 'vC']

g_args = ['Idc', 'u', 'v1', 'v2', 'vC']

j_args = {'fx': ['R', 'u'], 'fy': ['u'], 'gx': ['u'], 'gy': ['u']}

s_args = OrderedDict()

sns_args = []

ia_args = OrderedDict([('vC', ['v1', 'v2']), ('Idc', ['R', 'v1', 'v2'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [0]),
             ('gxc', []),
             ('gx', [0]),
             ('gyc', [0]),
             ('gy', [0, 0, 0, 1, 2])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [1]),
             ('gxc', []),
             ('gx', [0]),
             ('gyc', [1]),
             ('gy', [1, 2, 3, 1, 1])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [0]),
             ('gxc', []),
             ('gx', [0]),
             ('gyc', [1e-08]),
             ('gy', [0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gx', 'gy']

init_seq = ['v1', 'v2', 'vC', 'Idc']

need_diag_eps = ['Idc', 'Idc']
