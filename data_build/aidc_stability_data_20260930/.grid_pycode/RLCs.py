from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "265f00d619e0b88bb23acba93402bb2a"

def f_update(IL, R, u, v1, v2, vC):
    return (u*(-IL*R + v1 - v2 - vC), IL*u,)


def g_update(IL, Idc):
    return (-IL - Idc, -Idc, Idc,)


def fx_update(R, u):
    return (-R*u, -u, u)


def fy_update(u):
    return (u, -u)


def gx_update():
    return (-1,)


def gy_update():
    return (-1, -1, 1)


def IL_ia():
    return 0


def vC_ia(v1, v2):
    return v1 - v2


def Idc_ia():
    return 0


# empty sns_update

f_args = ['IL', 'R', 'u', 'v1', 'v2', 'vC']

g_args = ['IL', 'Idc']

j_args = {'fx': ['R', 'u'], 'fy': ['u'], 'gx': [], 'gy': []}

s_args = OrderedDict()

sns_args = []

ia_args = OrderedDict([('IL', []), ('vC', ['v1', 'v2']), ('Idc', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 1]),
             ('fyc', []),
             ('fy', [0, 0]),
             ('gxc', []),
             ('gx', [0]),
             ('gyc', [0]),
             ('gy', [0, 1, 2])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 0]),
             ('fyc', []),
             ('fy', [3, 4]),
             ('gxc', []),
             ('gx', [0]),
             ('gyc', [2]),
             ('gy', [2, 2, 2])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0]),
             ('gxc', []),
             ('gx', [0]),
             ('gyc', [1e-08]),
             ('gy', [0, 0, 0])])

j_names = ['fx', 'fy', 'gx', 'gy']

init_seq = ['IL', 'v1', 'v2', 'vC', 'Idc']

need_diag_eps = ['Idc', 'Idc']
