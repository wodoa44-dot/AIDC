from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "6c979db5624a437c34afc2c245625f08"

def f_update(IL, R, u, v1, v2):
    return (u*(-IL*R + v1 - v2),)


def g_update(IL, Idc, u):
    return (-IL*u - Idc, -Idc, Idc,)


def fx_update(R, u):
    return (-R*u,)


def fy_update(u):
    return (u, -u)


def gx_update(u):
    return (-u,)


def gy_update():
    return (-1, -1, 1)


def IL_ia(R, v1, v2):
    return (v1 - v2)/R


def Idc_ia(R, u, v1, v2):
    return -u*(v1 - v2)/R


# empty sns_update

f_args = ['IL', 'R', 'u', 'v1', 'v2']

g_args = ['IL', 'Idc', 'u']

j_args = {'fx': ['R', 'u'], 'fy': ['u'], 'gx': ['u'], 'gy': []}

s_args = OrderedDict()

sns_args = []

ia_args = OrderedDict([('IL', ['R', 'v1', 'v2']), ('Idc', ['R', 'u', 'v1', 'v2'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [0, 0]),
             ('gxc', []),
             ('gx', [0]),
             ('gyc', []),
             ('gy', [0, 1, 2])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [2, 3]),
             ('gxc', []),
             ('gx', [0]),
             ('gyc', []),
             ('gy', [1, 1, 1])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [0, 0]),
             ('gxc', []),
             ('gx', [0]),
             ('gyc', []),
             ('gy', [0, 0, 0])])

j_names = ['fx', 'fy', 'gx', 'gy']

init_seq = ['v1', 'v2', 'IL', 'Idc']

need_diag_eps = []
