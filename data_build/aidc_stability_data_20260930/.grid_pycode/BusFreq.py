from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "f20cf461b8ef0e417c0c6872febb5fb8"

def f_update(L_y, WO_x, a, a0):
    return (-L_y + a - a0, L_y - WO_x,)


def g_update(L_y, Tw, WO_x, WO_y, f, iwn):
    return (-Tw*WO_y + iwn*(L_y - WO_x), WO_y - f + 1, 0, 0,)


def fx_update():
    return (-1, 1, -1)


def fy_update():
    return (1,)


def gx_update(iwn):
    return (iwn, -iwn)


def gy_update(Tw):
    return (-Tw, 1, -1)


def L_y_ia(a, a0):
    return a - a0


def WO_x_ia(L_y):
    return L_y


def WO_y_ia():
    return 0


def f_ia():
    return 1


def iwn_svc(fn, u):
    return (1/2)*u/(pi*fn)


# empty sns_update

f_args = ['L_y', 'WO_x', 'a', 'a0']

g_args = ['L_y', 'Tw', 'WO_x', 'WO_y', 'f', 'iwn']

j_args = {'fx': [], 'fy': [], 'gx': ['iwn'], 'gy': ['Tw']}

s_args = OrderedDict([('iwn', ['fn', 'u'])])

sns_args = []

ia_args = OrderedDict([('L_y', ['a', 'a0']), ('WO_x', ['L_y']), ('WO_y', []), ('f', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 1]),
             ('fyc', []),
             ('fy', [0]),
             ('gxc', []),
             ('gx', [0, 0]),
             ('gyc', [0]),
             ('gy', [0, 1, 1])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 1]),
             ('fyc', []),
             ('fy', [4]),
             ('gxc', []),
             ('gx', [0, 1]),
             ('gyc', [2]),
             ('gy', [2, 2, 3])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0]),
             ('fyc', []),
             ('fy', [0]),
             ('gxc', []),
             ('gx', [0, 0]),
             ('gyc', [1e-08]),
             ('gy', [0, 0, 0])])

j_names = ['fx', 'fy', 'gx', 'gy']

init_seq = ['a', 'L_y', 'WO_x', 'WO_y', 'f', 'v']

need_diag_eps = ['WO_y', 'WO_y']
