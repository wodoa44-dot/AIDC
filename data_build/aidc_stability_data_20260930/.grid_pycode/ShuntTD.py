from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "c80293d6aaf58dbe4a4c38df49442910"

# empty f_update

def g_update(a, b, dae_t, g, sys_f, u, v, vta, vtb, vtc):
    return ((1/3)*sqrt(3)*v*cos(a + 2*pi*dae_t*sys_f) - vta, -1/3*sqrt(3)*v*cos(a + 2*pi*dae_t*sys_f + (1/3)*pi) - vtb, -1/3*sqrt(3)*v*sin(a + 2*pi*dae_t*sys_f + (1/6)*pi) - vtc, g*u*v**2, -b*u*v**2,)


def gy_update(a, b, dae_t, g, sys_f, u, v):
    return (-1, -1/3*sqrt(3)*v*sin(a + 2*pi*dae_t*sys_f), (1/3)*sqrt(3)*cos(a + 2*pi*dae_t*sys_f), -1, (1/3)*sqrt(3)*v*sin(a + 2*pi*dae_t*sys_f + (1/3)*pi), -1/3*sqrt(3)*cos(a + 2*pi*dae_t*sys_f + (1/3)*pi), -1, -1/3*sqrt(3)*v*cos(a + 2*pi*dae_t*sys_f + (1/6)*pi), -1/3*sqrt(3)*sin(a + 2*pi*dae_t*sys_f + (1/6)*pi), 2*g*u*v, -2*b*u*v)


def vta_ia(a, dae_t, sys_f, v):
    return (1/3)*sqrt(3)*v*cos(a + 2*pi*dae_t*sys_f)


def vtb_ia(a, dae_t, sys_f, v):
    return -1/3*sqrt(3)*v*cos(a + 2*pi*dae_t*sys_f + (1/3)*pi)


def vtc_ia(a, dae_t, sys_f, v):
    return -1/3*sqrt(3)*v*sin(a + 2*pi*dae_t*sys_f + (1/6)*pi)


# empty sns_update

f_args = []

g_args = ['a', 'b', 'dae_t', 'g', 'sys_f', 'u', 'v', 'vta', 'vtb', 'vtc']

j_args = {'gy': ['a', 'b', 'dae_t', 'g', 'sys_f', 'u', 'v']}

s_args = OrderedDict()

sns_args = []

ia_args = OrderedDict([('vta', ['a', 'dae_t', 'sys_f', 'v']),
             ('vtb', ['a', 'dae_t', 'sys_f', 'v']),
             ('vtc', ['a', 'dae_t', 'sys_f', 'v'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 4])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 3, 4, 1, 3, 4, 2, 3, 4, 4, 4])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['gy']

init_seq = ['a', 'v', 'vta', 'vtb', 'vtc']

need_diag_eps = []
