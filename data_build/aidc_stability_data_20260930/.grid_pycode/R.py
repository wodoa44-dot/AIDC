from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "7a4ce46998ac0eab247219933f1d13db"

# empty f_update

def g_update(Idc, R, u, v1, v2):
    return (-Idc + u*(-v1 + v2)/R, -Idc, Idc,)


def gy_update(R, u):
    return (-1, -u/R, u/R, -1, 1)


def Idc_ia(R, u, v1, v2):
    return u*(-v1 + v2)/R


# empty sns_update

f_args = []

g_args = ['Idc', 'R', 'u', 'v1', 'v2']

j_args = {'gy': ['R', 'u']}

s_args = OrderedDict()

sns_args = []

ia_args = OrderedDict([('Idc', ['R', 'u', 'v1', 'v2'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0, 0, 1, 2])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 1, 2, 0, 0])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0, 0, 0, 0])])

j_names = ['gy']

init_seq = ['v1', 'v2', 'Idc']

need_diag_eps = []
