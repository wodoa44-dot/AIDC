from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "ec27e626fd040509549ea25d119e3786"

def f_update(u, v1, v2):
    return (-u*(v1 - v2),)


def g_update(IL):
    return (-IL, IL,)


def fy_update(u):
    return (-u, u)


def gx_update():
    return (-1, 1)


def IL_ia():
    return 0


# empty sns_update

f_args = ['u', 'v1', 'v2']

g_args = ['IL']

j_args = {'fy': ['u'], 'gx': []}

s_args = OrderedDict()

sns_args = []

ia_args = OrderedDict([('IL', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', [0, 0]),
             ('gxc', []),
             ('gx', [0, 1]),
             ('gyc', []),
             ('gy', [])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', [1, 2]),
             ('gxc', []),
             ('gx', [0, 0]),
             ('gyc', []),
             ('gy', [])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', [0, 0]),
             ('gxc', []),
             ('gx', [0, 0]),
             ('gyc', []),
             ('gy', [])])

j_names = ['fy', 'gx']

init_seq = ['IL', 'v1', 'v2']

need_diag_eps = []
