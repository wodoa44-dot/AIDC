from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "3dbef626a7847ab0c047783ee6d43ca9"

def f_update(a, am, v, vm):
    return (a - am, v - vm,)


# empty g_update

def fx_update():
    return (-1, -1)


def fy_update():
    return (1, 1)


def am_ia(a):
    return a


def vm_ia(v):
    return v


# empty sns_update

f_args = ['a', 'am', 'v', 'vm']

g_args = []

j_args = {'fx': [], 'fy': []}

s_args = OrderedDict()

sns_args = []

ia_args = OrderedDict([('am', ['a']), ('vm', ['v'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1]),
             ('fyc', []),
             ('fy', [0, 1]),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1]),
             ('fyc', []),
             ('fy', [2, 3]),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0]),
             ('fyc', []),
             ('fy', [0, 0]),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [])])

j_names = ['fx', 'fy']

init_seq = ['a', 'am', 'v', 'vm']

need_diag_eps = []
