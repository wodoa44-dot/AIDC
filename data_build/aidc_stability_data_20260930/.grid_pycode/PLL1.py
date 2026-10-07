from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "3c998a8ff60f298ee4e1cf64d7274541"

def f_update(Ki, PI_y, a, ae, af_y, am, fn, u):
    return (a - af_y, Ki*u*(af_y - am), 2*pi*PI_y*fn, ae - am,)


def g_update(Kp, PI_xi, PI_y, af_y, am, u):
    return (Kp*u*(af_y - am) + PI_xi - PI_y, 0,)


def fx_update(Ki, u):
    return (-1, Ki*u, -Ki*u, 1, -1)


def fy_update(fn):
    return (1, 2*pi*fn)


def gx_update(Kp, u):
    return (Kp*u, 1, -Kp*u)


def gy_update():
    return (-1,)


def af_y_ia(a):
    return a


def PI_xi_ia():
    return 0.0


def ae_ia(a):
    return a


def am_ia(a):
    return a


def PI_y_ia(Kp, af_y, am, u):
    return Kp*u*(af_y - am)


# empty sns_update

f_args = ['Ki', 'PI_y', 'a', 'ae', 'af_y', 'am', 'fn', 'u']

g_args = ['Kp', 'PI_xi', 'PI_y', 'af_y', 'am', 'u']

j_args = {'fx': ['Ki', 'u'], 'fy': ['fn'], 'gx': ['Kp', 'u'], 'gy': []}

s_args = OrderedDict()

sns_args = []

ia_args = OrderedDict([('af_y', ['a']),
             ('PI_xi', []),
             ('ae', ['a']),
             ('am', ['a']),
             ('PI_y', ['Kp', 'af_y', 'am', 'u'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 1, 3, 3]),
             ('fyc', []),
             ('fy', [0, 2]),
             ('gxc', []),
             ('gx', [0, 0, 0]),
             ('gyc', []),
             ('gy', [0])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 3, 2, 3]),
             ('fyc', []),
             ('fy', [5, 4]),
             ('gxc', []),
             ('gx', [0, 1, 3]),
             ('gyc', []),
             ('gy', [4])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0]),
             ('gyc', []),
             ('gy', [0])])

j_names = ['fx', 'fy', 'gx', 'gy']

init_seq = ['a', 'af_y', 'PI_xi', 'ae', 'am', 'PI_y']

need_diag_eps = []
