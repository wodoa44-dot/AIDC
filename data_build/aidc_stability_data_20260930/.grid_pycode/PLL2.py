from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "957019166039c617d97d2c72860585b0"

def f_update(Ki, PI_y, a, am, fn, v):
    return (Ki*v*sin(a - am), 2*pi*PI_y*fn,)


def g_update(Kp, PI_xi, PI_y, a, am, v):
    return (Kp*v*sin(a - am) + PI_xi - PI_y, 0, 0,)


def fx_update(Ki, a, am, v):
    return (-Ki*v*cos(a - am),)


def fy_update(Ki, a, am, fn, v):
    return (Ki*v*cos(a - am), Ki*sin(a - am), 2*pi*fn)


def gx_update(Kp, a, am, v):
    return (1, -Kp*v*cos(a - am))


def gy_update(Kp, a, am, v):
    return (-1, Kp*v*cos(a - am), Kp*sin(a - am))


def PI_xi_ia():
    return 0


def am_ia(a):
    return a


def PI_y_ia(Kp, a, am, v):
    return Kp*v*sin(a - am)


# empty sns_update

f_args = ['Ki', 'PI_y', 'a', 'am', 'fn', 'v']

g_args = ['Kp', 'PI_xi', 'PI_y', 'a', 'am', 'v']

j_args = {'fx': ['Ki', 'a', 'am', 'v'],
 'fy': ['Ki', 'a', 'am', 'fn', 'v'],
 'gx': ['Kp', 'a', 'am', 'v'],
 'gy': ['Kp', 'a', 'am', 'v']}

s_args = OrderedDict()

sns_args = []

ia_args = OrderedDict([('PI_xi', []), ('am', ['a']), ('PI_y', ['Kp', 'a', 'am', 'v'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [0, 0, 1]),
             ('gxc', []),
             ('gx', [0, 0]),
             ('gyc', []),
             ('gy', [0, 0, 0])])

jjac = OrderedDict([('fxc', []),
             ('fx', [1]),
             ('fyc', []),
             ('fy', [3, 4, 2]),
             ('gxc', []),
             ('gx', [0, 1]),
             ('gyc', []),
             ('gy', [2, 3, 4])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0]),
             ('gyc', []),
             ('gy', [0, 0, 0])])

j_names = ['fx', 'fy', 'gx', 'gy']

init_seq = ['PI_xi', 'a', 'am', 'v', 'PI_y']

need_diag_eps = []
