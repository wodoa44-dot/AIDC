from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "68459fb0f007b667fd311f5985ce3919"

# empty f_update

def g_update(Mw, agen, delta, omega, wgen):
    return (-omega, -delta, Mw*wgen, Mw*agen,)


def gy_update():
    return (-1, -1)


def gx_update(Mw):
    return (Mw, Mw)


def omega_ia(a0a):
    return a0a


def delta_ia(d0a):
    return d0a


def Mw_svc(M, Mr):
    return M/Mr


def d0w_svc(Mw, d0):
    return Mw*d0


def a0w_svc(Mw, a0):
    return Mw*a0


# empty sns_update

f_args = []

g_args = ['Mw', 'agen', 'delta', 'omega', 'wgen']

j_args = {'gx': ['Mw'], 'gy': []}

s_args = OrderedDict([('Mw', ['M', 'Mr']), ('d0w', ['Mw', 'd0']), ('a0w', ['Mw', 'a0'])])

sns_args = []

ia_args = OrderedDict([('omega', ['a0a']), ('delta', ['d0a'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', [2, 3]),
             ('gyc', [0, 1]),
             ('gy', [0, 1])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', [0, 1]),
             ('gyc', [2, 3]),
             ('gy', [2, 3])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', [0, 0]),
             ('gyc', [1e-08, 1e-08]),
             ('gy', [0, 0])])

j_names = ['gy', 'gx']

init_seq = ['wgen', 'agen', 'omega', 'delta', 'omega_sub', 'delta_sub']

need_diag_eps = ['omega', 'delta', 'omega', 'delta']
