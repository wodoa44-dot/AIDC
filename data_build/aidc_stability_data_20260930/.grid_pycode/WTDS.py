from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "966d5157a4055b4e8649195c97b85578"

def f_update(D, Pe, Pm, s1_y, wge, wr0):
    return (-1.0*D*(s1_y - wr0) + 1.0*(-Pe + Pm)/wge, 0, 0, 0,)


def g_update(Pe0, Pm, s1_y, w0, wr0):
    return (Pe0 - Pm, w0 - wr0, s1_y - 1.0, 0,)


def fx_update(D):
    return (-1.0*D,)


def fy_update(D, Pe, Pm, wge):
    return (1.0/wge, 1.0*D, -1.0*(-Pe + Pm)/wge**2, -1.0/wge)


def gy_update():
    return (-1, -1)


def gx_update():
    return (1,)


def wr0_ia(w0):
    return w0


def s1_y_ia(wr0):
    return wr0


def Pm_ia(Pe0):
    return Pe0


def H2_svc(H):
    return 2*H


def Kshaft_svc():
    return 1.00000000000000


# empty sns_update

f_args = ['D', 'Pe', 'Pm', 's1_y', 'wge', 'wr0']

g_args = ['Pe0', 'Pm', 's1_y', 'w0', 'wr0']

j_args = {'fx': ['D'], 'fy': ['D', 'Pe', 'Pm', 'wge'], 'gx': [], 'gy': []}

s_args = OrderedDict([('H2', ['H']), ('Kshaft', [])])

sns_args = []

ia_args = OrderedDict([('wr0', ['w0']), ('s1_y', ['wr0']), ('Pm', ['Pe0'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [2]),
             ('gyc', []),
             ('gy', [0, 1])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [4, 5, 6, 7]),
             ('gxc', []),
             ('gx', [0]),
             ('gyc', []),
             ('gy', [4, 5])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0]),
             ('gyc', []),
             ('gy', [0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['wr0', 's1_y', 's3_y', 'wt', 'wg', 'Pm', 'wge', 'Pe']

need_diag_eps = []
