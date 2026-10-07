from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "3bb19136bb47a4db66f34c9ae2b1856d"

def f_update(K, LAW_y, LL_x, LL_y, vi):
    return (-LL_x + vi, K*LL_y - LAW_y, 0,)


def g_update(LAW_y, LL_LT1_z1, LL_LT2_z1, LL_x, LL_y, TA, TB, ue, v, vbus, vf0, vi, vout, vref, vref0):
    return (-v + vbus, LAW_y*ue - vout, -vref + vref0, -v - vi + vref, LL_LT1_z1*LL_LT2_z1*(-LL_x + LL_y) + LL_x*TB - LL_y*TB + TA*(-LL_x + vi), ue*(-vf0 + vout), 0, 0, 0,)


def fx_update():
    return (-1, -1)


def fy_update(K):
    return (1, K)


def gy_update(LL_LT1_z1, LL_LT2_z1, TA, TB, ue):
    return (-1, 1, -1, -1, -1, 1, -1, TA, LL_LT1_z1*LL_LT2_z1 - TB, ue)


def gx_update(LL_LT1_z1, LL_LT2_z1, TA, TB, ue):
    return (ue, -LL_LT1_z1*LL_LT2_z1 - TA + TB)


def vi_ia(K, vf0):
    return vf0/K


def LL_x_ia(vi):
    return vi


def LL_y_ia(vi):
    return vi


def LAW_y_ia(K, LL_y):
    return K*LL_y


def v_ia(vbus):
    return vbus


def vout_ia(ue, vf0):
    return ue*vf0


def vref_ia(K, v, vf0):
    return v + vf0/K


def ue_svc(u, ug):
    return u*ug


def TA_svc(TATB, TB):
    return TATB*TB


def vref0_svc(vref):
    return vref


# empty sns_update

f_args = ['K', 'LAW_y', 'LL_x', 'LL_y', 'vi']

g_args = ['LAW_y',
 'LL_LT1_z1',
 'LL_LT2_z1',
 'LL_x',
 'LL_y',
 'TA',
 'TB',
 'ue',
 'v',
 'vbus',
 'vf0',
 'vi',
 'vout',
 'vref',
 'vref0']

j_args = {'fx': [],
 'fy': ['K'],
 'gx': ['LL_LT1_z1', 'LL_LT2_z1', 'TA', 'TB', 'ue'],
 'gy': ['LL_LT1_z1', 'LL_LT2_z1', 'TA', 'TB', 'ue']}

s_args = OrderedDict([('ue', ['u', 'ug']), ('TA', ['TATB', 'TB']), ('vref0', ['vref'])])

sns_args = []

ia_args = OrderedDict([('vi', ['K', 'vf0']),
             ('LL_x', ['vi']),
             ('LL_y', ['vi']),
             ('LAW_y', ['K', 'LL_y']),
             ('v', ['vbus']),
             ('vout', ['ue', 'vf0']),
             ('vref', ['K', 'v', 'vf0'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1]),
             ('fyc', []),
             ('fy', [0, 1]),
             ('gxc', []),
             ('gx', [1, 4]),
             ('gyc', [1, 4]),
             ('gy', [0, 0, 1, 2, 3, 3, 3, 4, 4, 5])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1]),
             ('fyc', []),
             ('fy', [6, 7]),
             ('gxc', []),
             ('gx', [1, 0]),
             ('gyc', [4, 7]),
             ('gy', [3, 11, 4, 5, 3, 5, 6, 6, 7, 4])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0]),
             ('fyc', []),
             ('fy', [0, 0]),
             ('gxc', []),
             ('gx', [0, 0]),
             ('gyc', [1e-08, 1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['vi',
 'LL_x',
 'LL_y',
 'LAW_y',
 'omega',
 'vbus',
 'v',
 'vout',
 'vref',
 'vf',
 'XadIfd',
 'a']

need_diag_eps = ['vout', 'LL_y', 'vout', 'LL_y']
