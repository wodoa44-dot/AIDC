from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "3d925525f20a3f73aa826e9ed1a0f411"

def f_update(LAG_y, LL_x, pd):
    return (-LAG_y + pd, LAG_y - LL_x, 0,)


def g_update(Dt, LAG_y, LL_LT1_z1, LL_LT2_z1, LL_x, LL_y, R, T2, T3, gain, omega, paux, paux0, pd, pout, pref, pref0, tm0, ue, wd, wref, wref0):
    return (-paux + paux0, -pout + ue*(-Dt*wd + LL_y), -wref + wref0, R*pref0 - pref, ue*(omega - wref) - wd, gain*ue*(paux + pref - wd) - pd, LL_LT1_z1*LL_LT2_z1*(-LL_x + LL_y) + LL_x*T3 - LL_y*T3 + T2*(LAG_y - LL_x), ue*(pout - tm0),)


def fx_update():
    return (-1, 1, -1)


def fy_update():
    return (1,)


def gy_update(Dt, LL_LT1_z1, LL_LT2_z1, T3, gain, ue):
    return (-1, -1, -Dt*ue, ue, -1, -1, -ue, -1, gain*ue, gain*ue, -gain*ue, -1, LL_LT1_z1*LL_LT2_z1 - T3, ue)


def gx_update(LL_LT1_z1, LL_LT2_z1, T2, T3, ue):
    return (ue, T2, -LL_LT1_z1*LL_LT2_z1 - T2 + T3)


def pd_ia(tm0, ue):
    return tm0*ue


def LAG_y_ia(pd):
    return pd


def LL_x_ia(LAG_y):
    return LAG_y


def paux_ia(paux0):
    return paux0


def pout_ia(tm0, ue):
    return tm0*ue


def wref_ia(wref0):
    return wref0


def pref_ia(R, tm0):
    return R*tm0


def wd_ia():
    return 0


def LL_y_ia(LAG_y):
    return LAG_y


def ue_svc(u, ug):
    return u*ug


def pref0_svc(tm0):
    return tm0


def paux0_svc():
    return 0


def gain_svc(R, ue):
    return ue/R


# empty sns_update

f_args = ['LAG_y', 'LL_x', 'pd']

g_args = ['Dt',
 'LAG_y',
 'LL_LT1_z1',
 'LL_LT2_z1',
 'LL_x',
 'LL_y',
 'R',
 'T2',
 'T3',
 'gain',
 'omega',
 'paux',
 'paux0',
 'pd',
 'pout',
 'pref',
 'pref0',
 'tm0',
 'ue',
 'wd',
 'wref',
 'wref0']

j_args = {'fx': [],
 'fy': [],
 'gx': ['LL_LT1_z1', 'LL_LT2_z1', 'T2', 'T3', 'ue'],
 'gy': ['Dt', 'LL_LT1_z1', 'LL_LT2_z1', 'T3', 'gain', 'ue']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('pref0', ['tm0']),
             ('paux0', []),
             ('gain', ['R', 'ue'])])

sns_args = []

ia_args = OrderedDict([('pd', ['tm0', 'ue']),
             ('LAG_y', ['pd']),
             ('LL_x', ['LAG_y']),
             ('paux', ['paux0']),
             ('pout', ['tm0', 'ue']),
             ('wref', ['wref0']),
             ('pref', ['R', 'tm0']),
             ('wd', []),
             ('LL_y', ['LAG_y'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 1]),
             ('fyc', []),
             ('fy', [0]),
             ('gxc', []),
             ('gx', [4, 6, 6]),
             ('gyc', [6]),
             ('gy', [0, 1, 1, 1, 2, 3, 4, 4, 5, 5, 5, 5, 6, 7])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 1]),
             ('fyc', []),
             ('fy', [8]),
             ('gxc', []),
             ('gx', [2, 0, 1]),
             ('gyc', [9]),
             ('gy', [3, 4, 7, 9, 5, 6, 5, 7, 3, 6, 7, 8, 9, 4])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0]),
             ('fyc', []),
             ('fy', [0]),
             ('gxc', []),
             ('gx', [0, 0, 0]),
             ('gyc', [1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['pd',
 'LAG_y',
 'LL_x',
 'omega',
 'paux',
 'pout',
 'wref',
 'pref',
 'wd',
 'LL_y',
 'tm']

need_diag_eps = ['LL_y', 'LL_y']
