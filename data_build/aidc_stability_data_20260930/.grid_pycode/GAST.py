from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "3abf9245b3afc813bb397936690e5503"

def f_update(LAG_y, LG2_y, LG3_y, LVG_y):
    return (-LAG_y + LVG_y, LAG_y - LG2_y, LG2_y - LG3_y, 0,)


def g_update(AT, Dt, KT, LG2_y, LG3_y, LVG_lt_z0, LVG_lt_z1, LVG_y, R, gain, omega, paux, paux0, pd, pout, pref, pref0, tm0, ue, v9, wd, wref, wref0):
    return (-paux + paux0, -pout + ue*(-Dt*wd + LG2_y), -wref + wref0, R*pref0 - pref, ue*(omega - wref) - wd, gain*ue*(paux + pref - wd) - pd, ue*(AT + KT*(AT - LG3_y)) - v9, LVG_lt_z0*v9 + LVG_lt_z1*pd - LVG_y, ue*(pout - tm0),)


def fx_update():
    return (-1, 1, -1, 1, -1)


def fy_update():
    return (1,)


def gy_update(Dt, LVG_lt_z0, LVG_lt_z1, gain, ue):
    return (-1, -1, -Dt*ue, -1, -1, -ue, -1, gain*ue, gain*ue, -gain*ue, -1, -1, LVG_lt_z1, LVG_lt_z0, -1, ue)


def gx_update(KT, ue):
    return (ue, ue, -KT*ue)


def pd_ia(tm0, ue):
    return tm0*ue


def v9_ia(AT, KT, tm0, ue):
    return ue*(AT + KT*(AT - tm0))


def LVG_y_ia(LVG_lt_z0, LVG_lt_z1, pd, v9):
    return LVG_lt_z0*v9 + LVG_lt_z1*pd


def LAG_y_ia(LVG_y):
    return LVG_y


def LG2_y_ia(LAG_y):
    return LAG_y


def LG3_y_ia(LG2_y):
    return LG2_y


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


def ue_svc(u, ug):
    return u*ug


def pref0_svc(tm0):
    return tm0


def paux0_svc():
    return 0


def gain_svc(R, ue):
    return ue/R


# empty sns_update

f_args = ['LAG_y', 'LG2_y', 'LG3_y', 'LVG_y']

g_args = ['AT',
 'Dt',
 'KT',
 'LG2_y',
 'LG3_y',
 'LVG_lt_z0',
 'LVG_lt_z1',
 'LVG_y',
 'R',
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
 'v9',
 'wd',
 'wref',
 'wref0']

j_args = {'fx': [],
 'fy': [],
 'gx': ['KT', 'ue'],
 'gy': ['Dt', 'LVG_lt_z0', 'LVG_lt_z1', 'gain', 'ue']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('pref0', ['tm0']),
             ('paux0', []),
             ('gain', ['R', 'ue'])])

sns_args = []

ia_args = OrderedDict([('pd', ['tm0', 'ue']),
             ('v9', ['AT', 'KT', 'tm0', 'ue']),
             ('LVG_y', ['LVG_lt_z0', 'LVG_lt_z1', 'pd', 'v9']),
             ('LAG_y', ['LVG_y']),
             ('LG2_y', ['LAG_y']),
             ('LG3_y', ['LG2_y']),
             ('paux', ['paux0']),
             ('pout', ['tm0', 'ue']),
             ('wref', ['wref0']),
             ('pref', ['R', 'tm0']),
             ('wd', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 1, 2, 2]),
             ('fyc', []),
             ('fy', [0]),
             ('gxc', []),
             ('gx', [1, 4, 6]),
             ('gyc', []),
             ('gy', [0, 1, 1, 2, 3, 4, 4, 5, 5, 5, 5, 6, 7, 7, 7, 8])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 1, 1, 2]),
             ('fyc', []),
             ('fy', [11]),
             ('gxc', []),
             ('gx', [1, 3, 2]),
             ('gyc', []),
             ('gy', [4, 5, 8, 6, 7, 6, 8, 4, 7, 8, 9, 10, 9, 10, 11, 5])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0]),
             ('gxc', []),
             ('gx', [0, 0, 0]),
             ('gyc', []),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['pd',
 'v9',
 'LVG_y',
 'LAG_y',
 'LG2_y',
 'LG3_y',
 'omega',
 'paux',
 'pout',
 'wref',
 'pref',
 'wd',
 'tm']

need_diag_eps = []
