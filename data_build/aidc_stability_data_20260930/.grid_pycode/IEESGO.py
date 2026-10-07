from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "0e4da2cea38c0b100ead69d12bb72284"

def f_update(F1_y, F2_x, F3_y, F4_y, F5_y, HL_y, K1, K2, K3, omega, ue, wref):
    return (-F1_y + K1*ue*(omega - wref), F1_y - F2_x, -F3_y + 1.0*HL_y, F3_y*K2 - F4_y, F4_y*K3 - F5_y, 0,)


def g_update(F1_y, F2_LT1_z1, F2_LT2_z1, F2_x, F2_y, F3_y, F4_y, F5_y, HL_lim_zi, HL_lim_zl, HL_lim_zu, HL_x, HL_y, K2, K3, PMAX, PMIN, T2, T3, paux, paux0, pout, pref0, tm0, ue, wref, wref0):
    return (-paux + paux0, -pout + ue*(F3_y*(1 - K2) + F4_y*(1 - K3) + F5_y), -wref + wref0, F2_LT1_z1*F2_LT2_z1*(-1.0*F2_x + F2_y) + 1.0*F2_x*T3 - F2_y*T3 + 1.0*T2*(F1_y - F2_x), -HL_x + 1.0*ue*(-F2_y + paux + pref0), 1.0*HL_lim_zi*HL_x + 1.0*HL_lim_zl*PMIN + 1.0*HL_lim_zu*PMAX - HL_y, ue*(pout - tm0),)


def fx_update(K1, K2, K3, ue):
    return (-1, K1*ue, 1, -1, -1, K2, -1, K3, -1)


def fy_update(K1, ue):
    return (-K1*ue, 1.0)


def gy_update(F2_LT1_z1, F2_LT2_z1, HL_lim_zi, T3, ue):
    return (-1, -1, -1, F2_LT1_z1*F2_LT2_z1 - T3, 1.0*ue, -1.0*ue, -1, 1.0*HL_lim_zi, -1, ue)


def gx_update(F2_LT1_z1, F2_LT2_z1, K2, K3, T2, T3, ue):
    return (ue*(1 - K2), ue*(1 - K3), ue, 1.0*T2, -1.0*F2_LT1_z1*F2_LT2_z1 - 1.0*T2 + 1.0*T3)


def wref_ia(wref0):
    return wref0


def F1_y_ia(K1, omega, ue, wref):
    return K1*ue*(omega - wref)


def F2_x_ia(F1_y):
    return F1_y


def F2_y_ia(F1_y):
    return F1_y


def paux_ia(paux0):
    return paux0


def HL_x_ia(F2_y, paux, pref0, ue):
    return 1.0*ue*(-F2_y + paux + pref0)


def HL_y_ia(HL_lim_zi, HL_lim_zl, HL_lim_zu, HL_x, PMAX, PMIN):
    return 1.0*HL_lim_zi*HL_x + 1.0*HL_lim_zl*PMIN + 1.0*HL_lim_zu*PMAX


def F3_y_ia(HL_y):
    return 1.0*HL_y


def F4_y_ia(F3_y, K2):
    return F3_y*K2


def F5_y_ia(F4_y, K3):
    return F4_y*K3


def pout_ia(tm0, ue):
    return tm0*ue


def ue_svc(u, ug):
    return u*ug


def pref0_svc(tm0):
    return tm0


def paux0_svc():
    return 0


# empty sns_update

f_args = ['F1_y',
 'F2_x',
 'F3_y',
 'F4_y',
 'F5_y',
 'HL_y',
 'K1',
 'K2',
 'K3',
 'omega',
 'ue',
 'wref']

g_args = ['F1_y',
 'F2_LT1_z1',
 'F2_LT2_z1',
 'F2_x',
 'F2_y',
 'F3_y',
 'F4_y',
 'F5_y',
 'HL_lim_zi',
 'HL_lim_zl',
 'HL_lim_zu',
 'HL_x',
 'HL_y',
 'K2',
 'K3',
 'PMAX',
 'PMIN',
 'T2',
 'T3',
 'paux',
 'paux0',
 'pout',
 'pref0',
 'tm0',
 'ue',
 'wref',
 'wref0']

j_args = {'fx': ['K1', 'K2', 'K3', 'ue'],
 'fy': ['K1', 'ue'],
 'gx': ['F2_LT1_z1', 'F2_LT2_z1', 'K2', 'K3', 'T2', 'T3', 'ue'],
 'gy': ['F2_LT1_z1', 'F2_LT2_z1', 'HL_lim_zi', 'T3', 'ue']}

s_args = OrderedDict([('ue', ['u', 'ug']), ('pref0', ['tm0']), ('paux0', [])])

sns_args = []

ia_args = OrderedDict([('wref', ['wref0']),
             ('F1_y', ['K1', 'omega', 'ue', 'wref']),
             ('F2_x', ['F1_y']),
             ('F2_y', ['F1_y']),
             ('paux', ['paux0']),
             ('HL_x', ['F2_y', 'paux', 'pref0', 'ue']),
             ('HL_y',
              ['HL_lim_zi', 'HL_lim_zl', 'HL_lim_zu', 'HL_x', 'PMAX', 'PMIN']),
             ('F3_y', ['HL_y']),
             ('F4_y', ['F3_y', 'K2']),
             ('F5_y', ['F4_y', 'K3']),
             ('pout', ['tm0', 'ue'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 1, 1, 2, 3, 3, 4, 4]),
             ('fyc', []),
             ('fy', [0, 2]),
             ('gxc', []),
             ('gx', [1, 1, 1, 3, 3]),
             ('gyc', [3]),
             ('gy', [0, 1, 2, 3, 4, 4, 4, 5, 5, 6])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 5, 0, 1, 2, 2, 3, 3, 4]),
             ('fyc', []),
             ('fy', [8, 11]),
             ('gxc', []),
             ('gx', [2, 3, 4, 0, 1]),
             ('gyc', [9]),
             ('gy', [6, 7, 8, 9, 6, 9, 10, 10, 11, 7])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0]),
             ('gyc', [1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['omega',
 'wref',
 'F1_y',
 'F2_x',
 'F2_y',
 'paux',
 'HL_x',
 'HL_y',
 'F3_y',
 'F4_y',
 'F5_y',
 'pout',
 'tm']

need_diag_eps = ['F2_y', 'F2_y']
