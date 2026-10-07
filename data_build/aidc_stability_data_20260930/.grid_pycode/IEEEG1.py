from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "893e8824f0991b1704993d9bc135555c"

def f_update(IAW_y, L4_y, L5_y, L6_y, L7_y, LL_x, vsl, wd):
    return (-LL_x + wd, vsl, IAW_y - L4_y, L4_y - L5_y, L5_y - L6_y, L6_y - L7_y, 0,)


def g_update(HL_zi, HL_zl, HL_zu, IAW_y, K, K1n, K2n, K3n, K4n, K5n, K6n, K7n, K8n, L4_y, L5_y, L6_y, L7_y, LL_LT1_z1, LL_LT2_z1, LL_x, LL_y, PHP, PLP, T1, T2, T3, UC, UO, omega, paux, paux0, pout, tm0, tm012, tm02, ue, vs, vsl, wd, wref, wref0, zsyn2):
    return (-paux + paux0, PHP*ue - pout, -wref + wref0, ue*(-omega + wref) - wd, K*LL_x*T1 + K*T2*(-LL_x + wd) + LL_LT1_z1*LL_LT2_z1*(-K*LL_x + LL_y) - LL_y*T1, -vs + ue*(-IAW_y + LL_y + paux + tm012)/T3, HL_zi*vs + HL_zl*UC + HL_zu*UO - vsl, -PHP + ue*(K1n*L4_y + K3n*L5_y + K5n*L6_y + K7n*L7_y), -PLP + ue*(K2n*L4_y + K4n*L5_y + K6n*L6_y + K8n*L7_y), ue*(pout - tm0), ue*zsyn2*(PLP - tm02),)


def fx_update():
    return (-1, 1, -1, 1, -1, 1, -1, 1, -1)


def fy_update():
    return (1, 1)


def gy_update(HL_zi, K, LL_LT1_z1, LL_LT2_z1, T1, T2, T3, ue, zsyn2):
    return (-1, -1, ue, -1, ue, -1, K*T2, LL_LT1_z1*LL_LT2_z1 - T1, ue/T3, ue/T3, -1, HL_zi, -1, -1, -1, ue, ue*zsyn2)


def gx_update(K, K1n, K2n, K3n, K4n, K5n, K6n, K7n, K8n, LL_LT1_z1, LL_LT2_z1, T1, T2, T3, ue):
    return (-ue, -K*LL_LT1_z1*LL_LT2_z1 + K*T1 - K*T2, -ue/T3, K1n*ue, K3n*ue, K5n*ue, K7n*ue, K2n*ue, K4n*ue, K6n*ue, K8n*ue)


def wd_ia():
    return 0


def LL_x_ia(wd):
    return wd


def IAW_y_ia(tm012):
    return tm012


def L4_y_ia(IAW_y):
    return IAW_y


def L5_y_ia(L4_y):
    return L4_y


def L6_y_ia(L5_y):
    return L5_y


def L7_y_ia(L6_y):
    return L6_y


def paux_ia(paux0):
    return paux0


def pout_ia(tm0, ue):
    return tm0*ue


def wref_ia(wref0):
    return wref0


def LL_y_ia(wd):
    return wd


def vs_ia():
    return 0


def vsl_ia(HL_zi, HL_zl, HL_zu, UC, UO, vs):
    return HL_zi*vs + HL_zl*UC + HL_zu*UO


def PHP_ia(K1n, K3n, K5n, K7n, L4_y, L5_y, L6_y, L7_y, ue):
    return ue*(K1n*L4_y + K3n*L5_y + K5n*L6_y + K7n*L7_y)


def PLP_ia(K2n, K4n, K6n, K8n, L4_y, L5_y, L6_y, L7_y, ue):
    return ue*(K2n*L4_y + K4n*L5_y + K6n*L6_y + K8n*L7_y)


def ue_svc(u, ug):
    return u*ug


def pref0_svc(tm0):
    return tm0


def paux0_svc():
    return 0


def _sumK18_svc(K1, K2, K3, K4, K5, K6, K7, K8):
    return K1 + K2 + K3 + K4 + K5 + K6 + K7 + K8


def _Kcoeff_svc(_sumK18):
    return _sumK18**(-1.0)


def K1n_svc(K1, _Kcoeff):
    return K1*_Kcoeff


def K2n_svc(K2, _Kcoeff):
    return K2*_Kcoeff


def K3n_svc(K3, _Kcoeff):
    return K3*_Kcoeff


def K4n_svc(K4, _Kcoeff):
    return K4*_Kcoeff


def K5n_svc(K5, _Kcoeff):
    return K5*_Kcoeff


def K6n_svc(K6, _Kcoeff):
    return K6*_Kcoeff


def K7n_svc(K7, _Kcoeff):
    return K7*_Kcoeff


def K8n_svc(K8, _Kcoeff):
    return K8*_Kcoeff


def _tm0K2_svc(K2n, K4n, K6n, K8n, tm0, zsyn2):
    return tm0*zsyn2*(K2n + K4n + K6n + K8n)


def _tm02K1_svc(K1n, K3n, K5n, K7n, tm02):
    return tm02*(K1n + K3n + K5n + K7n)


def tm012_svc(tm0, tm02):
    return tm0 + tm02


# empty sns_update

f_args = ['IAW_y', 'L4_y', 'L5_y', 'L6_y', 'L7_y', 'LL_x', 'vsl', 'wd']

g_args = ['HL_zi',
 'HL_zl',
 'HL_zu',
 'IAW_y',
 'K',
 'K1n',
 'K2n',
 'K3n',
 'K4n',
 'K5n',
 'K6n',
 'K7n',
 'K8n',
 'L4_y',
 'L5_y',
 'L6_y',
 'L7_y',
 'LL_LT1_z1',
 'LL_LT2_z1',
 'LL_x',
 'LL_y',
 'PHP',
 'PLP',
 'T1',
 'T2',
 'T3',
 'UC',
 'UO',
 'omega',
 'paux',
 'paux0',
 'pout',
 'tm0',
 'tm012',
 'tm02',
 'ue',
 'vs',
 'vsl',
 'wd',
 'wref',
 'wref0',
 'zsyn2']

j_args = {'fx': [],
 'fy': [],
 'gx': ['K',
        'K1n',
        'K2n',
        'K3n',
        'K4n',
        'K5n',
        'K6n',
        'K7n',
        'K8n',
        'LL_LT1_z1',
        'LL_LT2_z1',
        'T1',
        'T2',
        'T3',
        'ue'],
 'gy': ['HL_zi',
        'K',
        'LL_LT1_z1',
        'LL_LT2_z1',
        'T1',
        'T2',
        'T3',
        'ue',
        'zsyn2']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('pref0', ['tm0']),
             ('paux0', []),
             ('_sumK18', ['K1', 'K2', 'K3', 'K4', 'K5', 'K6', 'K7', 'K8']),
             ('_Kcoeff', ['_sumK18']),
             ('K1n', ['K1', '_Kcoeff']),
             ('K2n', ['K2', '_Kcoeff']),
             ('K3n', ['K3', '_Kcoeff']),
             ('K4n', ['K4', '_Kcoeff']),
             ('K5n', ['K5', '_Kcoeff']),
             ('K6n', ['K6', '_Kcoeff']),
             ('K7n', ['K7', '_Kcoeff']),
             ('K8n', ['K8', '_Kcoeff']),
             ('_tm0K2', ['K2n', 'K4n', 'K6n', 'K8n', 'tm0', 'zsyn2']),
             ('_tm02K1', ['K1n', 'K3n', 'K5n', 'K7n', 'tm02']),
             ('tm012', ['tm0', 'tm02'])])

sns_args = []

ia_args = OrderedDict([('wd', []),
             ('LL_x', ['wd']),
             ('IAW_y', ['tm012']),
             ('L4_y', ['IAW_y']),
             ('L5_y', ['L4_y']),
             ('L6_y', ['L5_y']),
             ('L7_y', ['L6_y']),
             ('paux', ['paux0']),
             ('pout', ['tm0', 'ue']),
             ('wref', ['wref0']),
             ('LL_y', ['wd']),
             ('vs', []),
             ('vsl', ['HL_zi', 'HL_zl', 'HL_zu', 'UC', 'UO', 'vs']),
             ('PHP',
              ['K1n',
               'K3n',
               'K5n',
               'K7n',
               'L4_y',
               'L5_y',
               'L6_y',
               'L7_y',
               'ue']),
             ('PLP',
              ['K2n',
               'K4n',
               'K6n',
               'K8n',
               'L4_y',
               'L5_y',
               'L6_y',
               'L7_y',
               'ue'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 2, 2, 3, 3, 4, 4, 5, 5]),
             ('fyc', []),
             ('fy', [0, 1]),
             ('gxc', []),
             ('gx', [3, 4, 5, 7, 7, 7, 7, 8, 8, 8, 8]),
             ('gyc', [4]),
             ('gy', [0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 5, 6, 6, 7, 8, 9, 10])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 2, 3, 3, 4, 4, 5]),
             ('fyc', []),
             ('fy', [10, 13]),
             ('gxc', []),
             ('gx', [6, 0, 1, 2, 3, 4, 5, 2, 3, 4, 5]),
             ('gyc', [11]),
             ('gy',
              [7, 8, 14, 9, 9, 10, 10, 11, 7, 11, 12, 12, 13, 14, 15, 8, 15])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             ('gyc', [1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['wd',
 'LL_x',
 'IAW_y',
 'L4_y',
 'L5_y',
 'L6_y',
 'L7_y',
 'omega',
 'paux',
 'pout',
 'wref',
 'LL_y',
 'vs',
 'vsl',
 'PHP',
 'PLP',
 'tm',
 'tm2']

need_diag_eps = ['LL_y', 'LL_y']
