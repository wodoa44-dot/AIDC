from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "2c6adb1603366534a51cea0128e0f02e"

def f_update(A1, A3, F1_x, F1_y, F2_x1, F2_x2, F2_y, LL1_x, LL1_y, LL2_x, Vks_y, WO_x, sig):
    return (-A1*F1_x - F1_y + sig, F1_x, -A3*F2_x1 + F1_y - F2_x2, F2_x1, F2_y - LL1_x, LL1_y - LL2_x, Vks_y - WO_x, 0,)


def g_update(A3, A4, A5, A6, F1_y, F2_LT1_z1, F2_LT2_z1, F2_LT3_z1, F2_LT4_z1, F2_x1, F2_x2, F2_y, KS, LL1_LT1_z1, LL1_LT2_z1, LL1_x, LL1_y, LL2_LT1_z1, LL2_LT2_z1, LL2_x, LL2_y, LSMAX, LSMIN, OLIM_zi, SW_s1, SW_s2, SW_s3, SW_s4, SW_s5, SW_s6, SnSb, T1, T2, T3, T4, T5, T6, VLIM_zi, VLIM_zl, VLIM_zu, Vks_y, Vss, WO_LT_z0, WO_LT_z1, WO_x, WO_y, dv_v, f, omega, sig, te, tm, tm0, ue, v, vsout):
    return (OLIM_zi*Vss*ue - vsout, -sig + ue*(SW_s1*(omega - 1) + SW_s2*(f - 1) + SW_s3*te/SnSb + SW_s4*(tm - tm0) + SW_s5*v + SW_s6*dv_v), A4*A5*F2_x1 + A4*F2_x2 - A4*F2_y + A6*(-A3*F2_x1 + F1_y - F2_x2) + F2_LT1_z1*F2_LT2_z1*F2_LT3_z1*F2_LT4_z1*(-F2_x2 + F2_y), LL1_LT1_z1*LL1_LT2_z1*(-LL1_x + LL1_y) + LL1_x*T2 - LL1_y*T2 + T1*(F2_y - LL1_x), LL2_LT1_z1*LL2_LT2_z1*(-LL2_x + LL2_y) + LL2_x*T4 - LL2_y*T4 + T3*(LL1_y - LL2_x), KS*LL2_y - Vks_y, T5*WO_LT_z0*(Vks_y - WO_x) + T6*WO_LT_z1*WO_x - T6*WO_y, LSMAX*VLIM_zu + LSMIN*VLIM_zl + VLIM_zi*WO_y - Vss, 0, 0, 0, 0, ue*vsout,)


def fx_update(A1, A3):
    return (-A1, -1, 1, 1, -A3, -1, 1, -1, -1, -1)


def fy_update():
    return (1, 1, 1, 1)


def gy_update(A4, F2_LT1_z1, F2_LT2_z1, F2_LT3_z1, F2_LT4_z1, KS, LL1_LT1_z1, LL1_LT2_z1, LL2_LT1_z1, LL2_LT2_z1, OLIM_zi, SW_s2, SW_s3, SW_s4, SW_s5, SnSb, T1, T2, T3, T4, T5, T6, VLIM_zi, WO_LT_z0, ue):
    return (-1, OLIM_zi*ue, -1, SW_s4*ue, SW_s3*ue/SnSb, SW_s5*ue, SW_s2*ue, -A4 + F2_LT1_z1*F2_LT2_z1*F2_LT3_z1*F2_LT4_z1, T1, LL1_LT1_z1*LL1_LT2_z1 - T2, T3, LL2_LT1_z1*LL2_LT2_z1 - T4, KS, -1, T5*WO_LT_z0, -T6, VLIM_zi, -1, ue)


def gx_update(A3, A4, A5, A6, F2_LT1_z1, F2_LT2_z1, F2_LT3_z1, F2_LT4_z1, LL1_LT1_z1, LL1_LT2_z1, LL2_LT1_z1, LL2_LT2_z1, SW_s1, T1, T2, T3, T4, T5, T6, WO_LT_z0, WO_LT_z1, ue):
    return (SW_s1*ue, A6, -A3*A6 + A4*A5, A4 - A6 - F2_LT1_z1*F2_LT2_z1*F2_LT3_z1*F2_LT4_z1, -LL1_LT1_z1*LL1_LT2_z1 - T1 + T2, -LL2_LT1_z1*LL2_LT2_z1 - T3 + T4, -T5*WO_LT_z0 + T6*WO_LT_z1)


def F1_x_ia():
    return 0


def sig_ia(SW_s1, SW_s3, SW_s4, SW_s5, SnSb, omega, tm, tm0, ue, v):
    return ue*(SW_s1*(omega - 1) + SW_s3*tm0/SnSb + SW_s4*(tm - tm0) + SW_s5*v)


def F1_y_ia(sig):
    return sig


def F2_x1_ia():
    return 0


def F2_x2_ia(F1_y):
    return F1_y


def F2_y_ia(F1_y):
    return F1_y


def LL1_x_ia(F2_y):
    return F2_y


def LL1_y_ia(F2_y):
    return F2_y


def LL2_x_ia(LL1_y):
    return LL1_y


def LL2_y_ia(LL1_y):
    return LL1_y


def Vks_y_ia(KS, LL2_y):
    return KS*LL2_y


def WO_x_ia(Vks_y):
    return Vks_y


def WO_y_ia(WO_LT_z1, WO_x):
    return WO_LT_z1*WO_x


def ue_svc(u, uee):
    return u*uee


# empty sns_update

f_args = ['A1',
 'A3',
 'F1_x',
 'F1_y',
 'F2_x1',
 'F2_x2',
 'F2_y',
 'LL1_x',
 'LL1_y',
 'LL2_x',
 'Vks_y',
 'WO_x',
 'sig']

g_args = ['A3',
 'A4',
 'A5',
 'A6',
 'F1_y',
 'F2_LT1_z1',
 'F2_LT2_z1',
 'F2_LT3_z1',
 'F2_LT4_z1',
 'F2_x1',
 'F2_x2',
 'F2_y',
 'KS',
 'LL1_LT1_z1',
 'LL1_LT2_z1',
 'LL1_x',
 'LL1_y',
 'LL2_LT1_z1',
 'LL2_LT2_z1',
 'LL2_x',
 'LL2_y',
 'LSMAX',
 'LSMIN',
 'OLIM_zi',
 'SW_s1',
 'SW_s2',
 'SW_s3',
 'SW_s4',
 'SW_s5',
 'SW_s6',
 'SnSb',
 'T1',
 'T2',
 'T3',
 'T4',
 'T5',
 'T6',
 'VLIM_zi',
 'VLIM_zl',
 'VLIM_zu',
 'Vks_y',
 'Vss',
 'WO_LT_z0',
 'WO_LT_z1',
 'WO_x',
 'WO_y',
 'dv_v',
 'f',
 'omega',
 'sig',
 'te',
 'tm',
 'tm0',
 'ue',
 'v',
 'vsout']

j_args = {'fx': ['A1', 'A3'],
 'fy': [],
 'gx': ['A3',
        'A4',
        'A5',
        'A6',
        'F2_LT1_z1',
        'F2_LT2_z1',
        'F2_LT3_z1',
        'F2_LT4_z1',
        'LL1_LT1_z1',
        'LL1_LT2_z1',
        'LL2_LT1_z1',
        'LL2_LT2_z1',
        'SW_s1',
        'T1',
        'T2',
        'T3',
        'T4',
        'T5',
        'T6',
        'WO_LT_z0',
        'WO_LT_z1',
        'ue'],
 'gy': ['A4',
        'F2_LT1_z1',
        'F2_LT2_z1',
        'F2_LT3_z1',
        'F2_LT4_z1',
        'KS',
        'LL1_LT1_z1',
        'LL1_LT2_z1',
        'LL2_LT1_z1',
        'LL2_LT2_z1',
        'OLIM_zi',
        'SW_s2',
        'SW_s3',
        'SW_s4',
        'SW_s5',
        'SnSb',
        'T1',
        'T2',
        'T3',
        'T4',
        'T5',
        'T6',
        'VLIM_zi',
        'WO_LT_z0',
        'ue']}

s_args = OrderedDict([('ue', ['u', 'uee'])])

sns_args = []

ia_args = OrderedDict([('F1_x', []),
             ('sig',
              ['SW_s1',
               'SW_s3',
               'SW_s4',
               'SW_s5',
               'SnSb',
               'omega',
               'tm',
               'tm0',
               'ue',
               'v']),
             ('F1_y', ['sig']),
             ('F2_x1', []),
             ('F2_x2', ['F1_y']),
             ('F2_y', ['F1_y']),
             ('LL1_x', ['F2_y']),
             ('LL1_y', ['F2_y']),
             ('LL2_x', ['LL1_y']),
             ('LL2_y', ['LL1_y']),
             ('Vks_y', ['KS', 'LL2_y']),
             ('WO_x', ['Vks_y']),
             ('WO_y', ['WO_LT_z1', 'WO_x'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 1, 2, 2, 2, 3, 4, 5, 6]),
             ('fyc', []),
             ('fy', [0, 4, 5, 6]),
             ('gxc', []),
             ('gx', [1, 2, 2, 2, 3, 4, 6]),
             ('gyc', [2, 3, 4, 6]),
             ('gy',
              [0, 0, 1, 1, 1, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 12])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 0, 1, 2, 3, 2, 4, 5, 6]),
             ('fyc', []),
             ('fy', [9, 10, 11, 13]),
             ('gxc', []),
             ('gx', [7, 1, 2, 3, 4, 5, 6]),
             ('gyc', [10, 11, 12, 14]),
             ('gy',
              [8,
               15,
               9,
               16,
               17,
               18,
               19,
               10,
               10,
               11,
               11,
               12,
               12,
               13,
               13,
               14,
               14,
               15,
               8])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08, 1e-08, 1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['F1_x',
 'omega',
 'tm',
 'v',
 'sig',
 'F1_y',
 'F2_x1',
 'F2_x2',
 'F2_y',
 'LL1_x',
 'LL1_y',
 'LL2_x',
 'LL2_y',
 'Vks_y',
 'WO_x',
 'vsout',
 'WO_y',
 'Vss',
 'te',
 'f',
 'vi']

need_diag_eps = ['F2_y', 'LL1_y', 'LL2_y', 'WO_y', 'F2_y', 'LL1_y', 'LL2_y', 'WO_y']
