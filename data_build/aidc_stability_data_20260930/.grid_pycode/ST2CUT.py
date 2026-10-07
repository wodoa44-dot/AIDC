from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "29763cd69f9958c1c55bcc8ea81d0c98"

def f_update(IN, K1, K2, L1_y, L2_y, LL1_x, LL1_y, LL2_x, LL2_y, LL3_x, WO_x, WO_y, sig, sig2):
    return (K1*sig - L1_y, K2*sig2 - L2_y, IN - WO_x, -LL1_x + WO_y, LL1_y - LL2_x, LL2_y - LL3_x, 0,)


def g_update(IN, L1_y, L2_y, LL1_LT1_z1, LL1_LT2_z1, LL1_x, LL1_y, LL2_LT1_z1, LL2_LT2_z1, LL2_x, LL2_y, LL3_LT1_z1, LL3_LT2_z1, LL3_x, LL3_y, LSMAX, LSMIN, OLIM_zi, SW2_s1, SW2_s2, SW2_s3, SW2_s4, SW2_s5, SW2_s6, SW_s1, SW_s2, SW_s3, SW_s4, SW_s5, SW_s6, SnSb, T10, T3, T4, T5, T6, T7, T8, T9, VSS_lim_zi, VSS_lim_zl, VSS_lim_zu, VSS_x, VSS_y, WO_LT_z0, WO_LT_z1, WO_x, WO_y, dv2_v, dv_v, f, f2, omega, sig, sig2, te, tm, tm0, ue, v, v2, vsout):
    return (OLIM_zi*VSS_y*ue - vsout, SW_s1*(omega - 1) + SW_s2*(f - 1) + SW_s3*te/SnSb + SW_s4*(tm - tm0) + SW_s5*v + SW_s6*dv_v - sig, SW2_s1*(omega - 1) + SW2_s2*(f2 - 1) + SW2_s3*te/SnSb + SW2_s4*(tm - tm0) + SW2_s5*v2 + SW2_s6*dv2_v - sig2, -IN + ue*(L1_y + L2_y), T3*WO_LT_z0*(IN - WO_x) + T4*WO_LT_z1*WO_x - T4*WO_y, LL1_LT1_z1*LL1_LT2_z1*(-LL1_x + LL1_y) + LL1_x*T6 - LL1_y*T6 + T5*(-LL1_x + WO_y), LL2_LT1_z1*LL2_LT2_z1*(-LL2_x + LL2_y) + LL2_x*T8 - LL2_y*T8 + T7*(LL1_y - LL2_x), LL3_LT1_z1*LL3_LT2_z1*(-LL3_x + LL3_y) + LL3_x*T10 - LL3_y*T10 + T9*(LL2_y - LL3_x), LL3_y - VSS_x, LSMAX*VSS_lim_zu + LSMIN*VSS_lim_zl + VSS_lim_zi*VSS_x - VSS_y, 0, 0, 0, 0, ue*vsout, 0, 0,)


def fx_update():
    return (-1, -1, -1, -1, -1, -1)


def fy_update(K1, K2):
    return (K1, K2, 1, 1, 1, 1)


def gy_update(LL1_LT1_z1, LL1_LT2_z1, LL2_LT1_z1, LL2_LT2_z1, LL3_LT1_z1, LL3_LT2_z1, OLIM_zi, SW2_s2, SW2_s3, SW2_s4, SW2_s5, SW_s2, SW_s3, SW_s4, SW_s5, SnSb, T10, T3, T4, T5, T6, T7, T8, T9, VSS_lim_zi, WO_LT_z0, ue):
    return (-1, OLIM_zi*ue, -1, SW_s4, SW_s3/SnSb, SW_s5, SW_s2, -1, SW2_s4, SW2_s3/SnSb, SW2_s5, SW2_s2, -1, T3*WO_LT_z0, -T4, T5, LL1_LT1_z1*LL1_LT2_z1 - T6, T7, LL2_LT1_z1*LL2_LT2_z1 - T8, T9, LL3_LT1_z1*LL3_LT2_z1 - T10, 1, -1, VSS_lim_zi, -1, ue)


def gx_update(LL1_LT1_z1, LL1_LT2_z1, LL2_LT1_z1, LL2_LT2_z1, LL3_LT1_z1, LL3_LT2_z1, SW2_s1, SW_s1, T10, T3, T4, T5, T6, T7, T8, T9, WO_LT_z0, WO_LT_z1, ue):
    return (SW_s1, SW2_s1, ue, ue, -T3*WO_LT_z0 + T4*WO_LT_z1, -LL1_LT1_z1*LL1_LT2_z1 - T5 + T6, -LL2_LT1_z1*LL2_LT2_z1 - T7 + T8, -LL3_LT1_z1*LL3_LT2_z1 + T10 - T9)


def sig_ia(SW_s1, SW_s3, SW_s4, SW_s5, SnSb, omega, tm, tm0, v):
    return SW_s1*(omega - 1) + SW_s3*tm0/SnSb + SW_s4*(tm - tm0) + SW_s5*v


def L1_y_ia(K1, sig):
    return K1*sig


def sig2_ia(SW2_s1, SW2_s3, SW2_s4, SW2_s5, SnSb, omega, tm, tm0, v2):
    return SW2_s1*(omega - 1) + SW2_s3*tm0/SnSb + SW2_s4*(tm - tm0) + SW2_s5*v2


def L2_y_ia(K2, sig2):
    return K2*sig2


def IN_ia(L1_y, L2_y, ue):
    return ue*(L1_y + L2_y)


def WO_x_ia(IN):
    return IN


def WO_y_ia(WO_LT_z1, WO_x):
    return WO_LT_z1*WO_x


def LL1_x_ia(WO_y):
    return WO_y


def LL1_y_ia(WO_y):
    return WO_y


def LL2_x_ia(LL1_y):
    return LL1_y


def LL2_y_ia(LL1_y):
    return LL1_y


def LL3_x_ia(LL2_y):
    return LL2_y


def LL3_y_ia(LL2_y):
    return LL2_y


def VSS_x_ia(LL3_y):
    return LL3_y


def VSS_y_ia(LSMAX, LSMIN, VSS_lim_zi, VSS_lim_zl, VSS_lim_zu, VSS_x):
    return LSMAX*VSS_lim_zu + LSMIN*VSS_lim_zl + VSS_lim_zi*VSS_x


def ue_svc(u, uee):
    return u*uee


def VOU_svc(VCUr, v0):
    return VCUr + v0


def VOL_svc(VCLr, v0):
    return VCLr + v0


# empty sns_update

f_args = ['IN',
 'K1',
 'K2',
 'L1_y',
 'L2_y',
 'LL1_x',
 'LL1_y',
 'LL2_x',
 'LL2_y',
 'LL3_x',
 'WO_x',
 'WO_y',
 'sig',
 'sig2']

g_args = ['IN',
 'L1_y',
 'L2_y',
 'LL1_LT1_z1',
 'LL1_LT2_z1',
 'LL1_x',
 'LL1_y',
 'LL2_LT1_z1',
 'LL2_LT2_z1',
 'LL2_x',
 'LL2_y',
 'LL3_LT1_z1',
 'LL3_LT2_z1',
 'LL3_x',
 'LL3_y',
 'LSMAX',
 'LSMIN',
 'OLIM_zi',
 'SW2_s1',
 'SW2_s2',
 'SW2_s3',
 'SW2_s4',
 'SW2_s5',
 'SW2_s6',
 'SW_s1',
 'SW_s2',
 'SW_s3',
 'SW_s4',
 'SW_s5',
 'SW_s6',
 'SnSb',
 'T10',
 'T3',
 'T4',
 'T5',
 'T6',
 'T7',
 'T8',
 'T9',
 'VSS_lim_zi',
 'VSS_lim_zl',
 'VSS_lim_zu',
 'VSS_x',
 'VSS_y',
 'WO_LT_z0',
 'WO_LT_z1',
 'WO_x',
 'WO_y',
 'dv2_v',
 'dv_v',
 'f',
 'f2',
 'omega',
 'sig',
 'sig2',
 'te',
 'tm',
 'tm0',
 'ue',
 'v',
 'v2',
 'vsout']

j_args = {'fx': [],
 'fy': ['K1', 'K2'],
 'gx': ['LL1_LT1_z1',
        'LL1_LT2_z1',
        'LL2_LT1_z1',
        'LL2_LT2_z1',
        'LL3_LT1_z1',
        'LL3_LT2_z1',
        'SW2_s1',
        'SW_s1',
        'T10',
        'T3',
        'T4',
        'T5',
        'T6',
        'T7',
        'T8',
        'T9',
        'WO_LT_z0',
        'WO_LT_z1',
        'ue'],
 'gy': ['LL1_LT1_z1',
        'LL1_LT2_z1',
        'LL2_LT1_z1',
        'LL2_LT2_z1',
        'LL3_LT1_z1',
        'LL3_LT2_z1',
        'OLIM_zi',
        'SW2_s2',
        'SW2_s3',
        'SW2_s4',
        'SW2_s5',
        'SW_s2',
        'SW_s3',
        'SW_s4',
        'SW_s5',
        'SnSb',
        'T10',
        'T3',
        'T4',
        'T5',
        'T6',
        'T7',
        'T8',
        'T9',
        'VSS_lim_zi',
        'WO_LT_z0',
        'ue']}

s_args = OrderedDict([('ue', ['u', 'uee']),
             ('VOU', ['VCUr', 'v0']),
             ('VOL', ['VCLr', 'v0'])])

sns_args = []

ia_args = OrderedDict([('sig',
              ['SW_s1',
               'SW_s3',
               'SW_s4',
               'SW_s5',
               'SnSb',
               'omega',
               'tm',
               'tm0',
               'v']),
             ('L1_y', ['K1', 'sig']),
             ('sig2',
              ['SW2_s1',
               'SW2_s3',
               'SW2_s4',
               'SW2_s5',
               'SnSb',
               'omega',
               'tm',
               'tm0',
               'v2']),
             ('L2_y', ['K2', 'sig2']),
             ('IN', ['L1_y', 'L2_y', 'ue']),
             ('WO_x', ['IN']),
             ('WO_y', ['WO_LT_z1', 'WO_x']),
             ('LL1_x', ['WO_y']),
             ('LL1_y', ['WO_y']),
             ('LL2_x', ['LL1_y']),
             ('LL2_y', ['LL1_y']),
             ('LL3_x', ['LL2_y']),
             ('LL3_y', ['LL2_y']),
             ('VSS_x', ['LL3_y']),
             ('VSS_y',
              ['LSMAX',
               'LSMIN',
               'VSS_lim_zi',
               'VSS_lim_zl',
               'VSS_lim_zu',
               'VSS_x'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 3, 4, 5]),
             ('fyc', []),
             ('fy', [0, 1, 2, 3, 4, 5]),
             ('gxc', []),
             ('gx', [1, 2, 3, 3, 4, 5, 6, 7]),
             ('gyc', [4, 5, 6, 7]),
             ('gy',
              [0,
               0,
               1,
               1,
               1,
               1,
               1,
               2,
               2,
               2,
               2,
               2,
               3,
               4,
               4,
               5,
               5,
               6,
               6,
               7,
               7,
               8,
               8,
               9,
               9,
               14])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 3, 4, 5]),
             ('fyc', []),
             ('fy', [8, 9, 10, 11, 12, 13]),
             ('gxc', []),
             ('gx', [6, 6, 0, 1, 2, 3, 4, 5]),
             ('gyc', [11, 12, 13, 14]),
             ('gy',
              [7,
               16,
               8,
               17,
               18,
               19,
               20,
               9,
               17,
               18,
               22,
               23,
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
               15,
               16,
               7])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08, 1e-08, 1e-08]),
             ('gy',
              [0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0,
               0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['omega',
 'tm',
 'v',
 'sig',
 'L1_y',
 'v2',
 'sig2',
 'L2_y',
 'IN',
 'WO_x',
 'WO_y',
 'LL1_x',
 'LL1_y',
 'LL2_x',
 'LL2_y',
 'LL3_x',
 'vsout',
 'LL3_y',
 'VSS_x',
 'VSS_y',
 'te',
 'f',
 'vi',
 'f2']

need_diag_eps = ['WO_y', 'LL1_y', 'LL2_y', 'LL3_y', 'WO_y', 'LL1_y', 'LL2_y', 'LL3_y']
