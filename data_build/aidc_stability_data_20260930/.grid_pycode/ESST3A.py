from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "8bcc5817b61ac22a9b12dfb304f12bf8"

def f_update(HG_y, KA, KM, LAW1_y, LAW2_y, LG_y, LL_x, LL_y, v, vrs):
    return (-LG_y + v, HG_y - LL_x, KA*LL_y - LAW1_y, KM*vrs - LAW2_y, 0,)


def g_update(FEX_y, HG_lt_z0, HG_lt_z1, HG_y, HLI_zi, HLI_zl, HLI_zu, IN, KC, KG, LAW1_y, LAW2_y, LG_y, LL_LT1_z1, LL_LT2_z1, LL_x, LL_y, TB, TC, UEL, UEL0, VBMAX, VB_lim_zi, VB_lim_zu, VB_x, VB_y, VE, VGMAX, VG_lim_zi, VG_lim_zu, VG_x, VG_y, VIMAX, VIMIN, XadIfd, ue, v, vbus, vf0, vi, vil, vout, vref, vref0, vrs, __zeros, __ones, __falses, __trues):
    return (-v + vbus, LAW2_y*VB_y*ue - vout, -UEL + UEL0, ue*(-IN*VE + KC*XadIfd), -FEX_y + select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),greater(IN, 1),__trues], [__ones,1 - 0.577*IN,sqrt(0.75 - IN**2),1.732 - 1.732*IN,__zeros,__zeros], default=nan), FEX_y*VE - VB_x, VBMAX*VB_lim_zu + VB_lim_zi*VB_x - VB_y, KG*vout - VG_x, VGMAX*VG_lim_zu + VG_lim_zi*VG_x - VG_y, LAW1_y - VG_y - vrs, -vref + vref0, -LG_y - vi + vref, HLI_zi*vi + HLI_zl*VIMIN + HLI_zu*VIMAX - vil, HG_lt_z0*UEL + HG_lt_z1*vil - HG_y, LL_LT1_z1*LL_LT2_z1*(-LL_x + LL_y) + LL_x*TB - LL_y*TB + TC*(HG_y - LL_x), ue*(-vf0 + vout), 0, 0, 0, 0, 0, 0, 0,)


def fx_update():
    return (-1, -1, -1, -1)


def fy_update(KA, KM):
    return (1, 1, KA, KM)


def gy_update(HG_lt_z0, HG_lt_z1, HLI_zi, IN, KC, KG, LAW2_y, LL_LT1_z1, LL_LT2_z1, TB, TC, VB_lim_zi, VE, VG_lim_zi, ue, __zeros, __ones, __falses, __trues):
    return (-1, 1, -1, LAW2_y*ue, -1, -VE*ue, KC*ue, select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),__trues], [__zeros,-0.577*__ones,-IN/sqrt(0.75 - IN**2),-1.732*__ones,__zeros], default=nan), -1, VE, -1, VB_lim_zi, -1, KG, -1, VG_lim_zi, -1, -1, -1, -1, 1, -1, HLI_zi, -1, HG_lt_z0, HG_lt_z1, -1, TC, LL_LT1_z1*LL_LT2_z1 - TB, ue)


def gx_update(LL_LT1_z1, LL_LT2_z1, TB, TC, VB_y, ue):
    return (VB_y*ue, 1, -1, -LL_LT1_z1*LL_LT2_z1 + TB - TC)


def v_ia(vbus):
    return vbus


def LG_y_ia(v):
    return v


def UEL_ia(UEL0):
    return UEL0


def vout_ia(ue, vf0):
    return ue*vf0


def VG_x_ia(KG, vout):
    return KG*vout


def VG_y_ia(VGMAX, VG_lim_zi, VG_lim_zu, VG_x):
    return VGMAX*VG_lim_zu + VG_lim_zi*VG_x


def IN_ia(KC, VE, XadIfd):
    return safe_div(KC*XadIfd, VE)


def FEX_y_ia(IN, __zeros, __ones, __falses, __trues):
    return select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),greater(IN, 1),__trues], [__ones,1 - 0.577*IN,sqrt(0.75 - IN**2),1.732 - 1.732*IN,__zeros,__zeros], default=nan)


def VB_x_ia(FEX_y, VE):
    return FEX_y*VE


def VB_y_ia(VBMAX, VB_lim_zi, VB_lim_zu, VB_x):
    return VBMAX*VB_lim_zu + VB_lim_zi*VB_x


def vrs_ia(KM, VB_y, vf0):
    return safe_div(vf0, VB_y)/KM


def vref_ia(KA, VG_y, v, vrs):
    return v + (VG_y + vrs)/KA


def vi_ia(v, vref):
    return -v + vref


def vil_ia(HLI_zi, HLI_zl, HLI_zu, VIMAX, VIMIN, vi):
    return HLI_zi*vi + HLI_zl*VIMIN + HLI_zu*VIMAX


def HG_y_ia(HG_lt_z0, HG_lt_z1, UEL, vil):
    return HG_lt_z0*UEL + HG_lt_z1*vil


def LL_x_ia(HG_y):
    return HG_y


def LL_y_ia(HG_y):
    return HG_y


def LAW1_y_ia(KA, LL_y):
    return KA*LL_y


def LAW2_y_ia(KM, vrs):
    return KM*vrs


def ue_svc(u, ug):
    return u*ug


def KPC_svc(KP, THETAP):
    return KP*exp(1j*radians(THETAP))


def UEL0_svc():
    return -9999


def VE_svc(Id, Iq, KI, KPC, XL, vd, vq):
    return abs(KPC*(vd + 1j*vq) + 1j*(Id + 1j*Iq)*(KI + KPC*XL))


def vref0_svc(vref):
    return vref


# empty sns_update

f_args = ['HG_y', 'KA', 'KM', 'LAW1_y', 'LAW2_y', 'LG_y', 'LL_x', 'LL_y', 'v', 'vrs']

g_args = ['FEX_y',
 'HG_lt_z0',
 'HG_lt_z1',
 'HG_y',
 'HLI_zi',
 'HLI_zl',
 'HLI_zu',
 'IN',
 'KC',
 'KG',
 'LAW1_y',
 'LAW2_y',
 'LG_y',
 'LL_LT1_z1',
 'LL_LT2_z1',
 'LL_x',
 'LL_y',
 'TB',
 'TC',
 'UEL',
 'UEL0',
 'VBMAX',
 'VB_lim_zi',
 'VB_lim_zu',
 'VB_x',
 'VB_y',
 'VE',
 'VGMAX',
 'VG_lim_zi',
 'VG_lim_zu',
 'VG_x',
 'VG_y',
 'VIMAX',
 'VIMIN',
 'XadIfd',
 'ue',
 'v',
 'vbus',
 'vf0',
 'vi',
 'vil',
 'vout',
 'vref',
 'vref0',
 'vrs',
 '__zeros',
 '__ones',
 '__falses',
 '__trues']

j_args = {'fx': [],
 'fy': ['KA', 'KM'],
 'gx': ['LL_LT1_z1', 'LL_LT2_z1', 'TB', 'TC', 'VB_y', 'ue'],
 'gy': ['HG_lt_z0',
        'HG_lt_z1',
        'HLI_zi',
        'IN',
        'KC',
        'KG',
        'LAW2_y',
        'LL_LT1_z1',
        'LL_LT2_z1',
        'TB',
        'TC',
        'VB_lim_zi',
        'VE',
        'VG_lim_zi',
        'ue',
        '__zeros',
        '__ones',
        '__falses',
        '__trues']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('KPC', ['KP', 'THETAP']),
             ('UEL0', []),
             ('VE', ['Id', 'Iq', 'KI', 'KPC', 'XL', 'vd', 'vq']),
             ('vref0', ['vref'])])

sns_args = []

ia_args = OrderedDict([('v', ['vbus']),
             ('LG_y', ['v']),
             ('UEL', ['UEL0']),
             ('vout', ['ue', 'vf0']),
             ('VG_x', ['KG', 'vout']),
             ('VG_y', ['VGMAX', 'VG_lim_zi', 'VG_lim_zu', 'VG_x']),
             ('IN', ['KC', 'VE', 'XadIfd']),
             ('FEX_y', ['IN', '__zeros', '__ones', '__falses', '__trues']),
             ('VB_x', ['FEX_y', 'VE']),
             ('VB_y', ['VBMAX', 'VB_lim_zi', 'VB_lim_zu', 'VB_x']),
             ('vrs', ['KM', 'VB_y', 'vf0']),
             ('vref', ['KA', 'VG_y', 'v', 'vrs']),
             ('vi', ['v', 'vref']),
             ('vil', ['HLI_zi', 'HLI_zl', 'HLI_zu', 'VIMAX', 'VIMIN', 'vi']),
             ('HG_y', ['HG_lt_z0', 'HG_lt_z1', 'UEL', 'vil']),
             ('LL_x', ['HG_y']),
             ('LL_y', ['HG_y']),
             ('LAW1_y', ['KA', 'LL_y']),
             ('LAW2_y', ['KM', 'vrs'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 3]),
             ('fyc', []),
             ('fy', [0, 1, 2, 3]),
             ('gxc', []),
             ('gx', [1, 9, 11, 14]),
             ('gyc', [1, 3, 14]),
             ('gy',
              [0,
               0,
               1,
               1,
               2,
               3,
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
               10,
               11,
               11,
               12,
               12,
               13,
               13,
               13,
               14,
               14,
               15])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 3]),
             ('fyc', []),
             ('fy', [5, 18, 19, 14]),
             ('gxc', []),
             ('gx', [3, 2, 0, 1]),
             ('gyc', [6, 8, 19]),
             ('gy',
              [5,
               23,
               6,
               11,
               7,
               8,
               21,
               8,
               9,
               9,
               10,
               10,
               11,
               6,
               12,
               12,
               13,
               13,
               14,
               15,
               15,
               16,
               16,
               17,
               7,
               17,
               18,
               18,
               19,
               6])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08, 1e-08]),
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
               0,
               0,
               0,
               0,
               0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['vbus',
 'v',
 'LG_y',
 'UEL',
 'vout',
 'VG_x',
 'VG_y',
 'XadIfd',
 'IN',
 'FEX_y',
 'VB_x',
 'VB_y',
 'vrs',
 'vref',
 'vi',
 'vil',
 'HG_y',
 'LL_x',
 'LL_y',
 'LAW1_y',
 'LAW2_y',
 'omega',
 'vf',
 'a',
 'vd',
 'vq',
 'Id',
 'Iq']

need_diag_eps = ['vout', 'IN', 'LL_y', 'vout', 'IN', 'LL_y']
