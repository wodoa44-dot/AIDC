from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "224b886b90012452dc62ddd701bcfaca"

def f_update(KA, LG_y, LL_x, LL_y, LR_y, WF_x, v, vl):
    return (-LG_y + v, -LL_x + vl, KA*LL_y - LR_y, LR_y - WF_x, 0,)


def g_update(HLI_zi, HLI_zl, HLI_zu, HLR_zi, HLR_zl, HLR_zu, KC, KF, LG_y, LL_LT1_z1, LL_LT2_z1, LL_x, LL_y, LR_y, TB, TC, TF, VIMAX, VIMIN, VRMAX, VRMIN, WF_x, WF_y, XadIfd, ue, v, vbus, vf0, vfmax, vfmin, vi, vl, vout, vref, vref0):
    return (-v + vbus, ue*(HLR_zi*LR_y + HLR_zl*vfmin + HLR_zu*vfmax) - vout, -vref + vref0, -LG_y - WF_y - vi + vref, HLI_zi*vi + HLI_zl*VIMIN + HLI_zu*VIMAX - vl, LL_LT1_z1*LL_LT2_z1*(-LL_x + LL_y) + LL_x*TB - LL_y*TB + TC*(-LL_x + vl), KF*(LR_y - WF_x) - TF*WF_y, -KC*XadIfd + VRMAX - vfmax, -KC*XadIfd + VRMIN - vfmin, ue*(-vf0 + vout), 0, 0, 0,)


def fx_update():
    return (-1, -1, -1, 1, -1)


def fy_update(KA):
    return (1, 1, KA)


def gy_update(HLI_zi, HLR_zl, HLR_zu, KC, LL_LT1_z1, LL_LT2_z1, TB, TC, TF, ue):
    return (-1, 1, -1, HLR_zu*ue, HLR_zl*ue, -1, 1, -1, -1, HLI_zi, -1, TC, LL_LT1_z1*LL_LT2_z1 - TB, -TF, -1, -KC, -1, -KC, ue)


def gx_update(HLR_zi, KF, LL_LT1_z1, LL_LT2_z1, TB, TC, ue):
    return (HLR_zi*ue, -1, -LL_LT1_z1*LL_LT2_z1 + TB - TC, KF, -KF)


def v_ia(vbus):
    return vbus


def LG_y_ia(v):
    return v


def vi_ia(KA, vf0):
    return vf0/KA


def vl_ia(HLI_zi, HLI_zl, HLI_zu, VIMAX, VIMIN, vi):
    return HLI_zi*vi + HLI_zl*VIMIN + HLI_zu*VIMAX


def LL_x_ia(vl):
    return vl


def LL_y_ia(vl):
    return vl


def LR_y_ia(KA, LL_y):
    return KA*LL_y


def WF_x_ia(LR_y):
    return LR_y


def vout_ia(ue, vf0):
    return ue*vf0


def vref_ia(KA, v, vf0):
    return v + vf0/KA


def WF_y_ia():
    return 0


def vfmax_ia(KC, VRMAX, XadIfd):
    return -KC*XadIfd + VRMAX


def vfmin_ia(KC, VRMIN, XadIfd):
    return -KC*XadIfd + VRMIN


def ue_svc(u, ug):
    return u*ug


def vref0_svc(vref):
    return vref


# empty sns_update

f_args = ['KA', 'LG_y', 'LL_x', 'LL_y', 'LR_y', 'WF_x', 'v', 'vl']

g_args = ['HLI_zi',
 'HLI_zl',
 'HLI_zu',
 'HLR_zi',
 'HLR_zl',
 'HLR_zu',
 'KC',
 'KF',
 'LG_y',
 'LL_LT1_z1',
 'LL_LT2_z1',
 'LL_x',
 'LL_y',
 'LR_y',
 'TB',
 'TC',
 'TF',
 'VIMAX',
 'VIMIN',
 'VRMAX',
 'VRMIN',
 'WF_x',
 'WF_y',
 'XadIfd',
 'ue',
 'v',
 'vbus',
 'vf0',
 'vfmax',
 'vfmin',
 'vi',
 'vl',
 'vout',
 'vref',
 'vref0']

j_args = {'fx': [],
 'fy': ['KA'],
 'gx': ['HLR_zi', 'KF', 'LL_LT1_z1', 'LL_LT2_z1', 'TB', 'TC', 'ue'],
 'gy': ['HLI_zi',
        'HLR_zl',
        'HLR_zu',
        'KC',
        'LL_LT1_z1',
        'LL_LT2_z1',
        'TB',
        'TC',
        'TF',
        'ue']}

s_args = OrderedDict([('ue', ['u', 'ug']), ('vref0', ['vref'])])

sns_args = []

ia_args = OrderedDict([('v', ['vbus']),
             ('LG_y', ['v']),
             ('vi', ['KA', 'vf0']),
             ('vl', ['HLI_zi', 'HLI_zl', 'HLI_zu', 'VIMAX', 'VIMIN', 'vi']),
             ('LL_x', ['vl']),
             ('LL_y', ['vl']),
             ('LR_y', ['KA', 'LL_y']),
             ('WF_x', ['LR_y']),
             ('vout', ['ue', 'vf0']),
             ('vref', ['KA', 'v', 'vf0']),
             ('WF_y', []),
             ('vfmax', ['KC', 'VRMAX', 'XadIfd']),
             ('vfmin', ['KC', 'VRMIN', 'XadIfd'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 3, 3]),
             ('fyc', []),
             ('fy', [0, 1, 2]),
             ('gxc', []),
             ('gx', [1, 3, 5, 6, 6]),
             ('gyc', [1, 5, 6]),
             ('gy', [0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 5, 5, 6, 7, 7, 8, 8, 9])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 2, 3]),
             ('fyc', []),
             ('fy', [5, 9, 10]),
             ('gxc', []),
             ('gx', [2, 0, 1, 2, 3]),
             ('gyc', [6, 10, 11]),
             ('gy',
              [5,
               17,
               6,
               12,
               13,
               7,
               7,
               8,
               11,
               8,
               9,
               9,
               10,
               11,
               12,
               15,
               13,
               15,
               6])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08, 1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['vbus',
 'v',
 'LG_y',
 'vi',
 'vl',
 'LL_x',
 'LL_y',
 'LR_y',
 'WF_x',
 'omega',
 'vout',
 'vref',
 'WF_y',
 'XadIfd',
 'vfmax',
 'vfmin',
 'vf',
 'a']

need_diag_eps = ['vout', 'LL_y', 'WF_y', 'vout', 'LL_y', 'WF_y']
