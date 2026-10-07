from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "7eeef2365cc0683e20edcf68173deafc"

def f_update(HLI_zi, HLI_zl, HLI_zu, KA, LG_y, LL_x, LL_y, LR_y, VIMAX, VIMIN, v, vi):
    return (-LG_y + v, HLI_zi*vi + HLI_zl*VIMIN + HLI_zu*VIMAX - LL_x, KA*LL_y - LR_y, 0,)


def g_update(HLI_zi, HLI_zl, HLI_zu, HLR_zi, HLR_zl, HLR_zu, KC, LG_y, LL_LT1_z1, LL_LT2_z1, LL_x, LL_y, LR_y, TB, TC, VIMAX, VIMIN, VRMAX, VRMIN, XadIfd, ue, v, vbus, vf0, vfmax, vfmin, vi, vout, vref0):
    return (-v + vbus, ue*(HLR_zi*LR_y + HLR_zl*vfmin + HLR_zu*vfmax) - vout, -LG_y - vi + vref0, LL_LT1_z1*LL_LT2_z1*(-LL_x + LL_y) + LL_x*TB - LL_y*TB + TC*(HLI_zi*vi + HLI_zl*VIMIN + HLI_zu*VIMAX - LL_x), -KC*XadIfd + VRMAX - vfmax, -KC*XadIfd + VRMIN - vfmin, ue*(-vf0 + vout), 0, 0, 0,)


def fx_update():
    return (-1, -1, -1)


def fy_update(HLI_zi, KA):
    return (1, HLI_zi, KA)


def gy_update(HLI_zi, HLR_zl, HLR_zu, KC, LL_LT1_z1, LL_LT2_z1, TB, TC, ue):
    return (-1, 1, -1, HLR_zu*ue, HLR_zl*ue, -1, HLI_zi*TC, LL_LT1_z1*LL_LT2_z1 - TB, -1, -KC, -1, -KC, ue)


def gx_update(HLR_zi, LL_LT1_z1, LL_LT2_z1, TB, TC, ue):
    return (HLR_zi*ue, -1, -LL_LT1_z1*LL_LT2_z1 + TB - TC)


def v_ia(vbus):
    return vbus


def LG_y_ia(v):
    return v


def vi_ia(KA, vf0):
    return vf0/KA


def LL_x_ia(HLI_zi, HLI_zl, HLI_zu, VIMAX, VIMIN, vi):
    return HLI_zi*vi + HLI_zl*VIMIN + HLI_zu*VIMAX


def LL_y_ia(HLI_zi, HLI_zl, HLI_zu, VIMAX, VIMIN, vi):
    return HLI_zi*vi + HLI_zl*VIMIN + HLI_zu*VIMAX


def LR_y_ia(KA, LL_y):
    return KA*LL_y


def vout_ia(ue, vf0):
    return ue*vf0


def vfmax_ia(KC, VRMAX, XadIfd):
    return -KC*XadIfd + VRMAX


def vfmin_ia(KC, VRMIN, XadIfd):
    return -KC*XadIfd + VRMIN


def ue_svc(u, ug):
    return u*ug


def vref0_svc(KA, v, vf0):
    return v + vf0/KA


# empty sns_update

f_args = ['HLI_zi',
 'HLI_zl',
 'HLI_zu',
 'KA',
 'LG_y',
 'LL_x',
 'LL_y',
 'LR_y',
 'VIMAX',
 'VIMIN',
 'v',
 'vi']

g_args = ['HLI_zi',
 'HLI_zl',
 'HLI_zu',
 'HLR_zi',
 'HLR_zl',
 'HLR_zu',
 'KC',
 'LG_y',
 'LL_LT1_z1',
 'LL_LT2_z1',
 'LL_x',
 'LL_y',
 'LR_y',
 'TB',
 'TC',
 'VIMAX',
 'VIMIN',
 'VRMAX',
 'VRMIN',
 'XadIfd',
 'ue',
 'v',
 'vbus',
 'vf0',
 'vfmax',
 'vfmin',
 'vi',
 'vout',
 'vref0']

j_args = {'fx': [],
 'fy': ['HLI_zi', 'KA'],
 'gx': ['HLR_zi', 'LL_LT1_z1', 'LL_LT2_z1', 'TB', 'TC', 'ue'],
 'gy': ['HLI_zi',
        'HLR_zl',
        'HLR_zu',
        'KC',
        'LL_LT1_z1',
        'LL_LT2_z1',
        'TB',
        'TC',
        'ue']}

s_args = OrderedDict([('ue', ['u', 'ug']), ('vref0', ['KA', 'v', 'vf0'])])

sns_args = []

ia_args = OrderedDict([('v', ['vbus']),
             ('LG_y', ['v']),
             ('vi', ['KA', 'vf0']),
             ('LL_x', ['HLI_zi', 'HLI_zl', 'HLI_zu', 'VIMAX', 'VIMIN', 'vi']),
             ('LL_y', ['HLI_zi', 'HLI_zl', 'HLI_zu', 'VIMAX', 'VIMIN', 'vi']),
             ('LR_y', ['KA', 'LL_y']),
             ('vout', ['ue', 'vf0']),
             ('vfmax', ['KC', 'VRMAX', 'XadIfd']),
             ('vfmin', ['KC', 'VRMIN', 'XadIfd'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2]),
             ('fyc', []),
             ('fy', [0, 1, 2]),
             ('gxc', []),
             ('gx', [1, 2, 3]),
             ('gyc', [1, 3]),
             ('gy', [0, 0, 1, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2]),
             ('fyc', []),
             ('fy', [4, 6, 7]),
             ('gxc', []),
             ('gx', [2, 0, 1]),
             ('gyc', [5, 7]),
             ('gy', [4, 13, 5, 8, 9, 6, 6, 7, 8, 11, 9, 11, 5])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0]),
             ('gyc', [1e-08, 1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['vbus',
 'v',
 'LG_y',
 'vi',
 'LL_x',
 'LL_y',
 'LR_y',
 'omega',
 'vout',
 'XadIfd',
 'vfmax',
 'vfmin',
 'vf',
 'a']

need_diag_eps = ['vout', 'LL_y', 'vout', 'LL_y']
