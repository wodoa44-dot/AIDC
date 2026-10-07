from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "e2af1ad4874da83901529923b3b064e5"

def f_update(HVG1_y, KA, LA_y, LG_y, LL1_x, LL1_y, LL_x, LL_y, LVG_y, WF_x, v):
    return (-LG_y + v, HVG1_y - LL_x, -LL1_x + LL_y, KA*LL1_y - LA_y, LVG_y - WF_x, 0,)


def g_update(HVG1_lt_z0, HVG1_lt_z1, HVG1_y, HVG_lt_z0, HVG_lt_z1, HVG_y, ILR, KF, KLR, LA_y, LG_y, LL1_LT1_z1, LL1_LT2_z1, LL1_x, LL1_y, LL_LT1_z1, LL_LT2_z1, LL_x, LL_y, LR_lim_zi, LR_lim_zl, LR_x, LR_y, LVG_lt_z0, LVG_lt_z1, LVG_y, OEL, OEL0, SG, SG0, SWUEL_s1, SWUEL_s2, SWUEL_s3, SWVOS_s1, SWVOS_s2, TB, TB1, TC, TC1, TF, UEL, UEL0, UEL2, UEL3, VIMAX, VIMIN, Vs, WF_x, WF_y, XadIfd, efdl, efdu, llim, ue, v, vas, vbus, vf0, vi, vil_lim_zi, vil_lim_zl, vil_lim_zu, vil_x, vil_y, vol_lim_zi, vol_lim_zl, vol_lim_zu, vol_x, vol_y, vout, vref, vref0, zero):
    return (-v + vbus, ue*vol_y - vout, -UEL + UEL0, -OEL + OEL0, -Vs, -vref + vref0, -SG + SG0, KLR*(-ILR + XadIfd) - LR_x, LR_lim_zi*LR_x + LR_lim_zl*zero - LR_y, ue*(-LG_y + SG*SWVOS_s1 + SWUEL_s1*UEL + Vs - WF_y + vref) - vi, vi - vil_x, VIMAX*vil_lim_zu + VIMIN*vil_lim_zl + vil_lim_zi*vil_x - vil_y, -UEL2 + ue*(SWUEL_s2*UEL + llim*(1 - SWUEL_s2)), HVG1_lt_z0*UEL2 + HVG1_lt_z1*vil_y - HVG1_y, LL_LT1_z1*LL_LT2_z1*(-LL_x + LL_y) + LL_x*TB - LL_y*TB + TC*(HVG1_y - LL_x), LL1_LT1_z1*LL1_LT2_z1*(-LL1_x + LL1_y) + LL1_x*TB1 - LL1_y*TB1 + TC1*(-LL1_x + LL_y), ue*(LA_y - LR_y + SG*SWVOS_s2) - vas, -UEL3 + ue*(SWUEL_s3*UEL + llim*(1 - SWUEL_s3)), HVG_lt_z0*UEL3 + HVG_lt_z1*vas - HVG_y, HVG_y*LVG_lt_z1 + LVG_lt_z0*OEL - LVG_y, LVG_y - vol_x, efdl*vol_lim_zl + efdu*vol_lim_zu + vol_lim_zi*vol_x - vol_y, KF*(LVG_y - WF_x) - TF*WF_y, ue*(-vf0 + vout), 0, 0, 0, 0, 0,)


def fx_update():
    return (-1, -1, -1, -1, -1)


def fy_update(KA):
    return (1, 1, 1, KA, 1)


def gy_update(HVG1_lt_z0, HVG1_lt_z1, HVG_lt_z0, HVG_lt_z1, KF, KLR, LL1_LT1_z1, LL1_LT2_z1, LL_LT1_z1, LL_LT2_z1, LR_lim_zi, LVG_lt_z0, LVG_lt_z1, SWUEL_s1, SWUEL_s2, SWUEL_s3, SWVOS_s1, SWVOS_s2, TB, TB1, TC, TC1, TF, ue, vil_lim_zi, vol_lim_zi):
    return (-1, 1, -1, ue, -1, -1, -1, -1, -1, -1, KLR, LR_lim_zi, -1, SWUEL_s1*ue, ue, ue, SWVOS_s1*ue, -1, -ue, 1, -1, vil_lim_zi, -1, SWUEL_s2*ue, -1, HVG1_lt_z1, HVG1_lt_z0, -1, TC, LL_LT1_z1*LL_LT2_z1 - TB, TC1, LL1_LT1_z1*LL1_LT2_z1 - TB1, SWVOS_s2*ue, -ue, -1, SWUEL_s3*ue, -1, HVG_lt_z1, HVG_lt_z0, -1, LVG_lt_z0, LVG_lt_z1, -1, 1, -1, vol_lim_zi, -1, KF, -TF, ue)


def gx_update(KF, LL1_LT1_z1, LL1_LT2_z1, LL_LT1_z1, LL_LT2_z1, TB, TB1, TC, TC1, ue):
    return (-ue, -LL_LT1_z1*LL_LT2_z1 + TB - TC, -LL1_LT1_z1*LL1_LT2_z1 + TB1 - TC1, ue, -KF)


def v_ia(vbus):
    return vbus


def LG_y_ia(v):
    return v


def UEL_ia(UEL0):
    return UEL0


def UEL2_ia(SWUEL_s2, UEL, llim, ue):
    return ue*(SWUEL_s2*UEL + llim*(1 - SWUEL_s2))


def SG_ia(SG0):
    return SG0


def Vs_ia():
    return 0


def WF_y_ia():
    return 0


def LR_x_ia(ILR, KLR, XadIfd):
    return KLR*(-ILR + XadIfd)


def LR_y_ia(LR_lim_zi, LR_lim_zl, LR_x, zero):
    return LR_lim_zi*LR_x + LR_lim_zl*zero


def vref_ia(KA, LR_y, SG, SWUEL_s1, SWVOS_s1, SWVOS_s2, UEL, ue, v, vf0):
    return ue*(-SG*SWVOS_s1 - SWUEL_s1*UEL + v + (LR_y - SG*SWVOS_s2 + vf0)/KA)


def vi_ia(LG_y, SG, SWUEL_s1, SWVOS_s1, UEL, Vs, WF_y, ue, vref):
    return ue*(-LG_y + SG*SWVOS_s1 + SWUEL_s1*UEL + Vs - WF_y + vref)


def vil_x_ia(vi):
    return vi


def vil_y_ia(VIMAX, VIMIN, vil_lim_zi, vil_lim_zl, vil_lim_zu, vil_x):
    return VIMAX*vil_lim_zu + VIMIN*vil_lim_zl + vil_lim_zi*vil_x


def HVG1_y_ia(HVG1_lt_z0, HVG1_lt_z1, UEL2, vil_y):
    return HVG1_lt_z0*UEL2 + HVG1_lt_z1*vil_y


def LL_x_ia(HVG1_y):
    return HVG1_y


def LL_y_ia(HVG1_y):
    return HVG1_y


def LL1_x_ia(LL_y):
    return LL_y


def LL1_y_ia(LL_y):
    return LL_y


def LA_y_ia(KA, LL1_y):
    return KA*LL1_y


def UEL3_ia(SWUEL_s3, UEL, llim, ue):
    return ue*(SWUEL_s3*UEL + llim*(1 - SWUEL_s3))


def vas_ia(LA_y, LR_y, SG, SWVOS_s2, ue):
    return ue*(LA_y - LR_y + SG*SWVOS_s2)


def HVG_y_ia(HVG_lt_z0, HVG_lt_z1, UEL3, vas):
    return HVG_lt_z0*UEL3 + HVG_lt_z1*vas


def OEL_ia(OEL0):
    return OEL0


def LVG_y_ia(HVG_y, LVG_lt_z0, LVG_lt_z1, OEL):
    return HVG_y*LVG_lt_z1 + LVG_lt_z0*OEL


def WF_x_ia(LVG_y):
    return LVG_y


def vout_ia(ue, vf0):
    return ue*vf0


def vol_x_ia(LVG_y):
    return LVG_y


def vol_y_ia(efdl, efdu, vol_lim_zi, vol_lim_zl, vol_lim_zu, vol_x):
    return efdl*vol_lim_zl + efdu*vol_lim_zu + vol_lim_zi*vol_x


def vref_ii(KA, LR_y, SG, SWUEL_s1, SWVOS_s1, SWVOS_s2, UEL, ue, v, vf0):
    return array([[ue*(-SG*SWVOS_s1 - SWUEL_s1*UEL + v + (LR_y - SG*SWVOS_s2 + vf0)/KA)]])


def vi_ii(LG_y, SG, SWUEL_s1, SWVOS_s1, UEL, Vs, WF_y, ue, vref):
    return array([[ue*(-LG_y + SG*SWVOS_s1 + SWUEL_s1*UEL + Vs - WF_y + vref)]])


def vas_ii(LA_y, LR_y, SG, SWVOS_s2, ue):
    return array([[ue*(LA_y - LR_y + SG*SWVOS_s2)]])


def vref_ij():
    return array([[0]])


def vi_ij():
    return array([[0]])


def vas_ij():
    return array([[0]])


def ue_svc(u, ug):
    return u*ug


def UEL0_svc():
    return -999


def OEL0_svc():
    return 999


def ulim_svc():
    return 9999


def llim_svc():
    return -9999


def SG0_svc():
    return 0


def zero_svc():
    return 0


def VA0_svc(LR_y, SG, SWVOS_s2, vf0):
    return LR_y - SG*SWVOS_s2 + vf0


def vref0_svc(vref):
    return vref


def sns_update(KC, VRMAX, VRMIN, XadIfd, vd, vq):
    return (-KC*XadIfd + VRMAX*abs(vd + 1j*vq), VRMIN*abs(vd + 1j*vq),)


f_args = ['HVG1_y',
 'KA',
 'LA_y',
 'LG_y',
 'LL1_x',
 'LL1_y',
 'LL_x',
 'LL_y',
 'LVG_y',
 'WF_x',
 'v']

g_args = ['HVG1_lt_z0',
 'HVG1_lt_z1',
 'HVG1_y',
 'HVG_lt_z0',
 'HVG_lt_z1',
 'HVG_y',
 'ILR',
 'KF',
 'KLR',
 'LA_y',
 'LG_y',
 'LL1_LT1_z1',
 'LL1_LT2_z1',
 'LL1_x',
 'LL1_y',
 'LL_LT1_z1',
 'LL_LT2_z1',
 'LL_x',
 'LL_y',
 'LR_lim_zi',
 'LR_lim_zl',
 'LR_x',
 'LR_y',
 'LVG_lt_z0',
 'LVG_lt_z1',
 'LVG_y',
 'OEL',
 'OEL0',
 'SG',
 'SG0',
 'SWUEL_s1',
 'SWUEL_s2',
 'SWUEL_s3',
 'SWVOS_s1',
 'SWVOS_s2',
 'TB',
 'TB1',
 'TC',
 'TC1',
 'TF',
 'UEL',
 'UEL0',
 'UEL2',
 'UEL3',
 'VIMAX',
 'VIMIN',
 'Vs',
 'WF_x',
 'WF_y',
 'XadIfd',
 'efdl',
 'efdu',
 'llim',
 'ue',
 'v',
 'vas',
 'vbus',
 'vf0',
 'vi',
 'vil_lim_zi',
 'vil_lim_zl',
 'vil_lim_zu',
 'vil_x',
 'vil_y',
 'vol_lim_zi',
 'vol_lim_zl',
 'vol_lim_zu',
 'vol_x',
 'vol_y',
 'vout',
 'vref',
 'vref0',
 'zero']

j_args = {'fx': [],
 'fy': ['KA'],
 'gx': ['KF',
        'LL1_LT1_z1',
        'LL1_LT2_z1',
        'LL_LT1_z1',
        'LL_LT2_z1',
        'TB',
        'TB1',
        'TC',
        'TC1',
        'ue'],
 'gy': ['HVG1_lt_z0',
        'HVG1_lt_z1',
        'HVG_lt_z0',
        'HVG_lt_z1',
        'KF',
        'KLR',
        'LL1_LT1_z1',
        'LL1_LT2_z1',
        'LL_LT1_z1',
        'LL_LT2_z1',
        'LR_lim_zi',
        'LVG_lt_z0',
        'LVG_lt_z1',
        'SWUEL_s1',
        'SWUEL_s2',
        'SWUEL_s3',
        'SWVOS_s1',
        'SWVOS_s2',
        'TB',
        'TB1',
        'TC',
        'TC1',
        'TF',
        'ue',
        'vil_lim_zi',
        'vol_lim_zi']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('UEL0', []),
             ('OEL0', []),
             ('ulim', []),
             ('llim', []),
             ('SG0', []),
             ('zero', []),
             ('VA0', ['LR_y', 'SG', 'SWVOS_s2', 'vf0']),
             ('vref0', ['vref'])])

sns_args = ['KC', 'VRMAX', 'VRMIN', 'XadIfd', 'vd', 'vq']

ia_args = OrderedDict([('v', ['vbus']),
             ('LG_y', ['v']),
             ('UEL', ['UEL0']),
             ('UEL2', ['SWUEL_s2', 'UEL', 'llim', 'ue']),
             ('SG', ['SG0']),
             ('Vs', []),
             ('WF_y', []),
             ('LR_x', ['ILR', 'KLR', 'XadIfd']),
             ('LR_y', ['LR_lim_zi', 'LR_lim_zl', 'LR_x', 'zero']),
             ('vref',
              ['KA',
               'LR_y',
               'SG',
               'SWUEL_s1',
               'SWVOS_s1',
               'SWVOS_s2',
               'UEL',
               'ue',
               'v',
               'vf0']),
             ('vi',
              ['LG_y',
               'SG',
               'SWUEL_s1',
               'SWVOS_s1',
               'UEL',
               'Vs',
               'WF_y',
               'ue',
               'vref']),
             ('vil_x', ['vi']),
             ('vil_y',
              ['VIMAX',
               'VIMIN',
               'vil_lim_zi',
               'vil_lim_zl',
               'vil_lim_zu',
               'vil_x']),
             ('HVG1_y', ['HVG1_lt_z0', 'HVG1_lt_z1', 'UEL2', 'vil_y']),
             ('LL_x', ['HVG1_y']),
             ('LL_y', ['HVG1_y']),
             ('LL1_x', ['LL_y']),
             ('LL1_y', ['LL_y']),
             ('LA_y', ['KA', 'LL1_y']),
             ('UEL3', ['SWUEL_s3', 'UEL', 'llim', 'ue']),
             ('vas', ['LA_y', 'LR_y', 'SG', 'SWVOS_s2', 'ue']),
             ('HVG_y', ['HVG_lt_z0', 'HVG_lt_z1', 'UEL3', 'vas']),
             ('OEL', ['OEL0']),
             ('LVG_y', ['HVG_y', 'LVG_lt_z0', 'LVG_lt_z1', 'OEL']),
             ('WF_x', ['LVG_y']),
             ('vout', ['ue', 'vf0']),
             ('vol_x', ['LVG_y']),
             ('vol_y',
              ['efdl',
               'efdu',
               'vol_lim_zi',
               'vol_lim_zl',
               'vol_lim_zu',
               'vol_x'])])

ii_args = OrderedDict([('vref',
              ['KA',
               'LR_y',
               'SG',
               'SWUEL_s1',
               'SWVOS_s1',
               'SWVOS_s2',
               'UEL',
               'ue',
               'v',
               'vf0']),
             ('vi',
              ['LG_y',
               'SG',
               'SWUEL_s1',
               'SWVOS_s1',
               'UEL',
               'Vs',
               'WF_y',
               'ue',
               'vref']),
             ('vas', ['LA_y', 'LR_y', 'SG', 'SWVOS_s2', 'ue'])])

ij_args = OrderedDict([('vref', []), ('vi', []), ('vas', [])])

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 3, 4]),
             ('fyc', []),
             ('fy', [0, 1, 2, 3, 4]),
             ('gxc', []),
             ('gx', [9, 14, 15, 16, 22]),
             ('gyc', [1, 14, 15, 22]),
             ('gy',
              [0,
               0,
               1,
               1,
               2,
               3,
               4,
               5,
               6,
               7,
               7,
               8,
               8,
               9,
               9,
               9,
               9,
               9,
               9,
               10,
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
               15,
               15,
               16,
               16,
               16,
               17,
               17,
               18,
               18,
               18,
               19,
               19,
               19,
               20,
               20,
               21,
               21,
               22,
               22,
               23])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 3, 4]),
             ('fyc', []),
             ('fy', [6, 19, 20, 21, 25]),
             ('gxc', []),
             ('gx', [0, 1, 2, 3, 4]),
             ('gyc', [7, 20, 21, 28]),
             ('gy',
              [6,
               32,
               7,
               27,
               8,
               9,
               10,
               11,
               12,
               13,
               30,
               13,
               14,
               8,
               10,
               11,
               12,
               15,
               28,
               15,
               16,
               16,
               17,
               8,
               18,
               17,
               18,
               19,
               19,
               20,
               20,
               21,
               12,
               14,
               22,
               8,
               23,
               22,
               23,
               24,
               9,
               24,
               25,
               25,
               26,
               26,
               27,
               25,
               28,
               7])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0]),
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
 'UEL2',
 'SG',
 'Vs',
 'WF_y',
 'XadIfd',
 'LR_x',
 'LR_y',
 'vref',
 'vi',
 'vil_x',
 'vil_y',
 'HVG1_y',
 'LL_x',
 'LL_y',
 'LL1_x',
 'LL1_y',
 'LA_y',
 'UEL3',
 'vas',
 'HVG_y',
 'OEL',
 'LVG_y',
 'WF_x',
 'omega',
 'vout',
 'vol_x',
 'vol_y',
 'vf',
 'a',
 'vd',
 'vq']

need_diag_eps = ['vout', 'LL_y', 'LL1_y', 'WF_y', 'vout', 'LL_y', 'LL1_y', 'WF_y']
