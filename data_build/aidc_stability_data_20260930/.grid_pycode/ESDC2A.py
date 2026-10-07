from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "194685f1b735e82dfb683ade0839b438"

def f_update(INT_y, KA, LA_y, LG_y, LL_x, LL_y, VFE, WF_x, ue, v, vi):
    return (-LG_y + v, -LL_x + vi, KA*LL_y - LA_y, ue*(LA_y - VFE), INT_y - WF_x, 0,)


def g_update(HG_lt_z0, HG_lt_z1, HG_y, INT_y, KE, KF, LL_LT1_z1, LL_LT2_z1, LL_x, LL_y, SAT_A, SAT_B, SL_z0, Se, TB, TC, TF1, UEL, VFE, WF_x, WF_y, ue, v, vbus, vf0, vi, vout, vref, vref0):
    return (-v + vbus, INT_y - vout, -vref + vref0, -WF_y - v - vi + vref, LL_LT1_z1*LL_LT2_z1*(-LL_x + LL_y) + LL_x*TB - LL_y*TB + TC*(-LL_x + vi), -UEL, HG_lt_z0*UEL + HG_lt_z1*LL_y - HG_y, SAT_B*SL_z0*(INT_y - SAT_A)**2 - Se, INT_y*KE + Se - VFE, KF*(INT_y - WF_x) - TF1*WF_y, ue*(-vf0 + vout), 0, 0, 0,)


def fx_update(ue):
    return (-1, -1, -1, ue, 1, -1)


def fy_update(KA, ue):
    return (1, 1, KA, -ue)


def gy_update(HG_lt_z0, HG_lt_z1, LL_LT1_z1, LL_LT2_z1, TB, TC, TF1, ue):
    return (-1, 1, -1, -1, -1, 1, -1, -1, TC, LL_LT1_z1*LL_LT2_z1 - TB, -1, HG_lt_z1, HG_lt_z0, -1, -1, 1, -1, -TF1, ue)


def gx_update(INT_y, KE, KF, LL_LT1_z1, LL_LT2_z1, SAT_A, SAT_B, SL_z0, TB, TC):
    return (1, -LL_LT1_z1*LL_LT2_z1 + TB - TC, SAT_B*SL_z0*(2*INT_y - 2*SAT_A), KE, KF, -KF)


def v_ia(vbus):
    return vbus


def LG_y_ia(v):
    return v


def vi_ia(KA, vfe0):
    return vfe0/KA


def LL_x_ia(vi):
    return vi


def LL_y_ia(vi):
    return vi


def LA_y_ia(KA, LL_y):
    return KA*LL_y


def INT_y_ia(vf0):
    return vf0


def WF_x_ia(INT_y):
    return INT_y


def vout_ia(ue, vf0):
    return ue*vf0


def vref_ia(KA, v, vfe0):
    return v + vfe0/KA


def UEL_ia():
    return 0


def HG_y_ia(HG_lt_z0, HG_lt_z1, LL_y, UEL):
    return HG_lt_z0*UEL + HG_lt_z1*LL_y


def Se_ia(Se0):
    return Se0


def VFE_ia(vfe0):
    return vfe0


def WF_y_ia():
    return 0


def ue_svc(u, ug):
    return u*ug


def VRMAXc_svc(VRMAX, _zVRM):
    return VRMAX - 999*_zVRM + 999


def SAT_E1_svc(E1):
    return E1


def SAT_E2_svc(E2, SAT_zSE2):
    return E2 - 2*SAT_zSE2 + 2


def SAT_SE1_svc(SE1):
    return SE1


def SAT_SE2_svc(SAT_zSE2, SE2):
    return -2*SAT_zSE2 + SE2 + 2


def SAT_a_svc(SAT_E1, SAT_E2, SAT_SE1, SAT_SE2):
    return sqrt(SAT_E1*SAT_SE1/(SAT_E2*SAT_SE2))*((greater(SAT_SE2, 0)) + (less(SAT_SE2, 0)))


def SAT_A_svc(SAT_E1, SAT_E2, SAT_a):
    return SAT_E2 - (SAT_E1 - SAT_E2)/(SAT_a - 1)


def SAT_B_svc(SAT_E1, SAT_E2, SAT_SE2, SAT_a):
    return SAT_E2*SAT_SE2*(SAT_a - 1)**2*((greater(SAT_a, 0)) + (less(SAT_a, 0)))/(SAT_E1 - SAT_E2)**2


def Se0_svc(SAT_A, SAT_B, vf0):
    return SAT_B*(SAT_A - vf0)**2*(greater(vf0, SAT_A))


def vfe0_svc(KE, Se0, vf0):
    return KE*vf0 + Se0


def vref0_svc(vref):
    return vref


def VRU_svc(VRMAXc, v):
    return VRMAXc*v


def VRL_svc(VRMIN, v):
    return VRMIN*v


# empty sns_update

f_args = ['INT_y', 'KA', 'LA_y', 'LG_y', 'LL_x', 'LL_y', 'VFE', 'WF_x', 'ue', 'v', 'vi']

g_args = ['HG_lt_z0',
 'HG_lt_z1',
 'HG_y',
 'INT_y',
 'KE',
 'KF',
 'LL_LT1_z1',
 'LL_LT2_z1',
 'LL_x',
 'LL_y',
 'SAT_A',
 'SAT_B',
 'SL_z0',
 'Se',
 'TB',
 'TC',
 'TF1',
 'UEL',
 'VFE',
 'WF_x',
 'WF_y',
 'ue',
 'v',
 'vbus',
 'vf0',
 'vi',
 'vout',
 'vref',
 'vref0']

j_args = {'fx': ['ue'],
 'fy': ['KA', 'ue'],
 'gx': ['INT_y',
        'KE',
        'KF',
        'LL_LT1_z1',
        'LL_LT2_z1',
        'SAT_A',
        'SAT_B',
        'SL_z0',
        'TB',
        'TC'],
 'gy': ['HG_lt_z0',
        'HG_lt_z1',
        'LL_LT1_z1',
        'LL_LT2_z1',
        'TB',
        'TC',
        'TF1',
        'ue']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('VRMAXc', ['VRMAX', '_zVRM']),
             ('SAT_E1', ['E1']),
             ('SAT_E2', ['E2', 'SAT_zSE2']),
             ('SAT_SE1', ['SE1']),
             ('SAT_SE2', ['SAT_zSE2', 'SE2']),
             ('SAT_a', ['SAT_E1', 'SAT_E2', 'SAT_SE1', 'SAT_SE2']),
             ('SAT_A', ['SAT_E1', 'SAT_E2', 'SAT_a']),
             ('SAT_B', ['SAT_E1', 'SAT_E2', 'SAT_SE2', 'SAT_a']),
             ('Se0', ['SAT_A', 'SAT_B', 'vf0']),
             ('vfe0', ['KE', 'Se0', 'vf0']),
             ('vref0', ['vref']),
             ('VRU', ['VRMAXc', 'v']),
             ('VRL', ['VRMIN', 'v'])])

sns_args = []

ia_args = OrderedDict([('v', ['vbus']),
             ('LG_y', ['v']),
             ('vi', ['KA', 'vfe0']),
             ('LL_x', ['vi']),
             ('LL_y', ['vi']),
             ('LA_y', ['KA', 'LL_y']),
             ('INT_y', ['vf0']),
             ('WF_x', ['INT_y']),
             ('vout', ['ue', 'vf0']),
             ('vref', ['KA', 'v', 'vfe0']),
             ('UEL', []),
             ('HG_y', ['HG_lt_z0', 'HG_lt_z1', 'LL_y', 'UEL']),
             ('Se', ['Se0']),
             ('VFE', ['vfe0']),
             ('WF_y', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 3, 4, 4]),
             ('fyc', []),
             ('fy', [0, 1, 2, 3]),
             ('gxc', []),
             ('gx', [1, 4, 7, 8, 9, 9]),
             ('gyc', [1, 4, 9]),
             ('gy',
              [0, 0, 1, 2, 3, 3, 3, 3, 4, 4, 5, 6, 6, 6, 7, 8, 8, 9, 10])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 2, 3, 4]),
             ('fyc', []),
             ('fy', [6, 9, 10, 14]),
             ('gxc', []),
             ('gx', [3, 1, 3, 3, 3, 4]),
             ('gyc', [7, 10, 15]),
             ('gy',
              [6,
               19,
               7,
               8,
               6,
               8,
               9,
               15,
               9,
               10,
               11,
               10,
               11,
               12,
               13,
               13,
               14,
               15,
               7])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08, 1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['vbus',
 'v',
 'LG_y',
 'vi',
 'LL_x',
 'LL_y',
 'LA_y',
 'INT_y',
 'WF_x',
 'omega',
 'vout',
 'vref',
 'UEL',
 'HG_y',
 'Se',
 'VFE',
 'WF_y',
 'vf',
 'XadIfd',
 'a']

need_diag_eps = ['vout', 'LL_y', 'WF_y', 'vout', 'LL_y', 'WF_y']
