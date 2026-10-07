from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "d90fbb92778330b4208fb57e38cb6c35"

def f_update(KA, LL_x, LL_y, LP_y, VFE, VR_y, WF_x, ue, v, vi):
    return (-LP_y + v, KA*vi - VR_y, -LL_x + VR_y, LL_y - WF_x, ue*(-VFE + VR_y), 0,)


def g_update(INT_y, KE, KF, LL_LT1_z1, LL_LT2_z1, LL_x, LL_y, LP_y, OEL, OEL0, SAT_A, SAT_B, SL_z0, Se, TF1, TF2, TF3, UEL, UEL0, VFE, VR_y, Vs, WF_x, WF_y, ue, v, vbus, vf0, vi, vout, vref, vref0):
    return (-v + vbus, INT_y*ue - vout, -UEL + UEL0, -OEL + OEL0, -Vs, -vref + vref0, ue*(-LP_y + Vs - WF_y + vref) - vi, LL_LT1_z1*LL_LT2_z1*(-LL_x + LL_y) + LL_x*TF2 - LL_y*TF2 + TF3*(-LL_x + VR_y), KF*(LL_y - WF_x) - TF1*WF_y, ue*(SAT_B*SL_z0*(INT_y - SAT_A)**2 - Se), ue*(INT_y*KE + Se - VFE), ue*(-vf0 + vout), 0, 0, 0,)


def fx_update(ue):
    return (-1, -1, 1, -1, -1, ue)


def fy_update(KA, ue):
    return (1, KA, 1, -ue)


def gy_update(KF, LL_LT1_z1, LL_LT2_z1, TF1, TF2, ue):
    return (-1, 1, -1, -1, -1, -1, -1, ue, ue, -1, -ue, LL_LT1_z1*LL_LT2_z1 - TF2, KF, -TF1, -ue, ue, -ue, ue)


def gx_update(INT_y, KE, KF, LL_LT1_z1, LL_LT2_z1, SAT_A, SAT_B, SL_z0, TF2, TF3, ue):
    return (ue, -ue, TF3, -LL_LT1_z1*LL_LT2_z1 + TF2 - TF3, -KF, SAT_B*SL_z0*ue*(2*INT_y - 2*SAT_A), KE*ue)


def v_ia(vbus):
    return vbus


def LP_y_ia(v):
    return v


def INT_y_ia(vf0):
    return vf0


def Se_ia(INT_y, SAT_A, SAT_B):
    return SAT_B*(INT_y - SAT_A)**2*(greater(INT_y, SAT_A))


def VFE_ia(INT_y, KE, Se):
    return INT_y*KE + Se


def vref_ia(KA, VFE, v):
    return v + VFE/KA


def vi_ia(ue, v, vref):
    return ue*(-v + vref)


def VR_y_ia(KA, vi):
    return KA*vi


def LL_x_ia(VR_y):
    return VR_y


def LL_y_ia(VR_y):
    return VR_y


def WF_x_ia(LL_y):
    return LL_y


def vout_ia(ue, vf0):
    return ue*vf0


def UEL_ia(UEL0):
    return UEL0


def OEL_ia(OEL0):
    return OEL0


def Vs_ia():
    return 0


def WF_y_ia():
    return 0


def ue_svc(u, ug):
    return u*ug


def UEL0_svc():
    return 0


def OEL0_svc():
    return 0


def VRMAXu_svc(VRMAX, ue):
    return VRMAX*ue - 999*ue + 999


def VRMINu_svc(VRMIN, ue):
    return VRMIN*ue + 999*ue - 999


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


def vref0_svc(vref):
    return vref


# empty sns_update

f_args = ['KA', 'LL_x', 'LL_y', 'LP_y', 'VFE', 'VR_y', 'WF_x', 'ue', 'v', 'vi']

g_args = ['INT_y',
 'KE',
 'KF',
 'LL_LT1_z1',
 'LL_LT2_z1',
 'LL_x',
 'LL_y',
 'LP_y',
 'OEL',
 'OEL0',
 'SAT_A',
 'SAT_B',
 'SL_z0',
 'Se',
 'TF1',
 'TF2',
 'TF3',
 'UEL',
 'UEL0',
 'VFE',
 'VR_y',
 'Vs',
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
        'TF2',
        'TF3',
        'ue'],
 'gy': ['KF', 'LL_LT1_z1', 'LL_LT2_z1', 'TF1', 'TF2', 'ue']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('UEL0', []),
             ('OEL0', []),
             ('VRMAXu', ['VRMAX', 'ue']),
             ('VRMINu', ['VRMIN', 'ue']),
             ('SAT_E1', ['E1']),
             ('SAT_E2', ['E2', 'SAT_zSE2']),
             ('SAT_SE1', ['SE1']),
             ('SAT_SE2', ['SAT_zSE2', 'SE2']),
             ('SAT_a', ['SAT_E1', 'SAT_E2', 'SAT_SE1', 'SAT_SE2']),
             ('SAT_A', ['SAT_E1', 'SAT_E2', 'SAT_a']),
             ('SAT_B', ['SAT_E1', 'SAT_E2', 'SAT_SE2', 'SAT_a']),
             ('vref0', ['vref'])])

sns_args = []

ia_args = OrderedDict([('v', ['vbus']),
             ('LP_y', ['v']),
             ('INT_y', ['vf0']),
             ('Se', ['INT_y', 'SAT_A', 'SAT_B']),
             ('VFE', ['INT_y', 'KE', 'Se']),
             ('vref', ['KA', 'VFE', 'v']),
             ('vi', ['ue', 'v', 'vref']),
             ('VR_y', ['KA', 'vi']),
             ('LL_x', ['VR_y']),
             ('LL_y', ['VR_y']),
             ('WF_x', ['LL_y']),
             ('vout', ['ue', 'vf0']),
             ('UEL', ['UEL0']),
             ('OEL', ['OEL0']),
             ('Vs', []),
             ('WF_y', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 2, 3, 4]),
             ('fyc', []),
             ('fy', [0, 1, 3, 4]),
             ('gxc', []),
             ('gx', [1, 6, 7, 7, 8, 9, 10]),
             ('gyc', [1, 7, 8, 9, 10]),
             ('gy', [0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 7, 8, 8, 9, 10, 10, 11])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 1, 2, 3, 1]),
             ('fyc', []),
             ('fy', [6, 12, 13, 16]),
             ('gxc', []),
             ('gx', [4, 0, 1, 2, 3, 4, 4]),
             ('gyc', [7, 13, 14, 15, 16]),
             ('gy',
              [6,
               20,
               7,
               8,
               9,
               10,
               11,
               10,
               11,
               12,
               14,
               13,
               13,
               14,
               15,
               15,
               16,
               7])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08, 1e-08, 1e-08, 1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['vbus',
 'v',
 'LP_y',
 'INT_y',
 'Se',
 'VFE',
 'vref',
 'vi',
 'VR_y',
 'LL_x',
 'LL_y',
 'WF_x',
 'omega',
 'vout',
 'UEL',
 'OEL',
 'Vs',
 'WF_y',
 'vf',
 'XadIfd',
 'a']

need_diag_eps = ['vout', 'LL_y', 'WF_y', 'Se', 'VFE', 'vout', 'LL_y', 'WF_y', 'Se', 'VFE']
