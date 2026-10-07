from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "36f2cb33eb65870914c2f564e36addb8"

def f_update(KA, LA_y, LG_y, LL_x, LL_y, VFE, VR_y, WF_x, ue, v, vi):
    return (-LG_y + v, -LL_x + vi, KA*LL_y - LA_y, ue*(-VFE + VR_y), VFE - WF_x, 0,)


def g_update(FEX_y, IN, INT_y, KB, KC, KD, KE, KF, KH, KL, LA_y, LL_LT1_z1, LL_LT2_z1, LL_x, LL_y, LVG_lt_z0, LVG_lt_z1, LVG_y, SAT_A, SAT_B, SL_z0, Se, TB, TC, TF, VFE, VHA, VL, VLR0, VLRx, VRMAX, VRMIN, VR_lim_zi, VR_lim_zl, VR_lim_zu, VR_x, VR_y, WF_x, WF_y, XadIfd, ue, v, vbus, vf0, vi, vout, vref, vref0, __zeros, __ones, __falses, __trues):
    return (-v + vbus, ue*(FEX_y*INT_y - vout), ue*(-IN*INT_y + KC*XadIfd), -FEX_y + select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),greater(IN, 1),__trues], [__ones,1 - 0.577*IN,sqrt(0.75 - IN**2),1.732 - 1.732*IN,__zeros,__zeros], default=nan), ue*(-WF_y - v - vi + vref), LL_LT1_z1*LL_LT2_z1*(-LL_x + LL_y) + LL_x*TB - LL_y*TB + TC*(-LL_x + vi), ue*(SAT_B*SL_z0*(INT_y - SAT_A)**2 - Se), ue*(INT_y*KE + KD*XadIfd + Se - VFE), -vref + vref0, KF*(VFE - WF_x) - TF*WF_y, -KH*VFE + LA_y - VHA, KL*(-VFE + VLRx) - VL, LVG_lt_z0*VL + LVG_lt_z1*VHA - LVG_y, KB*LVG_y - VR_x, VRMAX*VR_lim_zu + VRMIN*VR_lim_zl + VR_lim_zi*VR_x - VR_y, VLR0 - VLRx, ue*(-vf0 + vout), 0, 0, 0,)


def fx_update():
    return (-1, -1, -1, -1)


def fy_update(KA, ue):
    return (1, 1, KA, -ue, ue, 1)


def gy_update(IN, INT_y, KB, KC, KD, KF, KH, KL, LL_LT1_z1, LL_LT2_z1, LVG_lt_z0, LVG_lt_z1, TB, TC, TF, VR_lim_zi, ue, __zeros, __ones, __falses, __trues):
    return (-1, 1, -ue, INT_y*ue, -INT_y*ue, KC*ue, select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),__trues], [__zeros,-0.577*__ones,-IN/sqrt(0.75 - IN**2),-1.732*__ones,__zeros], default=nan), -1, -ue, -ue, ue, -ue, TC, LL_LT1_z1*LL_LT2_z1 - TB, -ue, ue, -ue, KD*ue, -1, KF, -TF, -KH, -1, -KL, -1, KL, LVG_lt_z1, LVG_lt_z0, -1, KB, -1, VR_lim_zi, -1, -1, ue)


def gx_update(FEX_y, IN, INT_y, KE, KF, LL_LT1_z1, LL_LT2_z1, SAT_A, SAT_B, SL_z0, TB, TC, ue):
    return (FEX_y*ue, -IN*ue, -LL_LT1_z1*LL_LT2_z1 + TB - TC, SAT_B*SL_z0*ue*(2*INT_y - 2*SAT_A), KE*ue, -KF, 1)


def v_ia(vbus):
    return vbus


def LG_y_ia(v):
    return v


def INT_y_ia():
    return 0.100000000000000


def FEX_y_ia():
    return 1


def IN_ia():
    return 1


def Se_ia(INT_y, SAT_A, SAT_B):
    return SAT_B*(INT_y - SAT_A)**2*(greater(INT_y, SAT_A))


def VFE_ia(INT_y, KD, KE, Se, XadIfd):
    return INT_y*KE + KD*XadIfd + Se


def vref_ia(KA, KB, KL, VFE, v):
    return v + (KL*VFE + VFE/KB)/KA


def vi_ia(v, vref):
    return -v + vref


def LL_x_ia(vi):
    return vi


def LL_y_ia(vi):
    return vi


def LA_y_ia(KA, LL_y):
    return KA*LL_y


def WF_x_ia(VFE):
    return VFE


def vout_ia(ue, vf0):
    return ue*vf0


def WF_y_ia():
    return 0


def VHA_ia(KH, LA_y, VFE):
    return -KH*VFE + LA_y


def VLRx_ia(KB, KL, VFE, VLR):
    return VLR*(less(VFE + VFE/(KB*KL), VLR)) + (VFE + VFE/(KB*KL))*(greater(VFE + VFE/(KB*KL), VLR))


def VL_ia(KL, VFE, VLRx):
    return KL*(-VFE + VLRx)


def LVG_y_ia(LVG_lt_z0, LVG_lt_z1, VHA, VL):
    return LVG_lt_z0*VL + LVG_lt_z1*VHA


def VR_x_ia(KB, LVG_y):
    return KB*LVG_y


def VR_y_ia(VRMAX, VRMIN, VR_lim_zi, VR_lim_zl, VR_lim_zu, VR_x):
    return VRMAX*VR_lim_zu + VRMIN*VR_lim_zl + VR_lim_zi*VR_x


def INT_y_FEX_y_IN_ii(FEX_y, IN, INT_y, KC, XadIfd, vf0, __zeros, __ones, __falses, __trues):
    return array([[FEX_y*INT_y - vf0], [-FEX_y + select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),greater(IN, 1),__trues], [__ones,1 - 0.577*IN,sqrt(0.75 - IN**2),1.732 - 1.732*IN,__zeros,__zeros], default=nan)], [-IN*INT_y + KC*XadIfd]])


def INT_y_FEX_y_IN_ij(FEX_y, IN, INT_y, __zeros, __ones, __falses, __trues):
    return array([[FEX_y, INT_y, 0], [0, -1, select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),__trues], [__zeros,-0.577*__ones,-IN/sqrt(0.75 - IN**2),-1.732*__ones,__zeros], default=nan)], [-IN, 0, -INT_y]])


def ue_svc(u, ug):
    return u*ug


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


def VLR0_svc(VLRx):
    return VLRx


# empty sns_update

f_args = ['KA', 'LA_y', 'LG_y', 'LL_x', 'LL_y', 'VFE', 'VR_y', 'WF_x', 'ue', 'v', 'vi']

g_args = ['FEX_y',
 'IN',
 'INT_y',
 'KB',
 'KC',
 'KD',
 'KE',
 'KF',
 'KH',
 'KL',
 'LA_y',
 'LL_LT1_z1',
 'LL_LT2_z1',
 'LL_x',
 'LL_y',
 'LVG_lt_z0',
 'LVG_lt_z1',
 'LVG_y',
 'SAT_A',
 'SAT_B',
 'SL_z0',
 'Se',
 'TB',
 'TC',
 'TF',
 'VFE',
 'VHA',
 'VL',
 'VLR0',
 'VLRx',
 'VRMAX',
 'VRMIN',
 'VR_lim_zi',
 'VR_lim_zl',
 'VR_lim_zu',
 'VR_x',
 'VR_y',
 'WF_x',
 'WF_y',
 'XadIfd',
 'ue',
 'v',
 'vbus',
 'vf0',
 'vi',
 'vout',
 'vref',
 'vref0',
 '__zeros',
 '__ones',
 '__falses',
 '__trues']

j_args = {'fx': [],
 'fy': ['KA', 'ue'],
 'gx': ['FEX_y',
        'IN',
        'INT_y',
        'KE',
        'KF',
        'LL_LT1_z1',
        'LL_LT2_z1',
        'SAT_A',
        'SAT_B',
        'SL_z0',
        'TB',
        'TC',
        'ue'],
 'gy': ['IN',
        'INT_y',
        'KB',
        'KC',
        'KD',
        'KF',
        'KH',
        'KL',
        'LL_LT1_z1',
        'LL_LT2_z1',
        'LVG_lt_z0',
        'LVG_lt_z1',
        'TB',
        'TC',
        'TF',
        'VR_lim_zi',
        'ue',
        '__zeros',
        '__ones',
        '__falses',
        '__trues']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('SAT_E1', ['E1']),
             ('SAT_E2', ['E2', 'SAT_zSE2']),
             ('SAT_SE1', ['SE1']),
             ('SAT_SE2', ['SAT_zSE2', 'SE2']),
             ('SAT_a', ['SAT_E1', 'SAT_E2', 'SAT_SE1', 'SAT_SE2']),
             ('SAT_A', ['SAT_E1', 'SAT_E2', 'SAT_a']),
             ('SAT_B', ['SAT_E1', 'SAT_E2', 'SAT_SE2', 'SAT_a']),
             ('vref0', ['vref']),
             ('VLR0', ['VLRx'])])

sns_args = []

ia_args = OrderedDict([('v', ['vbus']),
             ('LG_y', ['v']),
             ('INT_y', []),
             ('FEX_y', []),
             ('IN', []),
             ('Se', ['INT_y', 'SAT_A', 'SAT_B']),
             ('VFE', ['INT_y', 'KD', 'KE', 'Se', 'XadIfd']),
             ('vref', ['KA', 'KB', 'KL', 'VFE', 'v']),
             ('vi', ['v', 'vref']),
             ('LL_x', ['vi']),
             ('LL_y', ['vi']),
             ('LA_y', ['KA', 'LL_y']),
             ('WF_x', ['VFE']),
             ('vout', ['ue', 'vf0']),
             ('WF_y', []),
             ('VHA', ['KH', 'LA_y', 'VFE']),
             ('VLRx', ['KB', 'KL', 'VFE', 'VLR']),
             ('VL', ['KL', 'VFE', 'VLRx']),
             ('LVG_y', ['LVG_lt_z0', 'LVG_lt_z1', 'VHA', 'VL']),
             ('VR_x', ['KB', 'LVG_y']),
             ('VR_y',
              ['VRMAX',
               'VRMIN',
               'VR_lim_zi',
               'VR_lim_zl',
               'VR_lim_zu',
               'VR_x'])])

ii_args = OrderedDict([('INT_y_FEX_y_IN',
              ['FEX_y',
               'IN',
               'INT_y',
               'KC',
               'XadIfd',
               'vf0',
               '__zeros',
               '__ones',
               '__falses',
               '__trues'])])

ij_args = OrderedDict([('INT_y_FEX_y_IN',
              ['FEX_y',
               'IN',
               'INT_y',
               '__zeros',
               '__ones',
               '__falses',
               '__trues'])])

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 4]),
             ('fyc', []),
             ('fy', [0, 1, 2, 3, 3, 4]),
             ('gxc', []),
             ('gx', [1, 2, 5, 6, 7, 9, 10]),
             ('gyc', [1, 2, 4, 5, 6, 7, 9]),
             ('gy',
              [0,
               0,
               1,
               1,
               2,
               2,
               3,
               3,
               4,
               4,
               4,
               4,
               5,
               5,
               6,
               7,
               7,
               7,
               8,
               9,
               9,
               10,
               10,
               11,
               11,
               11,
               12,
               12,
               12,
               13,
               13,
               14,
               14,
               15,
               16])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 4]),
             ('fyc', []),
             ('fy', [6, 10, 11, 13, 20, 13]),
             ('gxc', []),
             ('gx', [3, 3, 1, 3, 3, 4, 2]),
             ('gyc', [7, 8, 10, 11, 12, 13, 15]),
             ('gy',
              [6,
               25,
               7,
               9,
               8,
               23,
               8,
               9,
               6,
               10,
               14,
               15,
               10,
               11,
               12,
               12,
               13,
               23,
               14,
               13,
               15,
               13,
               16,
               13,
               17,
               21,
               16,
               17,
               18,
               18,
               19,
               19,
               20,
               21,
               7])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08]),
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
               0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['vbus',
 'v',
 'LG_y',
 'XadIfd',
 ['INT_y', 'FEX_y', 'IN'],
 'Se',
 'VFE',
 'vref',
 'vi',
 'LL_x',
 'LL_y',
 'LA_y',
 'WF_x',
 'omega',
 'vout',
 'WF_y',
 'VHA',
 'VLRx',
 'VL',
 'LVG_y',
 'VR_x',
 'VR_y',
 'vf',
 'a']

need_diag_eps = ['vout',
 'IN',
 'vi',
 'LL_y',
 'Se',
 'VFE',
 'WF_y',
 'vout',
 'IN',
 'vi',
 'LL_y',
 'Se',
 'VFE',
 'WF_y']
