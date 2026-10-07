from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "d9fefe79264e0efd560e3ad9dc125c13"

def f_update(KA, LA_y, LG_y, LL_x, LL_y, LVG_y, VFE, WF_x, ue, v, vi):
    return (-LG_y + v, -LL_x + vi, KA*LL_y - LA_y, ue*(LVG_y - VFE), VFE - WF_x, 0,)


def g_update(FEX_y, HVG_lt_z0, HVG_lt_z1, HVG_y, IN, INT_y, KC, KD, KE, KF, LA_y, LG_y, LL_LT1_z1, LL_LT2_z1, LL_x, LL_y, LVG_lt_z0, LVG_lt_z1, LVG_y, OEL, OEL0, SAT_A, SAT_B, SL_z0, Se, TB, TC, TF, UEL, UEL0, VFE, Vs, WF_x, WF_y, XadIfd, ue, v, vbus, vf0, vi, vout, vref, vref0, __zeros, __ones, __falses, __trues):
    return (-v + vbus, FEX_y*INT_y*ue - vout, -UEL + UEL0, -OEL + OEL0, -Vs, -vref + vref0, ue*(-IN*INT_y + KC*XadIfd), -FEX_y + select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),greater(IN, 1),__trues], [__ones,1 - 0.577*IN,sqrt(0.75 - IN**2),1.732 - 1.732*IN,__zeros,__zeros], default=nan), ue*(-LG_y + OEL + UEL + Vs - vi + vref), LL_LT1_z1*LL_LT2_z1*(-LL_x + LL_y) + LL_x*TB - LL_y*TB + TC*(-LL_x + vi), HVG_lt_z0*UEL + HVG_lt_z1*LA_y - HVG_y, HVG_y*LVG_lt_z1 + LVG_lt_z0*OEL - LVG_y, ue*(SAT_B*SL_z0*(INT_y - SAT_A)**2 - Se), ue*(INT_y*KE + KD*XadIfd + Se - VFE), KF*(VFE - WF_x) - TF*WF_y, ue*(-vf0 + vout), 0, 0, 0,)


def fx_update():
    return (-1, -1, -1, -1)


def fy_update(KA, ue):
    return (1, 1, KA, ue, -ue, 1)


def gy_update(HVG_lt_z0, IN, INT_y, KC, KD, KF, LL_LT1_z1, LL_LT2_z1, LVG_lt_z0, LVG_lt_z1, TB, TC, TF, ue, __zeros, __ones, __falses, __trues):
    return (-1, 1, -1, INT_y*ue, -1, -1, -1, -1, -INT_y*ue, KC*ue, select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),__trues], [__zeros,-0.577*__ones,-IN/sqrt(0.75 - IN**2),-1.732*__ones,__zeros], default=nan), -1, ue, ue, ue, ue, -ue, TC, LL_LT1_z1*LL_LT2_z1 - TB, HVG_lt_z0, -1, LVG_lt_z0, LVG_lt_z1, -1, -ue, ue, -ue, KD*ue, KF, -TF, ue)


def gx_update(FEX_y, HVG_lt_z1, IN, INT_y, KE, KF, LL_LT1_z1, LL_LT2_z1, SAT_A, SAT_B, SL_z0, TB, TC, ue):
    return (FEX_y*ue, -IN*ue, -ue, -LL_LT1_z1*LL_LT2_z1 + TB - TC, HVG_lt_z1, SAT_B*SL_z0*ue*(2*INT_y - 2*SAT_A), KE*ue, -KF)


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


def vref_ia(KA, VFE, v):
    return v + VFE/KA


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


def UEL_ia(UEL0):
    return UEL0


def OEL_ia(OEL0):
    return OEL0


def Vs_ia():
    return 0


def HVG_y_ia(HVG_lt_z0, HVG_lt_z1, LA_y, UEL):
    return HVG_lt_z0*UEL + HVG_lt_z1*LA_y


def LVG_y_ia(HVG_y, LVG_lt_z0, LVG_lt_z1, OEL):
    return HVG_y*LVG_lt_z1 + LVG_lt_z0*OEL


def WF_y_ia():
    return 0


def INT_y_FEX_y_IN_ii(FEX_y, IN, INT_y, KC, XadIfd, vf0, __zeros, __ones, __falses, __trues):
    return array([[FEX_y*INT_y - vf0], [-FEX_y + select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),greater(IN, 1),__trues], [__ones,1 - 0.577*IN,sqrt(0.75 - IN**2),1.732 - 1.732*IN,__zeros,__zeros], default=nan)], [-IN*INT_y + KC*XadIfd]])


def INT_y_FEX_y_IN_ij(FEX_y, IN, INT_y, __zeros, __ones, __falses, __trues):
    return array([[FEX_y, INT_y, 0], [0, -1, select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),__trues], [__zeros,-0.577*__ones,-IN/sqrt(0.75 - IN**2),-1.732*__ones,__zeros], default=nan)], [-IN, 0, -INT_y]])


def ue_svc(u, ug):
    return u*ug


def UEL0_svc():
    return -999


def OEL0_svc():
    return 999


def VAMAXu_svc(VAMAX, ue):
    return VAMAX*ue - 999*ue + 999


def VAMINu_svc(VAMIN, ue):
    return VAMIN*ue + 999*ue - 999


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

f_args = ['KA', 'LA_y', 'LG_y', 'LL_x', 'LL_y', 'LVG_y', 'VFE', 'WF_x', 'ue', 'v', 'vi']

g_args = ['FEX_y',
 'HVG_lt_z0',
 'HVG_lt_z1',
 'HVG_y',
 'IN',
 'INT_y',
 'KC',
 'KD',
 'KE',
 'KF',
 'LA_y',
 'LG_y',
 'LL_LT1_z1',
 'LL_LT2_z1',
 'LL_x',
 'LL_y',
 'LVG_lt_z0',
 'LVG_lt_z1',
 'LVG_y',
 'OEL',
 'OEL0',
 'SAT_A',
 'SAT_B',
 'SL_z0',
 'Se',
 'TB',
 'TC',
 'TF',
 'UEL',
 'UEL0',
 'VFE',
 'Vs',
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
        'HVG_lt_z1',
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
 'gy': ['HVG_lt_z0',
        'IN',
        'INT_y',
        'KC',
        'KD',
        'KF',
        'LL_LT1_z1',
        'LL_LT2_z1',
        'LVG_lt_z0',
        'LVG_lt_z1',
        'TB',
        'TC',
        'TF',
        'ue',
        '__zeros',
        '__ones',
        '__falses',
        '__trues']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('UEL0', []),
             ('OEL0', []),
             ('VAMAXu', ['VAMAX', 'ue']),
             ('VAMINu', ['VAMIN', 'ue']),
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
             ('LG_y', ['v']),
             ('INT_y', []),
             ('FEX_y', []),
             ('IN', []),
             ('Se', ['INT_y', 'SAT_A', 'SAT_B']),
             ('VFE', ['INT_y', 'KD', 'KE', 'Se', 'XadIfd']),
             ('vref', ['KA', 'VFE', 'v']),
             ('vi', ['v', 'vref']),
             ('LL_x', ['vi']),
             ('LL_y', ['vi']),
             ('LA_y', ['KA', 'LL_y']),
             ('WF_x', ['VFE']),
             ('vout', ['ue', 'vf0']),
             ('UEL', ['UEL0']),
             ('OEL', ['OEL0']),
             ('Vs', []),
             ('HVG_y', ['HVG_lt_z0', 'HVG_lt_z1', 'LA_y', 'UEL']),
             ('LVG_y', ['HVG_y', 'LVG_lt_z0', 'LVG_lt_z1', 'OEL']),
             ('WF_y', [])])

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
             ('gx', [1, 6, 8, 9, 10, 12, 13, 14]),
             ('gyc', [1, 6, 8, 9, 12, 13, 14]),
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
               6,
               7,
               7,
               8,
               8,
               8,
               8,
               8,
               9,
               9,
               10,
               10,
               11,
               11,
               11,
               12,
               13,
               13,
               13,
               14,
               14,
               15])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 4]),
             ('fyc', []),
             ('fy', [6, 14, 15, 17, 19, 19]),
             ('gxc', []),
             ('gx', [3, 3, 0, 1, 2, 3, 3, 4]),
             ('gyc', [7, 12, 14, 15, 18, 19, 20]),
             ('gy',
              [6,
               24,
               7,
               13,
               8,
               9,
               10,
               11,
               12,
               22,
               12,
               13,
               8,
               9,
               10,
               11,
               14,
               14,
               15,
               8,
               16,
               9,
               16,
               17,
               18,
               18,
               19,
               22,
               19,
               20,
               7])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0, 0]),
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
 'UEL',
 'OEL',
 'Vs',
 'HVG_y',
 'LVG_y',
 'WF_y',
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
