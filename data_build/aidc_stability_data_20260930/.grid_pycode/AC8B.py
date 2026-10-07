from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "36d0b3750f1dfe46009ce5a4396c3d15"

def f_update(KA, LA_y, LG_y, PID_WO_x, PID_uin, PID_y, PID_ys, VFE, kI, ue, v, vi):
    return (-LG_y + v, kI*(2*PID_y - 2*PID_ys + vi), -PID_WO_x + PID_uin, KA*PID_y - LA_y, ue*(LA_y - VFE), 0,)


def g_update(FEX_y, IN, INT_y, KC, KD, KE, LG_y, OEL, OEL0, PID_WO_x, PID_WO_y, PID_lim_zi, PID_lim_zl, PID_lim_zu, PID_uin, PID_xi, PID_y, PID_ys, SAT_A, SAT_B, SL_z0, Se, Td, UEL, UEL0, VFE, VPMAX, VPMIN, Vs, XadIfd, kD, kP, ue, v, vbus, vf0, vi, vout, vref, vref0, __zeros, __ones, __falses, __trues):
    return (-v + vbus, ue*(FEX_y*INT_y - vout), ue*(-IN*INT_y + KC*XadIfd), -FEX_y + select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),greater(IN, 1),__trues], [__ones,1 - 0.577*IN,sqrt(0.75 - IN**2),1.732 - 1.732*IN,__zeros,__zeros], default=nan), -UEL + UEL0, -OEL + OEL0, -Vs, -vref + vref0, ue*(-LG_y + OEL + UEL + Vs - vi + vref), -PID_uin + vi, -PID_WO_y*Td + kD*(-PID_WO_x + PID_uin), PID_WO_y + PID_xi - PID_ys + kP*vi, PID_lim_zi*PID_ys + PID_lim_zl*VPMIN + PID_lim_zu*VPMAX - PID_y, ue*(SAT_B*SL_z0*(INT_y - SAT_A)**2 - Se), ue*(INT_y*KE + KD*XadIfd + Se - VFE), ue*(-vf0 + vout), 0, 0, 0,)


def fx_update(ue):
    return (-1, -1, -1, ue)


def fy_update(KA, kI, ue):
    return (1, kI, -2*kI, 2*kI, 1, KA, -ue)


def gy_update(IN, INT_y, KC, KD, PID_lim_zi, Td, kD, kP, ue, __zeros, __ones, __falses, __trues):
    return (-1, 1, -ue, INT_y*ue, -INT_y*ue, KC*ue, select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),__trues], [__zeros,-0.577*__ones,-IN/sqrt(0.75 - IN**2),-1.732*__ones,__zeros], default=nan), -1, -1, -1, -1, -1, ue, ue, ue, ue, -ue, 1, -1, kD, -Td, kP, 1, -1, PID_lim_zi, -1, -ue, ue, -ue, KD*ue, ue)


def gx_update(FEX_y, IN, INT_y, KE, SAT_A, SAT_B, SL_z0, kD, ue):
    return (FEX_y*ue, -IN*ue, -ue, -kD, 1, SAT_B*SL_z0*ue*(2*INT_y - 2*SAT_A), KE*ue)


def v_ia(vbus):
    return vbus


def LG_y_ia(v):
    return v


def INT_y_ia():
    return 0.100000000000000


def FEX_y_ia():
    return 0.500000000000000


def IN_ia():
    return 1


def Se_ia(INT_y, SAT_A, SAT_B):
    return SAT_B*(INT_y - SAT_A)**2*(greater(INT_y, SAT_A))


def VFE_ia(INT_y, KD, KE, Se, XadIfd):
    return INT_y*KE + KD*XadIfd + Se


def PID_xi_ia(KA, VFE):
    return VFE/KA


def vref_ia(v):
    return v


def vi_ia(v, vref):
    return -v + vref


def PID_uin_ia(vi):
    return vi


def PID_WO_x_ia(PID_uin):
    return PID_uin


def PID_ys_ia(KA, VFE, kP, vi):
    return kP*vi + VFE/KA


def PID_y_ia(PID_lim_zi, PID_lim_zl, PID_lim_zu, PID_ys, VPMAX, VPMIN):
    return PID_lim_zi*PID_ys + PID_lim_zl*VPMIN + PID_lim_zu*VPMAX


def LA_y_ia(KA, PID_y):
    return KA*PID_y


def vout_ia(ue, vf0):
    return ue*vf0


def UEL_ia(UEL0):
    return UEL0


def OEL_ia(OEL0):
    return OEL0


def Vs_ia():
    return 0


def PID_WO_y_ia():
    return 0


def INT_y_FEX_y_IN_ii(FEX_y, IN, INT_y, KC, XadIfd, vf0, __zeros, __ones, __falses, __trues):
    return array([[FEX_y*INT_y - vf0], [-FEX_y + select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),greater(IN, 1),__trues], [__ones,1 - 0.577*IN,sqrt(0.75 - IN**2),1.732 - 1.732*IN,__zeros,__zeros], default=nan)], [-IN*INT_y + KC*XadIfd]])


def INT_y_FEX_y_IN_ij(FEX_y, IN, INT_y, __zeros, __ones, __falses, __trues):
    return array([[FEX_y, INT_y, 0], [0, -1, select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),__trues], [__zeros,-0.577*__ones,-IN/sqrt(0.75 - IN**2),-1.732*__ones,__zeros], default=nan)], [-IN, 0, -INT_y]])


def ue_svc(u, ug):
    return u*ug


def UEL0_svc():
    return 0


def OEL0_svc():
    return 0


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


def vref0_svc(v):
    return v


# empty sns_update

f_args = ['KA',
 'LA_y',
 'LG_y',
 'PID_WO_x',
 'PID_uin',
 'PID_y',
 'PID_ys',
 'VFE',
 'kI',
 'ue',
 'v',
 'vi']

g_args = ['FEX_y',
 'IN',
 'INT_y',
 'KC',
 'KD',
 'KE',
 'LG_y',
 'OEL',
 'OEL0',
 'PID_WO_x',
 'PID_WO_y',
 'PID_lim_zi',
 'PID_lim_zl',
 'PID_lim_zu',
 'PID_uin',
 'PID_xi',
 'PID_y',
 'PID_ys',
 'SAT_A',
 'SAT_B',
 'SL_z0',
 'Se',
 'Td',
 'UEL',
 'UEL0',
 'VFE',
 'VPMAX',
 'VPMIN',
 'Vs',
 'XadIfd',
 'kD',
 'kP',
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

j_args = {'fx': ['ue'],
 'fy': ['KA', 'kI', 'ue'],
 'gx': ['FEX_y', 'IN', 'INT_y', 'KE', 'SAT_A', 'SAT_B', 'SL_z0', 'kD', 'ue'],
 'gy': ['IN',
        'INT_y',
        'KC',
        'KD',
        'PID_lim_zi',
        'Td',
        'kD',
        'kP',
        'ue',
        '__zeros',
        '__ones',
        '__falses',
        '__trues']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('UEL0', []),
             ('OEL0', []),
             ('SAT_E1', ['E1']),
             ('SAT_E2', ['E2', 'SAT_zSE2']),
             ('SAT_SE1', ['SE1']),
             ('SAT_SE2', ['SAT_zSE2', 'SE2']),
             ('SAT_a', ['SAT_E1', 'SAT_E2', 'SAT_SE1', 'SAT_SE2']),
             ('SAT_A', ['SAT_E1', 'SAT_E2', 'SAT_a']),
             ('SAT_B', ['SAT_E1', 'SAT_E2', 'SAT_SE2', 'SAT_a']),
             ('vref0', ['v'])])

sns_args = []

ia_args = OrderedDict([('v', ['vbus']),
             ('LG_y', ['v']),
             ('INT_y', []),
             ('FEX_y', []),
             ('IN', []),
             ('Se', ['INT_y', 'SAT_A', 'SAT_B']),
             ('VFE', ['INT_y', 'KD', 'KE', 'Se', 'XadIfd']),
             ('PID_xi', ['KA', 'VFE']),
             ('vref', ['v']),
             ('vi', ['v', 'vref']),
             ('PID_uin', ['vi']),
             ('PID_WO_x', ['PID_uin']),
             ('PID_ys', ['KA', 'VFE', 'kP', 'vi']),
             ('PID_y',
              ['PID_lim_zi',
               'PID_lim_zl',
               'PID_lim_zu',
               'PID_ys',
               'VPMAX',
               'VPMIN']),
             ('LA_y', ['KA', 'PID_y']),
             ('vout', ['ue', 'vf0']),
             ('UEL', ['UEL0']),
             ('OEL', ['OEL0']),
             ('Vs', []),
             ('PID_WO_y', [])])

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
             ('fx', [0, 2, 3, 4]),
             ('fyc', []),
             ('fy', [0, 1, 1, 1, 2, 3, 4]),
             ('gxc', []),
             ('gx', [1, 2, 8, 10, 11, 13, 14]),
             ('gyc', [1, 2, 8, 10, 13, 14]),
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
               5,
               6,
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
               12,
               13,
               14,
               14,
               14,
               15])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 2, 3, 3]),
             ('fyc', []),
             ('fy', [6, 14, 17, 18, 15, 18, 20]),
             ('gxc', []),
             ('gx', [4, 4, 0, 2, 1, 4, 4]),
             ('gyc', [7, 8, 14, 16, 19, 20]),
             ('gy',
              [6,
               24,
               7,
               9,
               8,
               22,
               8,
               9,
               10,
               11,
               12,
               13,
               10,
               11,
               12,
               13,
               14,
               14,
               15,
               15,
               16,
               14,
               16,
               17,
               17,
               18,
               19,
               19,
               20,
               22,
               7])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08]),
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
 'PID_xi',
 'vref',
 'vi',
 'PID_uin',
 'PID_WO_x',
 'PID_ys',
 'PID_y',
 'LA_y',
 'omega',
 'vout',
 'UEL',
 'OEL',
 'Vs',
 'PID_WO_y',
 'vf',
 'a']

need_diag_eps = ['vout',
 'IN',
 'vi',
 'PID_WO_y',
 'Se',
 'VFE',
 'vout',
 'IN',
 'vi',
 'PID_WO_y',
 'Se',
 'VFE']
