from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "ecf279bb957ecff29d239fd755c396ae"

def f_update(KA, KE, LA_y, LL_x, LL_y, LS_y, Se, W_x, ue, v, vi, vp):
    return (ue*(-KE*vp + LA_y - Se*vp), -LS_y + 1.0*v, -LL_x + vi, KA*LL_y - LA_y, -W_x + vp, 0,)


def g_update(KF1, LL_LT1_z1, LL_LT2_z1, LL_x, LL_y, LS_y, SAT_A, SAT_B, SL_z0, Se, TB, TC, TF1, W_x, W_y, ue, v, vbus, vf0, vi, vout, vp, vref, vref0):
    return (-v + vbus, ue*vp - vout, -vref + vref0, SAT_B*SL_z0*(-SAT_A + vp)**2 - Se*vp, -LS_y - W_y - vi + vref, LL_LT1_z1*LL_LT2_z1*(-LL_x + LL_y) + LL_x*TB - LL_y*TB + TC*(-LL_x + vi), KF1*(-W_x + vp) - TF1*W_y, ue*(-vf0 + vout), 0, 0, 0,)


def fx_update(KE, Se, ue):
    return (ue*(-KE - Se), ue, -1, -1, -1, 1, -1)


def fy_update(KA, ue, vp):
    return (-ue*vp, 1.0, 1, KA)


def gy_update(LL_LT1_z1, LL_LT2_z1, TB, TC, TF1, ue, vp):
    return (-1, 1, -1, -1, -vp, 1, -1, -1, TC, LL_LT1_z1*LL_LT2_z1 - TB, -TF1, ue)


def gx_update(KF1, LL_LT1_z1, LL_LT2_z1, SAT_A, SAT_B, SL_z0, Se, TB, TC, ue, vp):
    return (ue, SAT_B*SL_z0*(-2*SAT_A + 2*vp) - Se, -1, -LL_LT1_z1*LL_LT2_z1 + TB - TC, KF1, -KF1)


def vp_ia(vf0):
    return vf0


def v_ia(vbus):
    return vbus


def LS_y_ia(v):
    return 1.0*v


def vi_ia(vb0):
    return vb0


def LL_x_ia(vi):
    return vi


def LL_y_ia(vi):
    return vi


def LA_y_ia(KA, LL_y):
    return KA*LL_y


def W_x_ia(vp):
    return vp


def vout_ia(ue, vf0):
    return ue*vf0


def vref_ia(v, vb0):
    return v + vb0


def Se_ia(Se0):
    return Se0


def W_y_ia():
    return 0


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


def Se0_svc(SAT_A, SAT_B, ug, vf0):
    return SAT_B*(SAT_A - vf0)**2*(greater(vf0, SAT_A))/(-ug + vf0 + 1)


def vr0_svc(KE, Se0, vf0):
    return vf0*(KE + Se0)


def vb0_svc(KA, vr0):
    return vr0/KA


def vref0_svc(vref):
    return vref


def VRTMAX_svc(VRMAX, v):
    return VRMAX*v


def VRTMIN_svc(VRMIN, v):
    return VRMIN*v


# empty sns_update

f_args = ['KA', 'KE', 'LA_y', 'LL_x', 'LL_y', 'LS_y', 'Se', 'W_x', 'ue', 'v', 'vi', 'vp']

g_args = ['KF1',
 'LL_LT1_z1',
 'LL_LT2_z1',
 'LL_x',
 'LL_y',
 'LS_y',
 'SAT_A',
 'SAT_B',
 'SL_z0',
 'Se',
 'TB',
 'TC',
 'TF1',
 'W_x',
 'W_y',
 'ue',
 'v',
 'vbus',
 'vf0',
 'vi',
 'vout',
 'vp',
 'vref',
 'vref0']

j_args = {'fx': ['KE', 'Se', 'ue'],
 'fy': ['KA', 'ue', 'vp'],
 'gx': ['KF1',
        'LL_LT1_z1',
        'LL_LT2_z1',
        'SAT_A',
        'SAT_B',
        'SL_z0',
        'Se',
        'TB',
        'TC',
        'ue',
        'vp'],
 'gy': ['LL_LT1_z1', 'LL_LT2_z1', 'TB', 'TC', 'TF1', 'ue', 'vp']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('SAT_E1', ['E1']),
             ('SAT_E2', ['E2', 'SAT_zSE2']),
             ('SAT_SE1', ['SE1']),
             ('SAT_SE2', ['SAT_zSE2', 'SE2']),
             ('SAT_a', ['SAT_E1', 'SAT_E2', 'SAT_SE1', 'SAT_SE2']),
             ('SAT_A', ['SAT_E1', 'SAT_E2', 'SAT_a']),
             ('SAT_B', ['SAT_E1', 'SAT_E2', 'SAT_SE2', 'SAT_a']),
             ('Se0', ['SAT_A', 'SAT_B', 'ug', 'vf0']),
             ('vr0', ['KE', 'Se0', 'vf0']),
             ('vb0', ['KA', 'vr0']),
             ('vref0', ['vref']),
             ('VRTMAX', ['VRMAX', 'v']),
             ('VRTMIN', ['VRMIN', 'v'])])

sns_args = []

ia_args = OrderedDict([('vp', ['vf0']),
             ('v', ['vbus']),
             ('LS_y', ['v']),
             ('vi', ['vb0']),
             ('LL_x', ['vi']),
             ('LL_y', ['vi']),
             ('LA_y', ['KA', 'LL_y']),
             ('W_x', ['vp']),
             ('vout', ['ue', 'vf0']),
             ('vref', ['v', 'vb0']),
             ('Se', ['Se0']),
             ('W_y', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 1, 2, 3, 4, 4]),
             ('fyc', []),
             ('fy', [0, 1, 2, 3]),
             ('gxc', []),
             ('gx', [1, 3, 4, 5, 6, 6]),
             ('gyc', [1, 3, 5, 6]),
             ('gy', [0, 0, 1, 2, 3, 4, 4, 4, 5, 5, 6, 7])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 3, 1, 2, 3, 0, 4]),
             ('fyc', []),
             ('fy', [9, 6, 10, 11]),
             ('gxc', []),
             ('gx', [0, 0, 1, 2, 0, 4]),
             ('gyc', [7, 9, 11, 12]),
             ('gy', [6, 16, 7, 8, 9, 8, 10, 12, 10, 11, 12, 7])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08, 1e-08, 1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['vp',
 'vbus',
 'v',
 'LS_y',
 'vi',
 'LL_x',
 'LL_y',
 'LA_y',
 'W_x',
 'omega',
 'vout',
 'vref',
 'Se',
 'W_y',
 'vf',
 'XadIfd',
 'a']

need_diag_eps = ['vout', 'Se', 'LL_y', 'W_y', 'vout', 'Se', 'LL_y', 'W_y']
