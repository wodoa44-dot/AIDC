from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "324e563b03b6d35dcc7f3f2a035506fb"

def f_update(KA, LA_y, LG_y, VFE, WF_x, WF_y, ue, v, vi, vout):
    return (-LG_y + v, KA*ue*(-WF_y + vi) - LA_y, ue*(LA_y - VFE), -WF_x + vout, 0,)


def g_update(INT_y, KE, KF, LG_y, SAT_A, SAT_B, SL_z0, Se, TF, VFE, WF_x, WF_y, ue, v, vbus, vf0, vi, vout, vref, vref0):
    return (-v + vbus, ue*(INT_y - vout), -vref + vref0, ue*(-LG_y - vi + vref), ue*(INT_y*KE + Se - VFE), SAT_B*SL_z0*(INT_y - SAT_A)**2 - Se, KF*(-WF_x + vout) - TF*WF_y, ue*(-vf0 + vout), 0, 0, 0,)


def fx_update(ue):
    return (-1, -1, ue, -1)


def fy_update(KA, ue):
    return (1, KA*ue, -KA*ue, -ue, 1)


def gy_update(KF, TF, ue):
    return (-1, 1, -ue, -1, ue, -ue, -ue, ue, -1, KF, -TF, ue)


def gx_update(INT_y, KE, KF, SAT_A, SAT_B, SL_z0, ue):
    return (ue, -ue, KE*ue, SAT_B*SL_z0*(2*INT_y - 2*SAT_A), -KF)


def v_ia(vbus):
    return vbus


def LG_y_ia(v):
    return v


def WF_y_ia():
    return 0


def vref_ia(v, vb0):
    return v + vb0


def vi_ia(v, vref):
    return -v + vref


def LA_y_ia(KA, WF_y, ue, vi):
    return KA*ue*(-WF_y + vi)


def INT_y_ia(vf0):
    return vf0


def vout_ia(ue, vf0):
    return ue*vf0


def WF_x_ia(vout):
    return vout


def VFE_ia(vfe0):
    return vfe0


def Se_ia(Se0):
    return Se0


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


def vr0_svc(KE, Se0, vf0):
    return KE*vf0 + Se0


def vb0_svc(KA, vr0):
    return vr0/KA


def vfe0_svc(KE, Se0, vf0):
    return KE*vf0 + Se0


def vref0_svc(vref):
    return vref


# empty sns_update

f_args = ['KA', 'LA_y', 'LG_y', 'VFE', 'WF_x', 'WF_y', 'ue', 'v', 'vi', 'vout']

g_args = ['INT_y',
 'KE',
 'KF',
 'LG_y',
 'SAT_A',
 'SAT_B',
 'SL_z0',
 'Se',
 'TF',
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
 'gx': ['INT_y', 'KE', 'KF', 'SAT_A', 'SAT_B', 'SL_z0', 'ue'],
 'gy': ['KF', 'TF', 'ue']}

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
             ('vr0', ['KE', 'Se0', 'vf0']),
             ('vb0', ['KA', 'vr0']),
             ('vfe0', ['KE', 'Se0', 'vf0']),
             ('vref0', ['vref'])])

sns_args = []

ia_args = OrderedDict([('v', ['vbus']),
             ('LG_y', ['v']),
             ('WF_y', []),
             ('vref', ['v', 'vb0']),
             ('vi', ['v', 'vref']),
             ('LA_y', ['KA', 'WF_y', 'ue', 'vi']),
             ('INT_y', ['vf0']),
             ('vout', ['ue', 'vf0']),
             ('WF_x', ['vout']),
             ('VFE', ['vfe0']),
             ('Se', ['Se0'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 3]),
             ('fyc', []),
             ('fy', [0, 1, 1, 2, 3]),
             ('gxc', []),
             ('gx', [1, 3, 4, 5, 6]),
             ('gyc', [1, 3, 4, 6]),
             ('gy', [0, 0, 1, 2, 3, 3, 4, 4, 5, 6, 6, 7])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 1, 3]),
             ('fyc', []),
             ('fy', [5, 8, 11, 9, 6]),
             ('gxc', []),
             ('gx', [2, 0, 2, 2, 3]),
             ('gyc', [6, 8, 9, 11]),
             ('gy', [5, 15, 6, 7, 7, 8, 9, 10, 10, 6, 11, 6])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08, 1e-08, 1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['vbus',
 'v',
 'LG_y',
 'WF_y',
 'vref',
 'vi',
 'LA_y',
 'INT_y',
 'vout',
 'WF_x',
 'omega',
 'VFE',
 'Se',
 'vf',
 'XadIfd',
 'a']

need_diag_eps = ['vout', 'vi', 'VFE', 'WF_y', 'vout', 'vi', 'VFE', 'WF_y']
