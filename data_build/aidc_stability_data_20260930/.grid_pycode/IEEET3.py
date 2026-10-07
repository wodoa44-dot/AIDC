from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "28f6a775bfbbdaabc3d9b7a8f26716a1"

def f_update(HL_zi, HL_zu, KA, KE, LA1_y, LA3_y, LG_y, VBMAX, VB_y, WF_x, WF_y, ue, v, vi):
    return (-LG_y + v, KA*ue*(-WF_y + vi) - LA3_y, -KE*LA1_y + ue*(HL_zi*VB_y + HL_zu*VBMAX), LA1_y - WF_x, 0,)


def g_update(KF, LA1_y, LA3_y, LG_y, OEL, OEL0, SQE, TF, UEL, UEL0, VB_y, VE, Vs, WF_x, WF_y, XadIfd, ue, v, vbus, vf0, vi, vout, vref, vref0, __zeros, __ones, __falses, __trues):
    return (-v + vbus, ue*(LA1_y - vout), -UEL + UEL0, -OEL + OEL0, -Vs, -vref + vref0, ue*(-LG_y + OEL + UEL + Vs - vi + vref), KF*(LA1_y - WF_x) - TF*WF_y, -SQE + VE**2 - 0.6084*XadIfd**2, -VB_y + select([less_equal(SQE, 0),greater(SQE, 0),__trues], [LA3_y*ue,ue*(LA3_y + sqrt(SQE)),__zeros], default=nan), ue*(-vf0 + vout), 0, 0, 0, 0, 0, 0, 0,)


def fx_update(KE):
    return (-1, -1, -KE, 1, -1)


def fy_update(HL_zi, KA, ue):
    return (1, KA*ue, -KA*ue, HL_zi*ue)


def gy_update(SQE, TF, XadIfd, ue, __zeros, __ones, __falses, __trues):
    return (-1, 1, -ue, -1, -1, -1, -1, ue, ue, ue, ue, -ue, -TF, -1, -1.2168*XadIfd, select([less_equal(SQE, 0),__trues], [__zeros,(1/2)*ue/sqrt(SQE)], default=nan), -1, ue)


def gx_update(KF, ue):
    return (ue, -ue, KF, -KF, ue)


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


def LA3_y_ia(KA, WF_y, ue, vi):
    return KA*ue*(-WF_y + vi)


def SQE_ia(VE, XadIfd):
    return VE**2 - 0.6084*XadIfd**2


def VB_y_ia(LA3_y, SQE, ue, __zeros, __ones, __falses, __trues):
    return select([less_equal(SQE, 0),greater(SQE, 0),__trues], [LA3_y*ue,ue*(LA3_y + sqrt(SQE)),__zeros], default=nan)


def LA1_y_ia(HL_zi, HL_zu, KE, VBMAX, VB_y, ue):
    return ue*(HL_zi*VB_y + HL_zu*VBMAX)/KE


def WF_x_ia(LA1_y):
    return LA1_y


def vout_ia(ue, vf0):
    return ue*vf0


def UEL_ia(UEL0):
    return UEL0


def OEL_ia(OEL0):
    return OEL0


def Vs_ia():
    return 0


def ue_svc(u, ug):
    return u*ug


def VE_svc(Id, Iq, KI, KP, vd, vq):
    return abs(1j*KI*(Id + 1j*Iq) + KP*(vd + 1j*vq))


def V40_svc(VE, XadIfd):
    return sqrt(VE**2 - 0.6084*XadIfd**2)


def VR0_svc(KE, V40, vf0):
    return KE*vf0 - V40


def vb0_svc(KA, VR0):
    return VR0/KA


def VRMAXc_svc(VRMAX, _zVRM):
    return VRMAX - 999*_zVRM + 999


def UEL0_svc():
    return 0


def OEL0_svc():
    return 0


def vref0_svc(vref):
    return vref


def zeros_svc():
    return 0.0


# empty sns_update

f_args = ['HL_zi',
 'HL_zu',
 'KA',
 'KE',
 'LA1_y',
 'LA3_y',
 'LG_y',
 'VBMAX',
 'VB_y',
 'WF_x',
 'WF_y',
 'ue',
 'v',
 'vi']

g_args = ['KF',
 'LA1_y',
 'LA3_y',
 'LG_y',
 'OEL',
 'OEL0',
 'SQE',
 'TF',
 'UEL',
 'UEL0',
 'VB_y',
 'VE',
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

j_args = {'fx': ['KE'],
 'fy': ['HL_zi', 'KA', 'ue'],
 'gx': ['KF', 'ue'],
 'gy': ['SQE',
        'TF',
        'XadIfd',
        'ue',
        '__zeros',
        '__ones',
        '__falses',
        '__trues']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('VE', ['Id', 'Iq', 'KI', 'KP', 'vd', 'vq']),
             ('V40', ['VE', 'XadIfd']),
             ('VR0', ['KE', 'V40', 'vf0']),
             ('vb0', ['KA', 'VR0']),
             ('VRMAXc', ['VRMAX', '_zVRM']),
             ('UEL0', []),
             ('OEL0', []),
             ('vref0', ['vref']),
             ('zeros', [])])

sns_args = []

ia_args = OrderedDict([('v', ['vbus']),
             ('LG_y', ['v']),
             ('WF_y', []),
             ('vref', ['v', 'vb0']),
             ('vi', ['v', 'vref']),
             ('LA3_y', ['KA', 'WF_y', 'ue', 'vi']),
             ('SQE', ['VE', 'XadIfd']),
             ('VB_y',
              ['LA3_y',
               'SQE',
               'ue',
               '__zeros',
               '__ones',
               '__falses',
               '__trues']),
             ('LA1_y', ['HL_zi', 'HL_zu', 'KE', 'VBMAX', 'VB_y', 'ue']),
             ('WF_x', ['LA1_y']),
             ('vout', ['ue', 'vf0']),
             ('UEL', ['UEL0']),
             ('OEL', ['OEL0']),
             ('Vs', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 3, 3]),
             ('fyc', []),
             ('fy', [0, 1, 1, 2]),
             ('gxc', []),
             ('gx', [1, 6, 7, 7, 9]),
             ('gyc', [1, 6, 7]),
             ('gy', [0, 0, 1, 2, 3, 4, 5, 6, 6, 6, 6, 6, 7, 8, 8, 9, 9, 10])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 2, 3]),
             ('fyc', []),
             ('fy', [5, 11, 12, 14]),
             ('gxc', []),
             ('gx', [2, 0, 2, 3, 1]),
             ('gyc', [6, 11, 12]),
             ('gy',
              [5, 18, 6, 7, 8, 9, 10, 7, 8, 9, 10, 11, 12, 13, 16, 13, 14, 6])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08, 1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['vbus',
 'v',
 'LG_y',
 'WF_y',
 'vref',
 'vi',
 'LA3_y',
 'XadIfd',
 'SQE',
 'VB_y',
 'LA1_y',
 'WF_x',
 'omega',
 'vout',
 'UEL',
 'OEL',
 'Vs',
 'vf',
 'a',
 'vd',
 'vq',
 'Id',
 'Iq']

need_diag_eps = ['vout', 'vi', 'WF_y', 'vout', 'vi', 'WF_y']
