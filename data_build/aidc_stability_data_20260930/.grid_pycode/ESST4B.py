from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "13fe97f909e4f49e25c1ad7b9b4c5c1b"

def f_update(KIM, KIR, LA_y, LG_y, PI1_y, PI1_ys, PI2_y, PI2_ys, VG_y, v, vi):
    return (-LG_y + v, KIR*(2*PI1_y - 2*PI1_ys + vi), -LA_y + 1.0*PI1_y, KIM*(LA_y + 2*PI2_y - 2*PI2_ys - VG_y), 0,)


def g_update(FEX_y, IN, KC, KG, KPM, KPR, LA_y, LG_y, PI1_lim_zi, PI1_lim_zl, PI1_lim_zu, PI1_xi, PI1_y, PI1_ys, PI2_lim_zi, PI2_lim_zl, PI2_lim_zu, PI2_xi, PI2_y, PI2_ys, UEL, VBMAX, VB_lim_zi, VB_lim_zu, VB_x, VB_y, VE, VGMAX, VG_lim_zi, VG_lim_zu, VG_x, VG_y, VMMAX, VMMIN, VRMAX, VRMIN, XadIfd, ue, v, vbus, vf0, vi, vout, vref, vref0, __zeros, __ones, __falses, __trues):
    return (-v + vbus, PI2_y*VB_y*ue - vout, -UEL, ue*(-IN*VE + KC*XadIfd), -FEX_y + select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),greater(IN, 1),__trues], [__ones,1 - 0.577*IN,sqrt(0.75 - IN**2),1.732 - 1.732*IN,__zeros,__zeros], default=nan), FEX_y*VE - VB_x, VBMAX*VB_lim_zu + VB_lim_zi*VB_x - VB_y, KG*vout - VG_x, VGMAX*VG_lim_zu + VG_lim_zi*VG_x - VG_y, -vref + vref0, -LG_y - vi + vref, KPR*vi + PI1_xi - PI1_ys, PI1_lim_zi*PI1_ys + PI1_lim_zl*VRMIN + PI1_lim_zu*VRMAX - PI1_y, KPM*(LA_y - VG_y) + PI2_xi - PI2_ys, PI2_lim_zi*PI2_ys + PI2_lim_zl*VMMIN + PI2_lim_zu*VMMAX - PI2_y, ue*(-vf0 + vout), 0, 0, 0, 0, 0, 0, 0,)


def fx_update(KIM):
    return (-1, -1, KIM)


def fy_update(KIM, KIR):
    return (1, KIR, -2*KIR, 2*KIR, 1.0, -KIM, -2*KIM, 2*KIM)


def gy_update(IN, KC, KG, KPM, KPR, PI1_lim_zi, PI2_lim_zi, PI2_y, VB_lim_zi, VB_y, VE, VG_lim_zi, ue, __zeros, __ones, __falses, __trues):
    return (-1, 1, -1, PI2_y*ue, VB_y*ue, -1, -VE*ue, KC*ue, select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),__trues], [__zeros,-0.577*__ones,-IN/sqrt(0.75 - IN**2),-1.732*__ones,__zeros], default=nan), -1, VE, -1, VB_lim_zi, -1, KG, -1, VG_lim_zi, -1, -1, 1, -1, KPR, -1, PI1_lim_zi, -1, -KPM, -1, PI2_lim_zi, -1, ue)


def gx_update(KPM):
    return (-1, 1, KPM, 1)


def v_ia(vbus):
    return vbus


def LG_y_ia(v):
    return v


def vout_ia(ue, vf0):
    return ue*vf0


def VG_x_ia(KG, vout):
    return KG*vout


def VG_y_ia(VGMAX, VG_lim_zi, VG_lim_zu, VG_x):
    return VGMAX*VG_lim_zu + VG_lim_zi*VG_x


def PI1_xi_ia(VG_y):
    return VG_y


def vref_ia(v):
    return v


def vi_ia(v, vref):
    return -v + vref


def PI1_ys_ia(KPR, VG_y, vi):
    return KPR*vi + VG_y


def PI1_y_ia(PI1_lim_zi, PI1_lim_zl, PI1_lim_zu, PI1_ys, VRMAX, VRMIN):
    return PI1_lim_zi*PI1_ys + PI1_lim_zl*VRMIN + PI1_lim_zu*VRMAX


def LA_y_ia(PI1_y):
    return 1.0*PI1_y


def IN_ia(KC, VE, XadIfd):
    return safe_div(KC*XadIfd, VE)


def FEX_y_ia(IN, __zeros, __ones, __falses, __trues):
    return select([less_equal(IN, 0),less_equal(IN, 0.433),less_equal(IN, 0.75),less_equal(IN, 1),greater(IN, 1),__trues], [__ones,1 - 0.577*IN,sqrt(0.75 - IN**2),1.732 - 1.732*IN,__zeros,__zeros], default=nan)


def VB_x_ia(FEX_y, VE):
    return FEX_y*VE


def VB_y_ia(VBMAX, VB_lim_zi, VB_lim_zu, VB_x):
    return VBMAX*VB_lim_zu + VB_lim_zi*VB_x


def PI2_xi_ia(VB_y, vf0):
    return safe_div(vf0, VB_y)


def UEL_ia():
    return 0


def PI2_ys_ia(KPM, LA_y, VB_y, VG_y, vf0):
    return KPM*(LA_y - VG_y) + safe_div(vf0, VB_y)


def PI2_y_ia(PI2_lim_zi, PI2_lim_zl, PI2_lim_zu, PI2_ys, VMMAX, VMMIN):
    return PI2_lim_zi*PI2_ys + PI2_lim_zl*VMMIN + PI2_lim_zu*VMMAX


def ue_svc(u, ug):
    return u*ug


def KPC_svc(KP, THETAP):
    return KP*exp(1j*radians(THETAP))


def VE_svc(Id, Iq, KI, KPC, XL, vd, vq):
    return abs(KPC*(vd + 1j*vq) + 1j*(Id + 1j*Iq)*(KI + KPC*XL))


def vref0_svc(vref):
    return vref


# empty sns_update

f_args = ['KIM',
 'KIR',
 'LA_y',
 'LG_y',
 'PI1_y',
 'PI1_ys',
 'PI2_y',
 'PI2_ys',
 'VG_y',
 'v',
 'vi']

g_args = ['FEX_y',
 'IN',
 'KC',
 'KG',
 'KPM',
 'KPR',
 'LA_y',
 'LG_y',
 'PI1_lim_zi',
 'PI1_lim_zl',
 'PI1_lim_zu',
 'PI1_xi',
 'PI1_y',
 'PI1_ys',
 'PI2_lim_zi',
 'PI2_lim_zl',
 'PI2_lim_zu',
 'PI2_xi',
 'PI2_y',
 'PI2_ys',
 'UEL',
 'VBMAX',
 'VB_lim_zi',
 'VB_lim_zu',
 'VB_x',
 'VB_y',
 'VE',
 'VGMAX',
 'VG_lim_zi',
 'VG_lim_zu',
 'VG_x',
 'VG_y',
 'VMMAX',
 'VMMIN',
 'VRMAX',
 'VRMIN',
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

j_args = {'fx': ['KIM'],
 'fy': ['KIM', 'KIR'],
 'gx': ['KPM'],
 'gy': ['IN',
        'KC',
        'KG',
        'KPM',
        'KPR',
        'PI1_lim_zi',
        'PI2_lim_zi',
        'PI2_y',
        'VB_lim_zi',
        'VB_y',
        'VE',
        'VG_lim_zi',
        'ue',
        '__zeros',
        '__ones',
        '__falses',
        '__trues']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('KPC', ['KP', 'THETAP']),
             ('VE', ['Id', 'Iq', 'KI', 'KPC', 'XL', 'vd', 'vq']),
             ('vref0', ['vref'])])

sns_args = []

ia_args = OrderedDict([('v', ['vbus']),
             ('LG_y', ['v']),
             ('vout', ['ue', 'vf0']),
             ('VG_x', ['KG', 'vout']),
             ('VG_y', ['VGMAX', 'VG_lim_zi', 'VG_lim_zu', 'VG_x']),
             ('PI1_xi', ['VG_y']),
             ('vref', ['v']),
             ('vi', ['v', 'vref']),
             ('PI1_ys', ['KPR', 'VG_y', 'vi']),
             ('PI1_y',
              ['PI1_lim_zi',
               'PI1_lim_zl',
               'PI1_lim_zu',
               'PI1_ys',
               'VRMAX',
               'VRMIN']),
             ('LA_y', ['PI1_y']),
             ('IN', ['KC', 'VE', 'XadIfd']),
             ('FEX_y', ['IN', '__zeros', '__ones', '__falses', '__trues']),
             ('VB_x', ['FEX_y', 'VE']),
             ('VB_y', ['VBMAX', 'VB_lim_zi', 'VB_lim_zu', 'VB_x']),
             ('PI2_xi', ['VB_y', 'vf0']),
             ('UEL', []),
             ('PI2_ys', ['KPM', 'LA_y', 'VB_y', 'VG_y', 'vf0']),
             ('PI2_y',
              ['PI2_lim_zi',
               'PI2_lim_zl',
               'PI2_lim_zu',
               'PI2_ys',
               'VMMAX',
               'VMMIN'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 2, 3]),
             ('fyc', []),
             ('fy', [0, 1, 1, 1, 2, 3, 3, 3]),
             ('gxc', []),
             ('gx', [10, 11, 13, 13]),
             ('gyc', [1, 3]),
             ('gy',
              [0,
               0,
               1,
               1,
               1,
               2,
               3,
               3,
               4,
               4,
               5,
               5,
               6,
               6,
               7,
               7,
               8,
               8,
               9,
               10,
               10,
               11,
               11,
               12,
               12,
               13,
               13,
               14,
               14,
               15])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 2, 2]),
             ('fyc', []),
             ('fy', [5, 15, 16, 17, 17, 13, 18, 19]),
             ('gxc', []),
             ('gx', [0, 1, 2, 3]),
             ('gyc', [6, 8]),
             ('gy',
              [5,
               23,
               6,
               11,
               19,
               7,
               8,
               21,
               8,
               9,
               9,
               10,
               10,
               11,
               6,
               12,
               12,
               13,
               14,
               14,
               15,
               15,
               16,
               16,
               17,
               13,
               18,
               18,
               19,
               6])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08]),
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
               0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['vbus',
 'v',
 'LG_y',
 'vout',
 'VG_x',
 'VG_y',
 'PI1_xi',
 'vref',
 'vi',
 'PI1_ys',
 'PI1_y',
 'LA_y',
 'XadIfd',
 'IN',
 'FEX_y',
 'VB_x',
 'VB_y',
 'PI2_xi',
 'omega',
 'UEL',
 'PI2_ys',
 'PI2_y',
 'vf',
 'a',
 'vd',
 'vq',
 'Id',
 'Iq']

need_diag_eps = ['vout', 'IN', 'vout', 'IN']
