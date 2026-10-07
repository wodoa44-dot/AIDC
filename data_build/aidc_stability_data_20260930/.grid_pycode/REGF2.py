from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "91f93de50ba1efea7c701b1dae51f900"

def f_update(INTw_y, Id, Iq, KIi, KIplim, KIqlim, KIv, PIplim_y, PIvd_y, PIvq_y, Paux, Pe, Psen_y, Psig_y, Qaux, Qe, Qsen_y, Qsig_y, dd, dw_y, plldw, udLag_y, udref, uqLag_y, uqref, vd, vq, vref2, wdrp, wref):
    return (Pe - Psen_y, Qe - Qsen_y, Paux + Psen_y - Psig_y, Qaux + Qsen_y - Qsig_y, KIplim*(-Psen_y + Psig_y), KIqlim*(-Qsen_y + Qsig_y), dw_y, -INTw_y + dd*plldw*wdrp + wdrp*(PIplim_y - Psen_y) + wref, KIv*(-vd + vref2), -KIv*vq, KIi*(-Id + PIvd_y), KIi*(-Iq + PIvq_y), -udLag_y + udref, -uqLag_y + uqref, 0, 0,)


def g_update(INTw_y, Id, Iq, KPi, KPplim, KPqlim, KPv, PIId_xi, PIId_y, PIIq_xi, PIIq_y, PIplim_xi, PIplim_y, PIqlim_xi, PIqlim_y, PIvd_xi, PIvd_y, PIvq_xi, PIvq_y, Paux, Pe, Psen_y, Psig_y, Qaux, Qdrp, Qe, Qsen_y, Qsig_y, a, delta, dw_lim_zi, dw_lim_zl, dw_lim_zu, dw_x, dw_y, dwmax, dwmin, rf, u, udLag_y, udref, uqLag_y, uqref, v, vd, vq, vref, vref2, w0, wref, xf):
    return (Paux, Qaux, KPplim*(-Psen_y + Psig_y) + PIplim_xi - PIplim_y, KPqlim*(-Qsen_y + Qsig_y) + PIqlim_xi - PIqlim_y, u*v*cos(a - delta) - vd, u*v*sin(a - delta) - vq, Id*vd + Iq*vq - Pe, Id*vq - Iq*vd - Qe, Id*rf - Iq*xf - udLag_y + vd, Id*xf + Iq*rf - uqLag_y + vq, 1 - wref, -dw_x + w0*(INTw_y - wref), dw_lim_zi*dw_x + dw_lim_zl*dwmin + dw_lim_zu*dwmax - dw_y, Qdrp*(PIqlim_y*u - Qsen_y) + vref - vref2, KPv*(-vd + vref2) + PIvd_xi - PIvd_y, -KPv*vq + PIvq_xi - PIvq_y, KPi*(-Id + PIvd_y) + PIId_xi - PIId_y, KPi*(-Iq + PIvq_y) + PIIq_xi - PIIq_y, Id*rf - Iq*xf + PIId_y - udref + vd, Id*xf + Iq*rf + PIIq_y - uqref + vq, -Pe*u, -Qe*u, 0, 0, 0,)


def fx_update(KIplim, KIqlim, wdrp):
    return (-1, -1, 1, -1, 1, -1, -KIplim, KIplim, -KIqlim, KIqlim, -wdrp, -1, -1, -1)


def fy_update(KIi, KIv, dd, wdrp):
    return (1, 1, 1, 1, 1, wdrp, 1, dd*wdrp, -KIv, KIv, -KIv, -KIi, KIi, -KIi, KIi, 1, 1)


def gy_update(Id, Iq, KPi, KPv, Qdrp, a, delta, dw_lim_zi, rf, u, v, vd, vq, w0, xf):
    return (1, 1, -1, -1, -1, -u*v*sin(a - delta), u*cos(a - delta), -1, u*v*cos(a - delta), u*sin(a - delta), Id, Iq, -1, vd, vq, -Iq, Id, -1, vq, -vd, 1, rf, -xf, 1, xf, rf, -1, -w0, -1, dw_lim_zi, -1, Qdrp*u, -1, -KPv, KPv, -1, -KPv, -1, -KPi, KPi, -1, -KPi, KPi, -1, 1, rf, -xf, 1, -1, 1, xf, rf, 1, -1, -u, -u)


def gx_update(KPplim, KPqlim, Qdrp, a, delta, u, v, w0):
    return (-KPplim, KPplim, 1, -KPqlim, KPqlim, 1, u*v*sin(a - delta), -u*v*cos(a - delta), -1, -1, w0, -Qdrp, 1, 1, 1, 1)


def Pe_ia(Pref):
    return Pref


def Psen_y_ia(Pe):
    return Pe


def Qe_ia(Qref):
    return Qref


def Qsen_y_ia(Qe):
    return Qe


def Paux_ia():
    return 0


def Psig_y_ia(Paux, Psen_y):
    return Paux + Psen_y


def Qaux_ia():
    return 0


def Qsig_y_ia(Qaux, Qsen_y):
    return Qaux + Qsen_y


def PIplim_xi_ia(Psen_y):
    return Psen_y


def PIqlim_xi_ia(Qsen_y):
    return Qsen_y


def delta_ia(a):
    return a


def wref_ia():
    return 1


def INTw_y_ia(wref):
    return wref


def PIvd_xi_ia(Id0):
    return Id0


def PIvq_xi_ia(Iq0):
    return Iq0


def PIId_xi_ia():
    return 0.0


def PIIq_xi_ia():
    return 0.0


def udref_ia(udref0):
    return udref0


def udLag_y_ia(udref):
    return udref


def uqref_ia(uqref0):
    return uqref0


def uqLag_y_ia(uqref):
    return uqref


def PIplim_y_ia(KPplim, Psen_y, Psig_y):
    return KPplim*(-Psen_y + Psig_y) + Psen_y


def PIqlim_y_ia(KPqlim, Qsen_y, Qsig_y):
    return KPqlim*(-Qsen_y + Qsig_y) + Qsen_y


def vd_ia(vd0):
    return vd0


def vq_ia(vq0):
    return vq0


def Id_ia(Id0):
    return Id0


def Iq_ia(Iq0):
    return Iq0


def dw_x_ia(INTw_y, w0, wref):
    return w0*(INTw_y - wref)


def dw_y_ia(dw_lim_zi, dw_lim_zl, dw_lim_zu, dw_x, dwmax, dwmin):
    return dw_lim_zi*dw_x + dw_lim_zl*dwmin + dw_lim_zu*dwmax


def vref2_ia(u, vref):
    return u*vref


def PIvd_y_ia(Id0, KPv, vd, vref2):
    return Id0 + KPv*(-vd + vref2)


def PIvq_y_ia(Iq0, KPv, vq):
    return Iq0 - KPv*vq


def PIId_y_ia(Id, KPi, PIvd_y):
    return KPi*(-Id + PIvd_y)


def PIIq_y_ia(Iq, KPi, PIvq_y):
    return KPi*(-Iq + PIvq_y)


def Pref_svc(gammap, p0s):
    return gammap*p0s


def Qref_svc(gammaq, q0s):
    return gammaq*q0s


def w0_svc(fn):
    return 2*pi*fn


def ixf_svc(xf):
    return xf**(-1.0)


def Id0_svc(Pref, u, v):
    return Pref*u/v


def Iq0_svc(Qref, u, v):
    return -Qref*u/v


def vd0_svc(u, v):
    return u*v


def vq0_svc():
    return 0


def Tint_svc(mf, wdrp):
    return mf*wdrp


def udref0_svc(Id0, Iq0, rf, vd0, xf):
    return Id0*rf - Iq0*xf + vd0


def uqref0_svc(Id0, Iq0, rf, vq0, xf):
    return Id0*xf + Iq0*rf + vq0


# empty sns_update

f_args = ['INTw_y',
 'Id',
 'Iq',
 'KIi',
 'KIplim',
 'KIqlim',
 'KIv',
 'PIplim_y',
 'PIvd_y',
 'PIvq_y',
 'Paux',
 'Pe',
 'Psen_y',
 'Psig_y',
 'Qaux',
 'Qe',
 'Qsen_y',
 'Qsig_y',
 'dd',
 'dw_y',
 'plldw',
 'udLag_y',
 'udref',
 'uqLag_y',
 'uqref',
 'vd',
 'vq',
 'vref2',
 'wdrp',
 'wref']

g_args = ['INTw_y',
 'Id',
 'Iq',
 'KPi',
 'KPplim',
 'KPqlim',
 'KPv',
 'PIId_xi',
 'PIId_y',
 'PIIq_xi',
 'PIIq_y',
 'PIplim_xi',
 'PIplim_y',
 'PIqlim_xi',
 'PIqlim_y',
 'PIvd_xi',
 'PIvd_y',
 'PIvq_xi',
 'PIvq_y',
 'Paux',
 'Pe',
 'Psen_y',
 'Psig_y',
 'Qaux',
 'Qdrp',
 'Qe',
 'Qsen_y',
 'Qsig_y',
 'a',
 'delta',
 'dw_lim_zi',
 'dw_lim_zl',
 'dw_lim_zu',
 'dw_x',
 'dw_y',
 'dwmax',
 'dwmin',
 'rf',
 'u',
 'udLag_y',
 'udref',
 'uqLag_y',
 'uqref',
 'v',
 'vd',
 'vq',
 'vref',
 'vref2',
 'w0',
 'wref',
 'xf']

j_args = {'fx': ['KIplim', 'KIqlim', 'wdrp'],
 'fy': ['KIi', 'KIv', 'dd', 'wdrp'],
 'gx': ['KPplim', 'KPqlim', 'Qdrp', 'a', 'delta', 'u', 'v', 'w0'],
 'gy': ['Id',
        'Iq',
        'KPi',
        'KPv',
        'Qdrp',
        'a',
        'delta',
        'dw_lim_zi',
        'rf',
        'u',
        'v',
        'vd',
        'vq',
        'w0',
        'xf']}

s_args = OrderedDict([('Pref', ['gammap', 'p0s']),
             ('Qref', ['gammaq', 'q0s']),
             ('w0', ['fn']),
             ('ixf', ['xf']),
             ('Id0', ['Pref', 'u', 'v']),
             ('Iq0', ['Qref', 'u', 'v']),
             ('vd0', ['u', 'v']),
             ('vq0', []),
             ('Tint', ['mf', 'wdrp']),
             ('udref0', ['Id0', 'Iq0', 'rf', 'vd0', 'xf']),
             ('uqref0', ['Id0', 'Iq0', 'rf', 'vq0', 'xf'])])

sns_args = []

ia_args = OrderedDict([('Pe', ['Pref']),
             ('Psen_y', ['Pe']),
             ('Qe', ['Qref']),
             ('Qsen_y', ['Qe']),
             ('Paux', []),
             ('Psig_y', ['Paux', 'Psen_y']),
             ('Qaux', []),
             ('Qsig_y', ['Qaux', 'Qsen_y']),
             ('PIplim_xi', ['Psen_y']),
             ('PIqlim_xi', ['Qsen_y']),
             ('delta', ['a']),
             ('wref', []),
             ('INTw_y', ['wref']),
             ('PIvd_xi', ['Id0']),
             ('PIvq_xi', ['Iq0']),
             ('PIId_xi', []),
             ('PIIq_xi', []),
             ('udref', ['udref0']),
             ('udLag_y', ['udref']),
             ('uqref', ['uqref0']),
             ('uqLag_y', ['uqref']),
             ('PIplim_y', ['KPplim', 'Psen_y', 'Psig_y']),
             ('PIqlim_y', ['KPqlim', 'Qsen_y', 'Qsig_y']),
             ('vd', ['vd0']),
             ('vq', ['vq0']),
             ('Id', ['Id0']),
             ('Iq', ['Iq0']),
             ('dw_x', ['INTw_y', 'w0', 'wref']),
             ('dw_y',
              ['dw_lim_zi',
               'dw_lim_zl',
               'dw_lim_zu',
               'dw_x',
               'dwmax',
               'dwmin']),
             ('vref2', ['u', 'vref']),
             ('PIvd_y', ['Id0', 'KPv', 'vd', 'vref2']),
             ('PIvq_y', ['Iq0', 'KPv', 'vq']),
             ('PIId_y', ['Id', 'KPi', 'PIvd_y']),
             ('PIIq_y', ['Iq', 'KPi', 'PIvq_y'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 2, 3, 3, 4, 4, 5, 5, 7, 7, 12, 13]),
             ('fyc', []),
             ('fy', [0, 1, 2, 3, 6, 7, 7, 7, 8, 8, 9, 10, 10, 11, 11, 12, 13]),
             ('gxc', []),
             ('gx', [2, 2, 2, 3, 3, 3, 4, 5, 8, 9, 11, 13, 14, 15, 16, 17]),
             ('gyc', [8, 9]),
             ('gy',
              [0,
               1,
               2,
               3,
               4,
               4,
               4,
               5,
               5,
               5,
               6,
               6,
               6,
               6,
               6,
               7,
               7,
               7,
               7,
               7,
               8,
               8,
               8,
               9,
               9,
               9,
               10,
               11,
               11,
               12,
               12,
               13,
               13,
               14,
               14,
               14,
               15,
               15,
               16,
               16,
               16,
               17,
               17,
               17,
               18,
               18,
               18,
               18,
               18,
               19,
               19,
               19,
               19,
               19,
               20,
               21])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 0, 2, 1, 3, 0, 2, 1, 3, 0, 7, 12, 13]),
             ('fyc', []),
             ('fy',
              [22,
               23,
               16,
               17,
               28,
               18,
               26,
               38,
               20,
               29,
               21,
               24,
               30,
               25,
               31,
               34,
               35]),
             ('gxc', []),
             ('gx', [0, 2, 4, 1, 3, 5, 6, 6, 12, 13, 7, 1, 8, 9, 10, 11]),
             ('gyc', [24, 25]),
             ('gy',
              [16,
               17,
               18,
               19,
               20,
               36,
               37,
               21,
               36,
               37,
               20,
               21,
               22,
               24,
               25,
               20,
               21,
               23,
               24,
               25,
               20,
               24,
               25,
               21,
               24,
               25,
               26,
               26,
               27,
               27,
               28,
               19,
               29,
               20,
               29,
               30,
               21,
               31,
               24,
               30,
               32,
               25,
               31,
               33,
               20,
               24,
               25,
               32,
               34,
               21,
               24,
               25,
               33,
               35,
               22,
               23])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
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

init_seq = ['Pe',
 'Psen_y',
 'Qe',
 'Qsen_y',
 'Paux',
 'Psig_y',
 'Qaux',
 'Qsig_y',
 'PIplim_xi',
 'PIqlim_xi',
 'a',
 'delta',
 'wref',
 'INTw_y',
 'PIvd_xi',
 'PIvq_xi',
 'PIId_xi',
 'PIIq_xi',
 'udref',
 'udLag_y',
 'uqref',
 'uqLag_y',
 'ud',
 'uq',
 'PIplim_y',
 'PIqlim_y',
 'vd',
 'vq',
 'Id',
 'Iq',
 'dw_x',
 'dw_y',
 'vref2',
 'PIvd_y',
 'PIvq_y',
 'PIId_y',
 'PIIq_y',
 'v',
 'plldw',
 'Idref',
 'Iqref']

need_diag_eps = ['Id', 'Iq', 'Id', 'Iq']
