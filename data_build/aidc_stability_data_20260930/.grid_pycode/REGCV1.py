from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "74a077b2efba49e5446c67d68d8dfe97"

def f_update(D, Id, Iq, KiId, KiIq, Kivd, Kivq, PIvd_y, PIvq_y, Pe, Pref2, dw, fn, udLag_y, udref, uqLag_y, uqref, vd, vq, vref2):
    return (-D*dw - Pe + Pref2, 2*pi*dw*fn, Kivd*(vd - vref2), Kivq*vq, KiId*(Id - PIvd_y), KiIq*(Iq - PIvq_y), -udLag_y + udref, -uqLag_y + uqref, 0, 0,)


def g_update(Id, Idref, Iq, Iqref, KpId, KpIq, Kpvd, Kpvq, PIId_xi, PIId_y, PIIq_xi, PIIq_y, PIvd_xi, PIvd_y, PIvq_xi, PIvq_y, Pe, Pref, Pref2, Qe, Qref, a, delta, dw, kv, kw, omega, ra, u, udLag_y, udref, uqLag_y, uqref, v, vd, vq, vref, vref2, xs):
    return (Pref*u - Pref2 - dw*kw, kv*(-Qe + Qref*u) + vref - vref2, dw - omega + 1, u*v*cos(a - delta) - vd, u*v*sin(a - delta) - vq, Id*vd + Iq*vq - Pe, Id*vq - Iq*vd - Qe, Id*ra - Iq*xs - udLag_y + vd, Id*xs + Iq*ra - uqLag_y + vq, Kpvd*(vd - vref2) + PIvd_xi - PIvd_y, Kpvq*vq + PIvq_xi - PIvq_y, KpId*(Id - PIvd_y) + PIId_xi - PIId_y, KpIq*(Iq - PIvq_y) + PIIq_xi - PIIq_y, -Iqref*xs + PIId_y - udref + vd, Idref*xs + PIIq_y - uqref + vq, -Pe*u, -Qe*u, 0, 0,)


def fx_update(D, fn):
    return (-D, 2*pi*fn, -1, -1)


def fy_update(KiId, KiIq, Kivd, Kivq):
    return (1, -1, -Kivd, Kivd, Kivq, KiId, -KiId, KiIq, -KiIq, 1, 1)


def gx_update(a, delta, kw, u, v):
    return (-kw, 1, u*v*sin(a - delta), -u*v*cos(a - delta), -1, -1, 1, 1, 1, 1)


def gy_update(Id, Iq, KpId, KpIq, Kpvd, Kpvq, a, delta, kv, ra, u, v, vd, vq, xs):
    return (-1, -1, -kv, -1, -1, -u*v*sin(a - delta), u*cos(a - delta), -1, u*v*cos(a - delta), u*sin(a - delta), Id, Iq, -1, vd, vq, -Iq, Id, -1, vq, -vd, 1, ra, -xs, 1, xs, ra, -Kpvd, Kpvd, -1, Kpvq, -1, KpId, -KpId, -1, KpIq, -KpIq, -1, 1, 1, -1, -xs, 1, 1, -1, xs, -u, -u)


def dw_ia():
    return 0


def delta_ia(a):
    return a


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


def Pref2_ia(Pref, u):
    return Pref*u


def vref2_ia(u, vref):
    return u*vref


def omega_ia(u):
    return u


def vd_ia(vd0):
    return vd0


def vq_ia(vq0):
    return vq0


def Pe_ia(Pref):
    return Pref


def Qe_ia(Qref):
    return Qref


def Id_ia(Id0):
    return Id0


def Iq_ia(Iq0):
    return Iq0


def PIvd_y_ia(Id0, Kpvd, vd, vref2):
    return Id0 + Kpvd*(vd - vref2)


def PIvq_y_ia(Iq0, Kpvq, vq):
    return Iq0 + Kpvq*vq


def PIId_y_ia(Id, KpId, PIvd_y):
    return KpId*(Id - PIvd_y)


def PIIq_y_ia(Iq, KpIq, PIvq_y):
    return KpIq*(Iq - PIvq_y)


def Pref_svc(gammap, p0s):
    return gammap*p0s


def Qref_svc(gammaq, q0s):
    return gammaq*q0s


def ixs_svc(xs):
    return xs**(-1.0)


def Id0_svc(Pref, u, v):
    return Pref*u/v


def Iq0_svc(Qref, u, v):
    return -Qref*u/v


def vd0_svc(u, v):
    return u*v


def vq0_svc():
    return 0


def udref0_svc(Id0, Iq0, ra, vd0, xs):
    return Id0*ra - Iq0*xs + vd0


def uqref0_svc(Id0, Iq0, ra, vq0, xs):
    return Id0*xs + Iq0*ra + vq0


# empty sns_update

f_args = ['D',
 'Id',
 'Iq',
 'KiId',
 'KiIq',
 'Kivd',
 'Kivq',
 'PIvd_y',
 'PIvq_y',
 'Pe',
 'Pref2',
 'dw',
 'fn',
 'udLag_y',
 'udref',
 'uqLag_y',
 'uqref',
 'vd',
 'vq',
 'vref2']

g_args = ['Id',
 'Idref',
 'Iq',
 'Iqref',
 'KpId',
 'KpIq',
 'Kpvd',
 'Kpvq',
 'PIId_xi',
 'PIId_y',
 'PIIq_xi',
 'PIIq_y',
 'PIvd_xi',
 'PIvd_y',
 'PIvq_xi',
 'PIvq_y',
 'Pe',
 'Pref',
 'Pref2',
 'Qe',
 'Qref',
 'a',
 'delta',
 'dw',
 'kv',
 'kw',
 'omega',
 'ra',
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
 'xs']

j_args = {'fx': ['D', 'fn'],
 'fy': ['KiId', 'KiIq', 'Kivd', 'Kivq'],
 'gx': ['a', 'delta', 'kw', 'u', 'v'],
 'gy': ['Id',
        'Iq',
        'KpId',
        'KpIq',
        'Kpvd',
        'Kpvq',
        'a',
        'delta',
        'kv',
        'ra',
        'u',
        'v',
        'vd',
        'vq',
        'xs']}

s_args = OrderedDict([('Pref', ['gammap', 'p0s']),
             ('Qref', ['gammaq', 'q0s']),
             ('ixs', ['xs']),
             ('Id0', ['Pref', 'u', 'v']),
             ('Iq0', ['Qref', 'u', 'v']),
             ('vd0', ['u', 'v']),
             ('vq0', []),
             ('udref0', ['Id0', 'Iq0', 'ra', 'vd0', 'xs']),
             ('uqref0', ['Id0', 'Iq0', 'ra', 'vq0', 'xs'])])

sns_args = []

ia_args = OrderedDict([('dw', []),
             ('delta', ['a']),
             ('PIvd_xi', ['Id0']),
             ('PIvq_xi', ['Iq0']),
             ('PIId_xi', []),
             ('PIIq_xi', []),
             ('udref', ['udref0']),
             ('udLag_y', ['udref']),
             ('uqref', ['uqref0']),
             ('uqLag_y', ['uqref']),
             ('Pref2', ['Pref', 'u']),
             ('vref2', ['u', 'vref']),
             ('omega', ['u']),
             ('vd', ['vd0']),
             ('vq', ['vq0']),
             ('Pe', ['Pref']),
             ('Qe', ['Qref']),
             ('Id', ['Id0']),
             ('Iq', ['Iq0']),
             ('PIvd_y', ['Id0', 'Kpvd', 'vd', 'vref2']),
             ('PIvq_y', ['Iq0', 'Kpvq', 'vq']),
             ('PIId_y', ['Id', 'KpId', 'PIvd_y']),
             ('PIIq_y', ['Iq', 'KpIq', 'PIvq_y'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 6, 7]),
             ('fyc', []),
             ('fy', [0, 0, 2, 2, 3, 4, 4, 5, 5, 6, 7]),
             ('gxc', []),
             ('gx', [0, 2, 3, 4, 7, 8, 9, 10, 11, 12]),
             ('gyc', [7, 8]),
             ('gy',
              [0,
               1,
               1,
               2,
               3,
               3,
               3,
               4,
               4,
               4,
               5,
               5,
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
               8,
               8,
               8,
               9,
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
               13,
               13,
               14,
               14,
               14,
               14,
               15,
               16])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 6, 7]),
             ('fyc', []),
             ('fy', [10, 15, 11, 13, 14, 17, 19, 18, 20, 23, 24]),
             ('gxc', []),
             ('gx', [0, 0, 1, 1, 6, 7, 2, 3, 4, 5]),
             ('gyc', [17, 18]),
             ('gy',
              [10,
               11,
               16,
               12,
               13,
               25,
               26,
               14,
               25,
               26,
               13,
               14,
               15,
               17,
               18,
               13,
               14,
               16,
               17,
               18,
               13,
               17,
               18,
               14,
               17,
               18,
               11,
               13,
               19,
               14,
               20,
               17,
               19,
               21,
               18,
               20,
               22,
               13,
               21,
               23,
               28,
               14,
               22,
               24,
               27,
               15,
               16])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
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
               0])])

j_names = ['fx', 'fy', 'gx', 'gy']

init_seq = ['dw',
 'a',
 'delta',
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
 'Pref2',
 'vref2',
 'omega',
 'vd',
 'vq',
 'Pe',
 'Qe',
 'Id',
 'Iq',
 'PIvd_y',
 'PIvq_y',
 'PIId_y',
 'PIIq_y',
 'v',
 'Idref',
 'Iqref']

need_diag_eps = ['Id', 'Iq', 'Id', 'Iq']
