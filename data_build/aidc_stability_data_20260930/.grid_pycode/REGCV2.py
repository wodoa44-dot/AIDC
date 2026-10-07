from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "a7f0fd3b51de4d4f04d55d56aec61bb5"

def f_update(D, Kivd, Kivq, LGId_y, LGIq_y, PIvd_y, PIvq_y, Pe, Pref2, dw, fn, vd, vq, vref2):
    return (-D*dw - Pe + Pref2, 2*pi*dw*fn, Kivd*(vd - vref2), Kivq*vq, -LGId_y + PIvd_y, -LGIq_y + PIvq_y,)


def g_update(Id, Iq, Kpvd, Kpvq, LGId_y, LGIq_y, PIvd_xi, PIvd_y, PIvq_xi, PIvq_y, Pe, Pref, Pref2, Qe, Qref, a, delta, dw, kv, kw, omega, u, v, vd, vq, vref, vref2):
    return (Pref*u - Pref2 - dw*kw, kv*(-Qe + Qref*u) + vref - vref2, dw - omega + 1, u*v*cos(a - delta) - vd, u*v*sin(a - delta) - vq, Id*vd + Iq*vq - Pe, Id*vq - Iq*vd - Qe, -Id + LGId_y, -Iq + LGIq_y, Kpvd*(vd - vref2) + PIvd_xi - PIvd_y, Kpvq*vq + PIvq_xi - PIvq_y, -Pe*u, -Qe*u, 0, 0,)


def fx_update(D, fn):
    return (-D, 2*pi*fn, -1, -1)


def fy_update(Kivd, Kivq):
    return (1, -1, -Kivd, Kivd, Kivq, 1, 1)


def gx_update(a, delta, kw, u, v):
    return (-kw, 1, u*v*sin(a - delta), -u*v*cos(a - delta), 1, 1, 1, 1)


def gy_update(Id, Iq, Kpvd, Kpvq, a, delta, kv, u, v, vd, vq):
    return (-1, -1, -kv, -1, -1, -u*v*sin(a - delta), u*cos(a - delta), -1, u*v*cos(a - delta), u*sin(a - delta), Id, Iq, -1, vd, vq, -Iq, Id, -1, vq, -vd, -1, -1, -Kpvd, Kpvd, -1, Kpvq, -1, -u, -u)


def dw_ia():
    return 0


def delta_ia(a):
    return a


def PIvd_xi_ia(Id0):
    return Id0


def PIvq_xi_ia(Iq0):
    return Iq0


def vd_ia(vd0):
    return vd0


def vref2_ia(u, vref):
    return u*vref


def PIvd_y_ia(Id0, Kpvd, vd, vref2):
    return Id0 + Kpvd*(vd - vref2)


def LGId_y_ia(PIvd_y):
    return PIvd_y


def vq_ia(vq0):
    return vq0


def PIvq_y_ia(Iq0, Kpvq, vq):
    return Iq0 + Kpvq*vq


def LGIq_y_ia(PIvq_y):
    return PIvq_y


def Pref2_ia(Pref, u):
    return Pref*u


def omega_ia(u):
    return u


def Pe_ia(Pref):
    return Pref


def Qe_ia(Qref):
    return Qref


def Id_ia(Id0):
    return Id0


def Iq_ia(Iq0):
    return Iq0


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


# empty sns_update

f_args = ['D',
 'Kivd',
 'Kivq',
 'LGId_y',
 'LGIq_y',
 'PIvd_y',
 'PIvq_y',
 'Pe',
 'Pref2',
 'dw',
 'fn',
 'vd',
 'vq',
 'vref2']

g_args = ['Id',
 'Iq',
 'Kpvd',
 'Kpvq',
 'LGId_y',
 'LGIq_y',
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
 'u',
 'v',
 'vd',
 'vq',
 'vref',
 'vref2']

j_args = {'fx': ['D', 'fn'],
 'fy': ['Kivd', 'Kivq'],
 'gx': ['a', 'delta', 'kw', 'u', 'v'],
 'gy': ['Id', 'Iq', 'Kpvd', 'Kpvq', 'a', 'delta', 'kv', 'u', 'v', 'vd', 'vq']}

s_args = OrderedDict([('Pref', ['gammap', 'p0s']),
             ('Qref', ['gammaq', 'q0s']),
             ('ixs', ['xs']),
             ('Id0', ['Pref', 'u', 'v']),
             ('Iq0', ['Qref', 'u', 'v']),
             ('vd0', ['u', 'v']),
             ('vq0', [])])

sns_args = []

ia_args = OrderedDict([('dw', []),
             ('delta', ['a']),
             ('PIvd_xi', ['Id0']),
             ('PIvq_xi', ['Iq0']),
             ('vd', ['vd0']),
             ('vref2', ['u', 'vref']),
             ('PIvd_y', ['Id0', 'Kpvd', 'vd', 'vref2']),
             ('LGId_y', ['PIvd_y']),
             ('vq', ['vq0']),
             ('PIvq_y', ['Iq0', 'Kpvq', 'vq']),
             ('LGIq_y', ['PIvq_y']),
             ('Pref2', ['Pref', 'u']),
             ('omega', ['u']),
             ('Pe', ['Pref']),
             ('Qe', ['Qref']),
             ('Id', ['Id0']),
             ('Iq', ['Iq0'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 4, 5]),
             ('fyc', []),
             ('fy', [0, 0, 2, 2, 3, 4, 5]),
             ('gxc', []),
             ('gx', [0, 2, 3, 4, 7, 8, 9, 10]),
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
               8,
               9,
               9,
               9,
               10,
               10,
               11,
               12])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 4, 5]),
             ('fyc', []),
             ('fy', [6, 11, 7, 9, 10, 15, 16]),
             ('gxc', []),
             ('gx', [0, 0, 1, 1, 4, 5, 2, 3]),
             ('gyc', [13, 14]),
             ('gy',
              [6,
               7,
               12,
               8,
               9,
               17,
               18,
               10,
               17,
               18,
               9,
               10,
               11,
               13,
               14,
               9,
               10,
               12,
               13,
               14,
               13,
               14,
               7,
               9,
               15,
               10,
               16,
               11,
               12])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0, 0]),
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
               0])])

j_names = ['fx', 'fy', 'gx', 'gy']

init_seq = ['dw',
 'a',
 'delta',
 'PIvd_xi',
 'PIvq_xi',
 'vd',
 'vref2',
 'PIvd_y',
 'LGId_y',
 'vq',
 'PIvq_y',
 'LGIq_y',
 'Pref2',
 'omega',
 'Pe',
 'Qe',
 'Id',
 'Iq',
 'v',
 'Idref',
 'Iqref']

need_diag_eps = ['Id', 'Iq', 'Id', 'Iq']
