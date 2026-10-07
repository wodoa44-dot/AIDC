from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "8f12e91070e1a7f0162d2c9a95d4697c"

def f_update(Id, Iq, T10, e1d, e1q, slip, te, tm, u, wb, x0, x1):
    return (u*(-te + tm), u*(e1q*slip*wb - (Iq*(x0 - x1) + e1d)/T10), u*(-e1d*slip*wb - (-Id*(x0 - x1) + e1q)/T10),)


def g_update(Id, Iq, a, aa, bb, c2, e1d, e1q, p, q, rs, slip, te, tm, u, v, vd, vq, x1):
    return (-u*v*sin(a) - vd, u*v*cos(a) - vq, -p + u*(Id*vd + Iq*vq), -q + u*(Id*vq - Iq*vd), u*(-Id*rs + Iq*x1 - e1d + vd), u*(-Id*x1 - Iq*rs - e1q + vq), -te + u*(Id*e1d + Iq*e1q), -tm + u*(aa + bb*slip + c2*slip**2), p, q,)


def fy_update(T10, u, x0, x1):
    return (-u, u, -u*(x0 - x1)/T10, -u*(-x0 + x1)/T10)


def fx_update(T10, e1d, e1q, slip, u, wb):
    return (e1q*u*wb, -u/T10, slip*u*wb, -e1d*u*wb, -slip*u*wb, -u/T10)


def gy_update(Id, Iq, a, e1d, e1q, rs, u, v, vd, vq, x1):
    return (-1, -u*v*cos(a), -u*sin(a), -1, -u*v*sin(a), u*cos(a), Id*u, Iq*u, -1, u*vd, u*vq, -Iq*u, Id*u, -1, u*vq, -u*vd, u, -rs*u, u*x1, u, -u*x1, -rs*u, e1d*u, e1q*u, -1, -1, 1, 1)


def gx_update(Id, Iq, bb, c2, slip, u):
    return (-u, -u, Id*u, Iq*u, u*(bb + 2*c2*slip))


def slip_ia(u):
    return 1.0*u


def e1d_ia(u):
    return 0.05*u


def e1q_ia(u):
    return 0.9*u


def Id_ia():
    return 1


def p_ia(Id, Iq, u, vd, vq):
    return u*(Id*vd + Iq*vq)


def q_ia(Id, Iq, u, vd, vq):
    return u*(Id*vq - Iq*vd)


def te_ia(Id, Iq, e1d, e1q, u):
    return u*(Id*e1d + Iq*e1q)


def tm_ia(aa, bb, c2, slip, u):
    return u*(aa + bb*slip + c2*slip**2)


def wb_svc(fn):
    return 2*pi*fn


def x0_svc(xm, xs):
    return xm + xs


def x1_svc(xm, xr1, xs):
    return xm*xr1/(xm + xr1) + xs


def T10_svc(rr1, wb, xm, xr1):
    return (xm + xr1)/(rr1*wb)


def M_svc(Hm):
    return 2*Hm


def aa_svc(c1, c2, c3):
    return c1 + c2 + c3


def bb_svc(c2, c3):
    return -c2 - 2*c3


# empty sns_update

f_args = ['Id', 'Iq', 'T10', 'e1d', 'e1q', 'slip', 'te', 'tm', 'u', 'wb', 'x0', 'x1']

g_args = ['Id',
 'Iq',
 'a',
 'aa',
 'bb',
 'c2',
 'e1d',
 'e1q',
 'p',
 'q',
 'rs',
 'slip',
 'te',
 'tm',
 'u',
 'v',
 'vd',
 'vq',
 'x1']

j_args = {'fx': ['T10', 'e1d', 'e1q', 'slip', 'u', 'wb'],
 'fy': ['T10', 'u', 'x0', 'x1'],
 'gx': ['Id', 'Iq', 'bb', 'c2', 'slip', 'u'],
 'gy': ['Id', 'Iq', 'a', 'e1d', 'e1q', 'rs', 'u', 'v', 'vd', 'vq', 'x1']}

s_args = OrderedDict([('wb', ['fn']),
             ('x0', ['xm', 'xs']),
             ('x1', ['xm', 'xr1', 'xs']),
             ('T10', ['rr1', 'wb', 'xm', 'xr1']),
             ('M', ['Hm']),
             ('aa', ['c1', 'c2', 'c3']),
             ('bb', ['c2', 'c3'])])

sns_args = []

ia_args = OrderedDict([('slip', ['u']),
             ('e1d', ['u']),
             ('e1q', ['u']),
             ('Id', []),
             ('p', ['Id', 'Iq', 'u', 'vd', 'vq']),
             ('q', ['Id', 'Iq', 'u', 'vd', 'vq']),
             ('te', ['Id', 'Iq', 'e1d', 'e1q', 'u']),
             ('tm', ['aa', 'bb', 'c2', 'slip', 'u'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', [0, 1, 2]),
             ('fx', [1, 1, 1, 2, 2, 2]),
             ('fyc', []),
             ('fy', [0, 0, 1, 2]),
             ('gxc', []),
             ('gx', [4, 5, 6, 6, 7]),
             ('gyc', [4, 5]),
             ('gy',
              [0,
               0,
               0,
               1,
               1,
               1,
               2,
               2,
               2,
               2,
               2,
               3,
               3,
               3,
               3,
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
               7,
               8,
               9])])

jjac = OrderedDict([('fxc', [0, 1, 2]),
             ('fx', [0, 1, 2, 0, 1, 2]),
             ('fyc', []),
             ('fy', [9, 10, 8, 7]),
             ('gxc', []),
             ('gx', [1, 2, 1, 2, 0]),
             ('gyc', [7, 8]),
             ('gy',
              [3,
               11,
               12,
               4,
               11,
               12,
               3,
               4,
               5,
               7,
               8,
               3,
               4,
               6,
               7,
               8,
               3,
               7,
               8,
               4,
               7,
               8,
               7,
               8,
               9,
               10,
               5,
               6])])

vjac = OrderedDict([('fxc', [1e-08, 1e-08, 1e-08]),
             ('fx', [0, 0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0]),
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
               0])])

j_names = ['fy', 'fx', 'gy', 'gx']

init_seq = ['slip', 'e1d', 'e1q', 'vd', 'vq', 'Id', 'Iq', 'p', 'q', 'te', 'tm', 'a', 'v']

need_diag_eps = ['slip', 'e1d', 'e1q', 'Id', 'Iq', 'slip', 'e1d', 'e1q', 'Id', 'Iq']
