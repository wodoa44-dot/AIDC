from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "b92937ff4bcd34933417e6a15f49e984"

def f_update(Id, Iq, T10, T20, e1d, e1q, e2d, e2q, slip, te, tm, u, wb, x0, x1, x2):
    return (u*(-te + tm), u*(e1q*slip*wb - (Iq*(x0 - x1) + e1d)/T10), u*(-e1d*slip*wb - (-Id*(x0 - x1) + e1q)/T10), u*(e1q*slip*wb - slip*wb*(e1q - e2q) + (-Iq*(x1 - x2) + e1d - e2d)/T20 - (Iq*(x0 - x1) + e1d)/T10), u*(-e1d*slip*wb + slip*wb*(e1d - e2d) + (Id*(x1 - x2) + e1q - e2q)/T20 - (-Id*(x0 - x1) + e1q)/T10),)


def g_update(Id, Iq, a, aa, bb, c2, e2d, e2q, p, q, rs, slip, te, tm, u, v, vd, vq, x2):
    return (-u*v*sin(a) - vd, u*v*cos(a) - vq, -p + u*(Id*vd + Iq*vq), -q + u*(Id*vq - Iq*vd), u*(-Id*rs + Iq*x2 - e2d + vd), u*(-Id*x2 - Iq*rs - e2q + vq), -te + u*(Id*e2d + Iq*e2q), -tm + u*(aa + bb*slip + c2*slip**2), p, q,)


def fy_update(T10, T20, u, x0, x1, x2):
    return (-u, u, -u*(x0 - x1)/T10, -u*(-x0 + x1)/T10, u*((-x1 + x2)/T20 - (x0 - x1)/T10), u*((x1 - x2)/T20 - (-x0 + x1)/T10))


def fx_update(T10, T20, e1d, e1q, e2d, e2q, slip, u, wb):
    return (e1q*u*wb, -u/T10, slip*u*wb, -e1d*u*wb, -slip*u*wb, -u/T10, u*(e1q*wb - wb*(e1q - e2q)), u*(T20**(-1.0) - 1/T10), -u/T20, slip*u*wb, u*(-e1d*wb + wb*(e1d - e2d)), u*(T20**(-1.0) - 1/T10), -slip*u*wb, -u/T20)


def gy_update(Id, Iq, a, e2d, e2q, rs, u, v, vd, vq, x2):
    return (-1, -u*v*cos(a), -u*sin(a), -1, -u*v*sin(a), u*cos(a), Id*u, Iq*u, -1, u*vd, u*vq, -Iq*u, Id*u, -1, u*vq, -u*vd, u, -rs*u, u*x2, u, -u*x2, -rs*u, e2d*u, e2q*u, -1, -1, 1, 1)


def gx_update(Id, Iq, bb, c2, slip, u):
    return (-u, -u, Id*u, Iq*u, u*(bb + 2*c2*slip))


def slip_ia(u):
    return 1.0*u


def e1d_ia(u):
    return 0.05*u


def e1q_ia(u):
    return 0.9*u


def e2d_ia(u):
    return 0.05*u


def e2q_ia(u):
    return 0.9*u


def Id_ia(u):
    return 0.9*u


def Iq_ia(u):
    return 0.1*u


def p_ia(Id, Iq, u, vd, vq):
    return u*(Id*vd + Iq*vq)


def q_ia(Id, Iq, u, vd, vq):
    return u*(Id*vq - Iq*vd)


def te_ia(Id, Iq, e2d, e2q, u):
    return u*(Id*e2d + Iq*e2q)


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


def x2_svc(xm, xr1, xr2, xs):
    return xm*xr1*xr2/(xm*xr1 + xm*xr2 + xr1*xr2) + xs


def T20_svc(rr2, wb, xm, xr1, xr2):
    return (xm*xr1/(xm + xr1) + xr2)/(rr2*wb)


# empty sns_update

f_args = ['Id',
 'Iq',
 'T10',
 'T20',
 'e1d',
 'e1q',
 'e2d',
 'e2q',
 'slip',
 'te',
 'tm',
 'u',
 'wb',
 'x0',
 'x1',
 'x2']

g_args = ['Id',
 'Iq',
 'a',
 'aa',
 'bb',
 'c2',
 'e2d',
 'e2q',
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
 'x2']

j_args = {'fx': ['T10', 'T20', 'e1d', 'e1q', 'e2d', 'e2q', 'slip', 'u', 'wb'],
 'fy': ['T10', 'T20', 'u', 'x0', 'x1', 'x2'],
 'gx': ['Id', 'Iq', 'bb', 'c2', 'slip', 'u'],
 'gy': ['Id', 'Iq', 'a', 'e2d', 'e2q', 'rs', 'u', 'v', 'vd', 'vq', 'x2']}

s_args = OrderedDict([('wb', ['fn']),
             ('x0', ['xm', 'xs']),
             ('x1', ['xm', 'xr1', 'xs']),
             ('T10', ['rr1', 'wb', 'xm', 'xr1']),
             ('M', ['Hm']),
             ('aa', ['c1', 'c2', 'c3']),
             ('bb', ['c2', 'c3']),
             ('x2', ['xm', 'xr1', 'xr2', 'xs']),
             ('T20', ['rr2', 'wb', 'xm', 'xr1', 'xr2'])])

sns_args = []

ia_args = OrderedDict([('slip', ['u']),
             ('e1d', ['u']),
             ('e1q', ['u']),
             ('e2d', ['u']),
             ('e2q', ['u']),
             ('Id', ['u']),
             ('Iq', ['u']),
             ('p', ['Id', 'Iq', 'u', 'vd', 'vq']),
             ('q', ['Id', 'Iq', 'u', 'vd', 'vq']),
             ('te', ['Id', 'Iq', 'e2d', 'e2q', 'u']),
             ('tm', ['aa', 'bb', 'c2', 'slip', 'u'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', [0, 1, 2, 3, 4]),
             ('fx', [1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]),
             ('fyc', []),
             ('fy', [0, 0, 1, 2, 3, 4]),
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

jjac = OrderedDict([('fxc', [0, 1, 2, 3, 4]),
             ('fx', [0, 1, 2, 0, 1, 2, 0, 1, 3, 4, 0, 2, 3, 4]),
             ('fyc', []),
             ('fy', [11, 12, 10, 9, 10, 9]),
             ('gxc', []),
             ('gx', [3, 4, 3, 4, 0]),
             ('gyc', [9, 10]),
             ('gy',
              [5,
               13,
               14,
               6,
               13,
               14,
               5,
               6,
               7,
               9,
               10,
               5,
               6,
               8,
               9,
               10,
               5,
               9,
               10,
               6,
               9,
               10,
               9,
               10,
               11,
               12,
               7,
               8])])

vjac = OrderedDict([('fxc', [1e-08, 1e-08, 1e-08, 1e-08, 1e-08]),
             ('fx', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0, 0]),
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

init_seq = ['slip',
 'e1d',
 'e1q',
 'e2d',
 'e2q',
 'vd',
 'vq',
 'Id',
 'Iq',
 'p',
 'q',
 'te',
 'tm',
 'a',
 'v']

need_diag_eps = ['slip',
 'e1d',
 'e1q',
 'e2d',
 'e2q',
 'Id',
 'Iq',
 'slip',
 'e1d',
 'e1q',
 'e2d',
 'e2q',
 'Id',
 'Iq']
