from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "cf5df6cdfc4ee3a2f5f75ee8bb60966d"

def f_update(D, fn, omega, te, tm, u):
    return (2*pi*fn*u*(omega - 1), u*(-D*(omega - 1) - te + tm),)


def g_update(Id, Iq, Pe, Qe, XadIfd, a, delta, psid, psiq, ra, te, tm, tm0, u, v, vd, vf, vf0, vq, xq):
    return (Id*xq + psid - vf, Iq*xq + psiq, -u*v*sin(a - delta) - vd, u*v*cos(a - delta) - vq, -tm + tm0, -te + u*(-Id*psiq + Iq*psid), u*vf0 - vf, -XadIfd + u*vf0, -Pe + u*(Id*vd + Iq*vq), -Qe + u*(Id*vq - Iq*vd), -psid + u*(Iq*ra + vq), psiq + u*(Id*ra + vd), -u*(Id*vd + Iq*vq), -u*(Id*vq - Iq*vd),)


def fx_update(D, fn, u):
    return (2*pi*fn*u, -D*u)


def fy_update(u):
    return (u, -u)


def gy_update(Id, Iq, a, delta, psid, psiq, ra, u, v, vd, vq, xq):
    return (xq, -1, 1, xq, 1, -1, -u*v*cos(a - delta), -u*sin(a - delta), -1, -u*v*sin(a - delta), u*cos(a - delta), -1, -psiq*u, psid*u, -1, Iq*u, -Id*u, -1, -1, u*vd, u*vq, Id*u, Iq*u, -1, u*vq, -u*vd, -Iq*u, Id*u, -1, ra*u, u, -1, ra*u, u, 1, -u*vd, -u*vq, -Id*u, -Iq*u, -u*vq, u*vd, Iq*u, -Id*u)


def gx_update(a, delta, u, v):
    return (u*v*cos(a - delta), u*v*sin(a - delta))


def delta_ia(delta0):
    return delta0


def omega_ia(u):
    return u


def Id_ia(Id0, u):
    return Id0*u


def Iq_ia(Iq0, u):
    return Iq0*u


def vd_ia(u, vd0):
    return u*vd0


def vq_ia(u, vq0):
    return u*vq0


def tm_ia(tm0):
    return tm0


def te_ia(tm0, u):
    return tm0*u


def vf_ia(u, vf0):
    return u*vf0


def XadIfd_ia(u, vf0):
    return u*vf0


def Pe_ia(Id0, Iq0, u, vd0, vq0):
    return u*(Id0*vd0 + Iq0*vq0)


def Qe_ia(Id0, Iq0, u, vd0, vq0):
    return u*(Id0*vq0 - Iq0*vd0)


def psid_ia(psid0, u):
    return psid0*u


def psiq_ia(psiq0, u):
    return psiq0*u


def p0_svc(gammap, p0s):
    return gammap*p0s


def q0_svc(gammaq, q0s):
    return gammaq*q0s


def _V_svc(a, v):
    return v*exp(1j*a)


def _S_svc(p0, q0):
    return p0 - 1j*q0


def _I_svc(_S, _V):
    return _S/conj(_V)


def _E_svc(_I, _V, ra, xq):
    return _I*(ra + 1j*xq) + _V


def _deltac_svc(_E):
    return log(_E/abs(_E))


def delta0_svc(_deltac, u):
    return u*imag(_deltac)


def vdq_svc(_V, _deltac, u):
    return _V*u*exp(-_deltac + 0.5*1j*pi)


def Idq_svc(_I, _deltac, u):
    return _I*u*exp(-_deltac + 0.5*1j*pi)


def Id0_svc(Idq):
    return real(Idq)


def Iq0_svc(Idq):
    return imag(Idq)


def vd0_svc(vdq):
    return real(vdq)


def vq0_svc(vdq):
    return imag(vdq)


def tm0_svc(Id0, Iq0, ra, u, vd0, vq0):
    return u*(Id0*(Id0*ra + vd0) + Iq0*(Iq0*ra + vq0))


def psid0_svc(Iq0, ra, u, vq0):
    return Iq0*ra*u + vq0


def psiq0_svc(Id0, ra, u, vd0):
    return -Id0*ra*u - vd0


def vf0_svc(Id0, Iq0, ra, vq0, xq):
    return Id0*xq + Iq0*ra + vq0


# empty sns_update

f_args = ['D', 'fn', 'omega', 'te', 'tm', 'u']

g_args = ['Id',
 'Iq',
 'Pe',
 'Qe',
 'XadIfd',
 'a',
 'delta',
 'psid',
 'psiq',
 'ra',
 'te',
 'tm',
 'tm0',
 'u',
 'v',
 'vd',
 'vf',
 'vf0',
 'vq',
 'xq']

j_args = {'fx': ['D', 'fn', 'u'],
 'fy': ['u'],
 'gx': ['a', 'delta', 'u', 'v'],
 'gy': ['Id',
        'Iq',
        'a',
        'delta',
        'psid',
        'psiq',
        'ra',
        'u',
        'v',
        'vd',
        'vq',
        'xq']}

s_args = OrderedDict([('p0', ['gammap', 'p0s']),
             ('q0', ['gammaq', 'q0s']),
             ('_V', ['a', 'v']),
             ('_S', ['p0', 'q0']),
             ('_I', ['_S', '_V']),
             ('_E', ['_I', '_V', 'ra', 'xq']),
             ('_deltac', ['_E']),
             ('delta0', ['_deltac', 'u']),
             ('vdq', ['_V', '_deltac', 'u']),
             ('Idq', ['_I', '_deltac', 'u']),
             ('Id0', ['Idq']),
             ('Iq0', ['Idq']),
             ('vd0', ['vdq']),
             ('vq0', ['vdq']),
             ('tm0', ['Id0', 'Iq0', 'ra', 'u', 'vd0', 'vq0']),
             ('psid0', ['Iq0', 'ra', 'u', 'vq0']),
             ('psiq0', ['Id0', 'ra', 'u', 'vd0']),
             ('vf0', ['Id0', 'Iq0', 'ra', 'vq0', 'xq'])])

sns_args = []

ia_args = OrderedDict([('delta', ['delta0']),
             ('omega', ['u']),
             ('Id', ['Id0', 'u']),
             ('Iq', ['Iq0', 'u']),
             ('vd', ['u', 'vd0']),
             ('vq', ['u', 'vq0']),
             ('tm', ['tm0']),
             ('te', ['tm0', 'u']),
             ('vf', ['u', 'vf0']),
             ('XadIfd', ['u', 'vf0']),
             ('Pe', ['Id0', 'Iq0', 'u', 'vd0', 'vq0']),
             ('Qe', ['Id0', 'Iq0', 'u', 'vd0', 'vq0']),
             ('psid', ['psid0', 'u']),
             ('psiq', ['psiq0', 'u'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1]),
             ('fyc', []),
             ('fy', [1, 1]),
             ('gxc', []),
             ('gx', [2, 3]),
             ('gyc', []),
             ('gy',
              [0,
               0,
               0,
               1,
               1,
               2,
               2,
               2,
               3,
               3,
               3,
               4,
               5,
               5,
               5,
               5,
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
               9,
               9,
               9,
               10,
               10,
               10,
               11,
               11,
               11,
               12,
               12,
               12,
               12,
               13,
               13,
               13,
               13])])

jjac = OrderedDict([('fxc', []),
             ('fx', [1, 1]),
             ('fyc', []),
             ('fy', [6, 7]),
             ('gxc', []),
             ('gx', [0, 0]),
             ('gyc', []),
             ('gy',
              [2,
               8,
               12,
               3,
               13,
               4,
               14,
               15,
               5,
               14,
               15,
               6,
               2,
               3,
               7,
               12,
               13,
               8,
               9,
               2,
               3,
               4,
               5,
               10,
               2,
               3,
               4,
               5,
               11,
               3,
               5,
               12,
               2,
               4,
               13,
               2,
               3,
               4,
               5,
               2,
               3,
               4,
               5])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0]),
             ('fyc', []),
             ('fy', [0, 0]),
             ('gxc', []),
             ('gx', [0, 0]),
             ('gyc', []),
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
               0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['delta',
 'omega',
 'Id',
 'Iq',
 'vd',
 'vq',
 'tm',
 'te',
 'vf',
 'XadIfd',
 'Pe',
 'Qe',
 'psid',
 'psiq',
 'a',
 'v']

need_diag_eps = []
