from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "ad93d11d9662a20355b7b0a4875032ba"

# empty f_update

def g_update(a, a0, p, plim_zi, plim_zl, plim_zu, pmax, pmin, q, qlim_zi, qlim_zl, qlim_zu, qmax, qmin, u, v, v0):
    return (u*(qlim_zi*(-v + v0) + qlim_zl*(-q + qmin) + qlim_zu*(-q + qmax)), u*(plim_zi*(-a + a0) + plim_zl*(-p + pmin) + plim_zu*(-p + pmax)), -p*u, -q*u,)


def gy_update(plim_zi, plim_zl, plim_zu, qlim_zi, qlim_zl, qlim_zu, u):
    return (u*(-qlim_zl - qlim_zu), -qlim_zi*u, u*(-plim_zl - plim_zu), -plim_zi*u, -u, -u)


def q_ia(q0, u):
    return q0*u


def p_ia(p0, u):
    return p0*u


def a_ia(a0, busa0, u):
    return a0*u + busa0*(1 - u)


def v_ia(busv0, u, v0):
    return busv0*(1 - u) + u*v0


# empty sns_update

f_args = []

g_args = ['a',
 'a0',
 'p',
 'plim_zi',
 'plim_zl',
 'plim_zu',
 'pmax',
 'pmin',
 'q',
 'qlim_zi',
 'qlim_zl',
 'qlim_zu',
 'qmax',
 'qmin',
 'u',
 'v',
 'v0']

j_args = {'gy': ['plim_zi', 'plim_zl', 'plim_zu', 'qlim_zi', 'qlim_zl', 'qlim_zu', 'u']}

s_args = OrderedDict()

sns_args = []

ia_args = OrderedDict([('q', ['q0', 'u']),
             ('p', ['p0', 'u']),
             ('a', ['a0', 'busa0', 'u']),
             ('v', ['busv0', 'u', 'v0'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [0, 1]),
             ('gy', [0, 0, 1, 1, 2, 3])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [0, 1]),
             ('gy', [0, 3, 1, 2, 1, 0])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [1e-08, 1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0])])

j_names = ['gy']

init_seq = ['q', 'p', 'a', 'v']

need_diag_eps = ['q', 'p', 'q', 'p']
