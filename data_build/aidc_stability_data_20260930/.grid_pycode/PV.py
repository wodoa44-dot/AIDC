from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "777e4013beb1b3b82c9c14f4ee4442b3"

# empty f_update

def g_update(p, q, qlim_zi, qlim_zl, qlim_zu, qmax, qmin, u, v, v0):
    return (u*(qlim_zi*(-v + v0) + qlim_zl*(-q + qmin) + qlim_zu*(-q + qmax)), -p*u, -q*u,)


def gy_update(qlim_zi, qlim_zl, qlim_zu, u):
    return (u*(-qlim_zl - qlim_zu), -qlim_zi*u, -u)


def q_ia(q0, u):
    return q0*u


def v_ia(busv0, u, v0):
    return busv0*(1 - u) + u*v0


def p_svc(p0):
    return p0


# empty sns_update

f_args = []

g_args = ['p', 'q', 'qlim_zi', 'qlim_zl', 'qlim_zu', 'qmax', 'qmin', 'u', 'v', 'v0']

j_args = {'gy': ['qlim_zi', 'qlim_zl', 'qlim_zu', 'u']}

s_args = OrderedDict([('p', ['p0'])])

sns_args = []

ia_args = OrderedDict([('q', ['q0', 'u']), ('v', ['busv0', 'u', 'v0'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [0]),
             ('gy', [0, 0, 2])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [0]),
             ('gy', [0, 2, 0])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [1e-08]),
             ('gy', [0, 0, 0])])

j_names = ['gy']

init_seq = ['q', 'a', 'v']

need_diag_eps = ['q', 'q']
