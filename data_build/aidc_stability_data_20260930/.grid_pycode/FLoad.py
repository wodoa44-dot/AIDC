from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "6cfbc994331345b2e5577e2014f3d56c"

# empty f_update

def g_update(ap, aq, bp, bq, f, pv0, qv0, v):
    return (0, f**bp*pv0*v**ap, f**bq*qv0*v**aq,)


def gy_update(ap, aq, bp, bq, f, pv0, qv0, v):
    return (bp*f**bp*pv0*v**ap/f, ap*f**bp*pv0*v**ap/v, bq*f**bq*qv0*v**aq/f, aq*f**bq*qv0*v**aq/v)


def pv0_svc(ap, kp, p0, u, v0):
    return (1/100)*kp*p0*u*v0**(-ap)


def qv0_svc(aq, kq, q0, u, v0):
    return (1/100)*kq*q0*u*v0**(-aq)


# empty sns_update

f_args = []

g_args = ['ap', 'aq', 'bp', 'bq', 'f', 'pv0', 'qv0', 'v']

j_args = {'gy': ['ap', 'aq', 'bp', 'bq', 'f', 'pv0', 'qv0', 'v']}

s_args = OrderedDict([('pv0', ['ap', 'kp', 'p0', 'u', 'v0']),
             ('qv0', ['aq', 'kq', 'q0', 'u', 'v0'])])

sns_args = []

ia_args = OrderedDict()

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [1, 1, 2, 2])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 2, 0, 2])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0, 0, 0])])

j_names = ['gy']

init_seq = ['f', 'a', 'v']

need_diag_eps = []
