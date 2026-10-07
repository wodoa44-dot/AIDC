from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "1b7278f4c3faae9bcf0cec7b55a445c4"

# empty f_update

def g_update(pi0, pp0, pz0, qi0, qp0, qz0, v):
    return (pi0*v + pp0 + pz0*v**2, qi0*v + qp0 + qz0*v**2,)


def gy_update(pi0, pz0, qi0, qz0, v):
    return (pi0 + 2*pz0*v, qi0 + 2*qz0*v)


def kps_svc(kpi, kpp, kpz):
    return kpi + kpp + kpz


def kqs_svc(kqi, kqp, kqz):
    return kqi + kqp + kqz


def rpp_svc(kpp, u):
    return (1/100)*kpp*u


def rpi_svc(kpi, u):
    return (1/100)*kpi*u


def rpz_svc(kpz, u):
    return (1/100)*kpz*u


def rqp_svc(kqp, u):
    return (1/100)*kqp*u


def rqi_svc(kqi, u):
    return (1/100)*kqi*u


def rqz_svc(kqz, u):
    return (1/100)*kqz*u


def pp0_svc(p0, rpp):
    return p0*rpp


def pi0_svc(p0, rpi, v0):
    return p0*rpi/v0


def pz0_svc(p0, rpz, v0):
    return p0*rpz/v0**2


def qp0_svc(q0, rqp):
    return q0*rqp


def qi0_svc(q0, rqi, v0):
    return q0*rqi/v0


def qz0_svc(q0, rqz, v0):
    return q0*rqz/v0**2


# empty sns_update

f_args = []

g_args = ['pi0', 'pp0', 'pz0', 'qi0', 'qp0', 'qz0', 'v']

j_args = {'gy': ['pi0', 'pz0', 'qi0', 'qz0', 'v']}

s_args = OrderedDict([('kps', ['kpi', 'kpp', 'kpz']),
             ('kqs', ['kqi', 'kqp', 'kqz']),
             ('rpp', ['kpp', 'u']),
             ('rpi', ['kpi', 'u']),
             ('rpz', ['kpz', 'u']),
             ('rqp', ['kqp', 'u']),
             ('rqi', ['kqi', 'u']),
             ('rqz', ['kqz', 'u']),
             ('pp0', ['p0', 'rpp']),
             ('pi0', ['p0', 'rpi', 'v0']),
             ('pz0', ['p0', 'rpz', 'v0']),
             ('qp0', ['q0', 'rqp']),
             ('qi0', ['q0', 'rqi', 'v0']),
             ('qz0', ['q0', 'rqz', 'v0'])])

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
             ('gy', [0, 1])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [1, 1])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0])])

j_names = ['gy']

init_seq = ['a', 'v']

need_diag_eps = []
