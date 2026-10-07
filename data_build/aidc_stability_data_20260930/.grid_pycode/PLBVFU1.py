from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "2a20e11f68e24ab44906336e9ac286d2"

def f_update(Vflt, Voffs, Vts, fn, foffs, fts, iVscale, ifscale, omega, u):
    return (-Vflt - Voffs + Vts*iVscale, -foffs + fts*ifscale - omega, 2*pi*fn*u*(omega - 1),)


def g_update(Vflt, a, delta, ra, v, xs):
    return (Vflt*v*xs*sin(a - delta)/(ra**2 + xs**2) + ra*v*(-Vflt*cos(a - delta) + v)/(ra**2 + xs**2), -Vflt*ra*v*sin(a - delta)/(ra**2 + xs**2) + v*xs*(-Vflt*cos(a - delta) + v)/(ra**2 + xs**2),)


def fx_update(fn, u):
    return (-1, -1, 2*pi*fn*u)


def gx_update(Vflt, a, delta, ra, v, xs):
    return (-ra*v*cos(a - delta)/(ra**2 + xs**2) + v*xs*sin(a - delta)/(ra**2 + xs**2), -Vflt*ra*v*sin(a - delta)/(ra**2 + xs**2) - Vflt*v*xs*cos(a - delta)/(ra**2 + xs**2), -ra*v*sin(a - delta)/(ra**2 + xs**2) - v*xs*cos(a - delta)/(ra**2 + xs**2), Vflt*ra*v*cos(a - delta)/(ra**2 + xs**2) - Vflt*v*xs*sin(a - delta)/(ra**2 + xs**2))


def gy_update(Vflt, a, delta, ra, v, xs):
    return (Vflt*ra*v*sin(a - delta)/(ra**2 + xs**2) + Vflt*v*xs*cos(a - delta)/(ra**2 + xs**2), Vflt*xs*sin(a - delta)/(ra**2 + xs**2) + ra*v/(ra**2 + xs**2) + ra*(-Vflt*cos(a - delta) + v)/(ra**2 + xs**2), -Vflt*ra*v*cos(a - delta)/(ra**2 + xs**2) + Vflt*v*xs*sin(a - delta)/(ra**2 + xs**2), -Vflt*ra*sin(a - delta)/(ra**2 + xs**2) + v*xs/(ra**2 + xs**2) + xs*(-Vflt*cos(a - delta) + v)/(ra**2 + xs**2))


def Vflt_ia(Voffs, Vts, iVscale):
    return -Voffs + Vts*iVscale


def omega_ia(foffs, fts, ifscale):
    return -foffs + fts*ifscale


def delta_ia(delta0):
    return delta0


def zs_svc(ra, xs):
    return ra + 1j*xs


def zs2n_svc(ra, xs):
    return ra**2 - xs**2


def Ec_svc(a, p, q, ra, v, xs):
    return v*exp(1j*a) + (ra + 1j*xs)*conj((p + 1j*q)*exp(-1j*a)/v)


def E0_svc(Ec):
    return abs(Ec)


def delta0_svc(Ec):
    return angle(Ec)


def Vts_svc():
    return 0


def fts_svc():
    return 0


def ifscale_svc(fscale):
    return fscale**(-1.0)


def iVscale_svc(Vscale):
    return Vscale**(-1.0)


def foffs_svc(fts, ifscale):
    return fts*ifscale - 1


def Voffs_svc(E0, Vts, iVscale):
    return -E0 + Vts*iVscale


# empty sns_update

f_args = ['Vflt',
 'Voffs',
 'Vts',
 'fn',
 'foffs',
 'fts',
 'iVscale',
 'ifscale',
 'omega',
 'u']

g_args = ['Vflt', 'a', 'delta', 'ra', 'v', 'xs']

j_args = {'fx': ['fn', 'u'],
 'gx': ['Vflt', 'a', 'delta', 'ra', 'v', 'xs'],
 'gy': ['Vflt', 'a', 'delta', 'ra', 'v', 'xs']}

s_args = OrderedDict([('zs', ['ra', 'xs']),
             ('zs2n', ['ra', 'xs']),
             ('Ec', ['a', 'p', 'q', 'ra', 'v', 'xs']),
             ('E0', ['Ec']),
             ('delta0', ['Ec']),
             ('Vts', []),
             ('fts', []),
             ('ifscale', ['fscale']),
             ('iVscale', ['Vscale']),
             ('foffs', ['fts', 'ifscale']),
             ('Voffs', ['E0', 'Vts', 'iVscale'])])

sns_args = []

ia_args = OrderedDict([('Vflt', ['Voffs', 'Vts', 'iVscale']),
             ('omega', ['foffs', 'fts', 'ifscale']),
             ('delta', ['delta0'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2]),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', [0, 0, 1, 1]),
             ('gyc', []),
             ('gy', [0, 0, 1, 1])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 1]),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', [0, 2, 0, 2]),
             ('gyc', []),
             ('gy', [3, 4, 3, 4])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0]),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', [0, 0, 0, 0]),
             ('gyc', []),
             ('gy', [0, 0, 0, 0])])

j_names = ['fx', 'gx', 'gy']

init_seq = ['Vflt', 'omega', 'delta', 'a', 'v']

need_diag_eps = []
