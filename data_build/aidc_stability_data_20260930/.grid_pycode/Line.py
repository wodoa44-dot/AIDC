from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "a53a0ac15d0e504be61ff250a222a654"

# empty f_update

def g_update(a1, a2, bh, bhk, gh, ghk, itap, itap2, phi, u, v1, v2):
    return (u*(-itap*v1*v2*(-bhk*sin(-a1 + a2 + phi) + ghk*cos(-a1 + a2 + phi)) + itap2*v1**2*(gh + ghk)), u*(-itap*v1*v2*(bhk*sin(-a1 + a2 + phi) + ghk*cos(-a1 + a2 + phi)) + v2**2*(gh + ghk)), u*(-itap*v1*v2*(-bhk*cos(-a1 + a2 + phi) - ghk*sin(-a1 + a2 + phi)) - itap2*v1**2*(bh + bhk)), u*(itap*v1*v2*(bhk*cos(-a1 + a2 + phi) - ghk*sin(-a1 + a2 + phi)) - v2**2*(bh + bhk)),)


def gy_update(a1, a2, bh, bhk, gh, ghk, itap, itap2, phi, u, v1, v2):
    return (-itap*u*v1*v2*(bhk*cos(-a1 + a2 + phi) + ghk*sin(-a1 + a2 + phi)), -itap*u*v1*v2*(-bhk*cos(-a1 + a2 + phi) - ghk*sin(-a1 + a2 + phi)), u*(-itap*v2*(-bhk*sin(-a1 + a2 + phi) + ghk*cos(-a1 + a2 + phi)) + 2*itap2*v1*(gh + ghk)), -itap*u*v1*(-bhk*sin(-a1 + a2 + phi) + ghk*cos(-a1 + a2 + phi)), -itap*u*v1*v2*(-bhk*cos(-a1 + a2 + phi) + ghk*sin(-a1 + a2 + phi)), -itap*u*v1*v2*(bhk*cos(-a1 + a2 + phi) - ghk*sin(-a1 + a2 + phi)), -itap*u*v2*(bhk*sin(-a1 + a2 + phi) + ghk*cos(-a1 + a2 + phi)), u*(-itap*v1*(bhk*sin(-a1 + a2 + phi) + ghk*cos(-a1 + a2 + phi)) + 2*v2*(gh + ghk)), -itap*u*v1*v2*(-bhk*sin(-a1 + a2 + phi) + ghk*cos(-a1 + a2 + phi)), -itap*u*v1*v2*(bhk*sin(-a1 + a2 + phi) - ghk*cos(-a1 + a2 + phi)), u*(-itap*v2*(-bhk*cos(-a1 + a2 + phi) - ghk*sin(-a1 + a2 + phi)) - 2*itap2*v1*(bh + bhk)), -itap*u*v1*(-bhk*cos(-a1 + a2 + phi) - ghk*sin(-a1 + a2 + phi)), itap*u*v1*v2*(bhk*sin(-a1 + a2 + phi) + ghk*cos(-a1 + a2 + phi)), itap*u*v1*v2*(-bhk*sin(-a1 + a2 + phi) - ghk*cos(-a1 + a2 + phi)), itap*u*v2*(bhk*cos(-a1 + a2 + phi) - ghk*sin(-a1 + a2 + phi)), u*(itap*v1*(bhk*cos(-a1 + a2 + phi) - ghk*sin(-a1 + a2 + phi)) - 2*v2*(bh + bhk)))


def gh_svc(g, g1):
    return 0.5*g + g1


def bh_svc(b, b1):
    return 0.5*b + b1


def gk_svc(g, g2):
    return 0.5*g + g2


def bk_svc(b, b2):
    return 0.5*b + b2


def yh_svc(bh, gh, u):
    return u*(1j*bh + gh)


def yk_svc(bk, gk, u):
    return u*(1j*bk + gk)


def yhk_svc(r, u, x):
    return u/(r + 1j*(x + 1.0e-8) + 1.0e-8)


def ghk_svc(yhk):
    return real(yhk)


def bhk_svc(yhk):
    return imag(yhk)


def itap_svc(tap):
    return tap**(-1.0)


def itap2_svc(tap):
    return tap**(-2.0)


# empty sns_update

f_args = []

g_args = ['a1', 'a2', 'bh', 'bhk', 'gh', 'ghk', 'itap', 'itap2', 'phi', 'u', 'v1', 'v2']

j_args = {'gy': ['a1',
        'a2',
        'bh',
        'bhk',
        'gh',
        'ghk',
        'itap',
        'itap2',
        'phi',
        'u',
        'v1',
        'v2']}

s_args = OrderedDict([('gh', ['g', 'g1']),
             ('bh', ['b', 'b1']),
             ('gk', ['g', 'g2']),
             ('bk', ['b', 'b2']),
             ('yh', ['bh', 'gh', 'u']),
             ('yk', ['bk', 'gk', 'u']),
             ('yhk', ['r', 'u', 'x']),
             ('ghk', ['yhk']),
             ('bhk', ['yhk']),
             ('itap', ['tap']),
             ('itap2', ['tap'])])

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
             ('gy', [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['gy']

init_seq = ['a1', 'a2', 'v1', 'v2']

need_diag_eps = []
