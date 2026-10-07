from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "0bebd4627b549ac862e88cdbb2dd187e"

# empty f_update

def g_update(a, aa, ab, ac, ap, b, bhk, d120, g, ghk, u, v, va, vb, vc, vnd, vnq, vp, vzd, vzq):
    return (-vp + (1/3)*sqrt((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2), aa - ap + arctan2(vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120), va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120)), va*cos(aa) + vb*cos(ab - d120) + vc*cos(ac + d120) - vnd, va*sin(aa) + vb*sin(ab - d120) + vc*sin(ac + d120) - vnq, va*cos(aa) + vb*cos(ab) + vc*cos(ac) - vzd, va*sin(aa) + vb*sin(ab) + vc*sin(ac) - vzq, u*(v**2*(g + ghk) - v*vp*(bhk*sin(a - ap) + ghk*cos(a - ap))), (1/3)*u*(-v*va*(-bhk*sin(a - aa) + ghk*cos(a - aa)) + va**2*(g + ghk)), (1/3)*u*(-v*vb*(bhk*sin(-a + ab + d120) + ghk*cos(-a + ab + d120)) + vb**2*(g + ghk)), (1/3)*u*(-v*vc*(-bhk*sin(a - ac + d120) + ghk*cos(a - ac + d120)) + vc**2*(g + ghk)), u*(-v**2*(b + bhk) - v*vp*(-bhk*cos(a - ap) + ghk*sin(a - ap))), (1/3)*u*(v*va*(bhk*cos(a - aa) + ghk*sin(a - aa)) - va**2*(b + bhk)), (1/3)*u*(v*vb*(bhk*cos(-a + ab + d120) - ghk*sin(-a + ab + d120)) - vb**2*(b + bhk)), (1/3)*u*(v*vc*(bhk*cos(a - ac + d120) + ghk*sin(a - ac + d120)) - vc**2*(b + bhk)),)


def gy_update(a, aa, ab, ac, ap, b, bhk, d120, g, ghk, u, v, va, vb, vc, vp):
    return (-1, (1/3)*((1/2)*(vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))*(-2*vb*cos(-aa + ab + d120) - 2*vc*cos(aa - ac + d120)) + (1/2)*(2*vb*sin(-aa + ab + d120) - 2*vc*sin(aa - ac + d120))*(va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120)))/sqrt((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2), (1/3)*(vb*(vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))*cos(-aa + ab + d120) - vb*(va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))*sin(-aa + ab + d120))/sqrt((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2), (1/3)*(vc*(vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))*cos(aa - ac + d120) + vc*(va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))*sin(aa - ac + d120))/sqrt((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2), (1/3)*(va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))/sqrt((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2), (1/3)*((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))*sin(-aa + ab + d120) + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))*cos(-aa + ab + d120))/sqrt((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2), (1/3)*(-(vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))*sin(aa - ac + d120) + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))*cos(aa - ac + d120))/sqrt((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2), -1, (-vb*sin(-aa + ab + d120) + vc*sin(aa - ac + d120))*(vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))/((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2) + (-vb*cos(-aa + ab + d120) - vc*cos(aa - ac + d120))*(va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))/((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2) + 1, -vb*(-vb*sin(-aa + ab + d120) + vc*sin(aa - ac + d120))*sin(-aa + ab + d120)/((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2) + vb*(va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))*cos(-aa + ab + d120)/((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2), vc*(-vb*sin(-aa + ab + d120) + vc*sin(aa - ac + d120))*sin(aa - ac + d120)/((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2) + vc*(va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))*cos(aa - ac + d120)/((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2), (-vb*sin(-aa + ab + d120) + vc*sin(aa - ac + d120))/((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2), (-vb*sin(-aa + ab + d120) + vc*sin(aa - ac + d120))*cos(-aa + ab + d120)/((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2) + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))*sin(-aa + ab + d120)/((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2), (-vb*sin(-aa + ab + d120) + vc*sin(aa - ac + d120))*cos(aa - ac + d120)/((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2) - (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))*sin(aa - ac + d120)/((vb*sin(-aa + ab + d120) - vc*sin(aa - ac + d120))**2 + (va + vb*cos(-aa + ab + d120) + vc*cos(aa - ac + d120))**2), -1, -va*sin(aa), -vb*sin(ab - d120), -vc*sin(ac + d120), cos(aa), cos(ab - d120), cos(ac + d120), -1, va*cos(aa), vb*cos(ab - d120), vc*cos(ac + d120), sin(aa), sin(ab - d120), sin(ac + d120), -1, -va*sin(aa), -vb*sin(ab), -vc*sin(ac), cos(aa), cos(ab), cos(ac), -1, va*cos(aa), vb*cos(ab), vc*cos(ac), sin(aa), sin(ab), sin(ac), -u*v*(bhk*sin(a - ap) + ghk*cos(a - ap)), -u*v*vp*(-bhk*cos(a - ap) + ghk*sin(a - ap)), -u*v*vp*(bhk*cos(a - ap) - ghk*sin(a - ap)), u*(2*v*(g + ghk) - vp*(bhk*sin(a - ap) + ghk*cos(a - ap))), -1/3*u*v*va*(-bhk*cos(a - aa) - ghk*sin(a - aa)), -1/3*u*v*va*(bhk*cos(a - aa) + ghk*sin(a - aa)), -1/3*u*va*(-bhk*sin(a - aa) + ghk*cos(a - aa)), (1/3)*u*(-v*(-bhk*sin(a - aa) + ghk*cos(a - aa)) + 2*va*(g + ghk)), -1/3*u*v*vb*(-bhk*cos(-a + ab + d120) + ghk*sin(-a + ab + d120)), -1/3*u*v*vb*(bhk*cos(-a + ab + d120) - ghk*sin(-a + ab + d120)), -1/3*u*vb*(bhk*sin(-a + ab + d120) + ghk*cos(-a + ab + d120)), (1/3)*u*(-v*(bhk*sin(-a + ab + d120) + ghk*cos(-a + ab + d120)) + 2*vb*(g + ghk)), -1/3*u*v*vc*(-bhk*cos(a - ac + d120) - ghk*sin(a - ac + d120)), -1/3*u*v*vc*(bhk*cos(a - ac + d120) + ghk*sin(a - ac + d120)), -1/3*u*vc*(-bhk*sin(a - ac + d120) + ghk*cos(a - ac + d120)), (1/3)*u*(-v*(-bhk*sin(a - ac + d120) + ghk*cos(a - ac + d120)) + 2*vc*(g + ghk)), -u*v*(-bhk*cos(a - ap) + ghk*sin(a - ap)), -u*v*vp*(-bhk*sin(a - ap) - ghk*cos(a - ap)), -u*v*vp*(bhk*sin(a - ap) + ghk*cos(a - ap)), u*(-2*v*(b + bhk) - vp*(-bhk*cos(a - ap) + ghk*sin(a - ap))), (1/3)*u*v*va*(-bhk*sin(a - aa) + ghk*cos(a - aa)), (1/3)*u*v*va*(bhk*sin(a - aa) - ghk*cos(a - aa)), (1/3)*u*va*(bhk*cos(a - aa) + ghk*sin(a - aa)), (1/3)*u*(v*(bhk*cos(a - aa) + ghk*sin(a - aa)) - 2*va*(b + bhk)), (1/3)*u*v*vb*(bhk*sin(-a + ab + d120) + ghk*cos(-a + ab + d120)), (1/3)*u*v*vb*(-bhk*sin(-a + ab + d120) - ghk*cos(-a + ab + d120)), (1/3)*u*vb*(bhk*cos(-a + ab + d120) - ghk*sin(-a + ab + d120)), (1/3)*u*(v*(bhk*cos(-a + ab + d120) - ghk*sin(-a + ab + d120)) - 2*vb*(b + bhk)), (1/3)*u*v*vc*(-bhk*sin(a - ac + d120) + ghk*cos(a - ac + d120)), (1/3)*u*v*vc*(bhk*sin(a - ac + d120) - ghk*cos(a - ac + d120)), (1/3)*u*vc*(bhk*cos(a - ac + d120) + ghk*sin(a - ac + d120)), (1/3)*u*(v*(bhk*cos(a - ac + d120) + ghk*sin(a - ac + d120)) - 2*vc*(b + bhk)))


def vp_ia(va, vb, vc):
    return (1/3)*va + (1/3)*vb + (1/3)*vc


def ap_ia(aa, ab, ac):
    return aa + ab + ac


def vnd_ia():
    return 0.0


def vnq_ia():
    return 0.0


def vzd_ia():
    return 0.0


def vzq_ia():
    return 0.0


def yhk_svc(r, u, x):
    return u/(r + 1j*x)


def ghk_svc(yhk):
    return real(yhk)


def bhk_svc(yhk):
    return imag(yhk)


def d120_svc():
    return (2/3)*pi


# empty sns_update

f_args = []

g_args = ['a',
 'aa',
 'ab',
 'ac',
 'ap',
 'b',
 'bhk',
 'd120',
 'g',
 'ghk',
 'u',
 'v',
 'va',
 'vb',
 'vc',
 'vnd',
 'vnq',
 'vp',
 'vzd',
 'vzq']

j_args = {'gy': ['a',
        'aa',
        'ab',
        'ac',
        'ap',
        'b',
        'bhk',
        'd120',
        'g',
        'ghk',
        'u',
        'v',
        'va',
        'vb',
        'vc',
        'vp']}

s_args = OrderedDict([('yhk', ['r', 'u', 'x']),
             ('ghk', ['yhk']),
             ('bhk', ['yhk']),
             ('d120', [])])

sns_args = []

ia_args = OrderedDict([('vp', ['va', 'vb', 'vc']),
             ('ap', ['aa', 'ab', 'ac']),
             ('vnd', []),
             ('vnq', []),
             ('vzd', []),
             ('vzq', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [7, 8, 9, 11, 12, 13]),
             ('gy',
              [0,
               0,
               0,
               0,
               0,
               0,
               0,
               1,
               1,
               1,
               1,
               1,
               1,
               1,
               2,
               2,
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
               3,
               3,
               4,
               4,
               4,
               4,
               4,
               4,
               4,
               5,
               5,
               5,
               5,
               5,
               5,
               5,
               6,
               6,
               6,
               6,
               7,
               7,
               7,
               7,
               8,
               8,
               8,
               8,
               9,
               9,
               9,
               9,
               10,
               10,
               10,
               10,
               11,
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
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [7, 8, 9, 11, 12, 13]),
             ('gy',
              [0,
               7,
               8,
               9,
               11,
               12,
               13,
               1,
               7,
               8,
               9,
               11,
               12,
               13,
               2,
               7,
               8,
               9,
               11,
               12,
               13,
               3,
               7,
               8,
               9,
               11,
               12,
               13,
               4,
               7,
               8,
               9,
               11,
               12,
               13,
               5,
               7,
               8,
               9,
               11,
               12,
               13,
               0,
               1,
               6,
               10,
               6,
               7,
               10,
               11,
               6,
               8,
               10,
               12,
               6,
               9,
               10,
               13,
               0,
               1,
               6,
               10,
               6,
               7,
               10,
               11,
               6,
               8,
               10,
               12,
               6,
               9,
               10,
               13])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08]),
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

j_names = ['gy']

init_seq = ['va',
 'vb',
 'vc',
 'vp',
 'aa',
 'ab',
 'ac',
 'ap',
 'vnd',
 'vnq',
 'vzd',
 'vzq',
 'a',
 'v']

need_diag_eps = ['aa', 'ab', 'ac', 'va', 'vb', 'vc', 'aa', 'ab', 'ac', 'va', 'vb', 'vc']
