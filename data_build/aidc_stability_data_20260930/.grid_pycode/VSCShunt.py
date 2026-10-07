from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "c0250a3b1650bcb4874016e01abad965"

# empty f_update

def g_update(a, ash, bsh, gsh, mode_s0, mode_s1, mode_s2, mode_s3, p0, pdc, psh, q0, qsh, u, v, v0, v1, v2, vdc0, vsh):
    return (-psh + u*(-bsh*v*vsh*sin(a - ash) + gsh*v**2 - gsh*v*vsh*cos(a - ash)), -qsh + u*(-bsh*v**2 + bsh*v*vsh*cos(a - ash) - gsh*v*vsh*sin(a - ash)), u*(mode_s0 + mode_s1)*(p0 - psh) + u*(mode_s2 + mode_s3)*(v1 - v2 - vdc0), u*(mode_s0 + mode_s2)*(q0 - qsh) + u*(mode_s1 + mode_s3)*(-v + v0), pdc + u*(bsh*v*vsh*sin(a - ash) - gsh*v*vsh*cos(a - ash) + gsh*vsh**2), -psh, -qsh, -pdc/(v1 - v2), pdc/(v1 - v2),)


def gy_update(a, ash, bsh, gsh, mode_s0, mode_s1, mode_s2, mode_s3, pdc, u, v, v1, v2, vsh):
    return (u*(bsh*v*vsh*cos(a - ash) - gsh*v*vsh*sin(a - ash)), u*(-bsh*v*sin(a - ash) - gsh*v*cos(a - ash)), -1, u*(-bsh*v*vsh*cos(a - ash) + gsh*v*vsh*sin(a - ash)), u*(-bsh*vsh*sin(a - ash) + 2*gsh*v - gsh*vsh*cos(a - ash)), u*(bsh*v*vsh*sin(a - ash) + gsh*v*vsh*cos(a - ash)), u*(bsh*v*cos(a - ash) - gsh*v*sin(a - ash)), -1, u*(-bsh*v*vsh*sin(a - ash) - gsh*v*vsh*cos(a - ash)), u*(-2*bsh*v + bsh*vsh*cos(a - ash) - gsh*vsh*sin(a - ash)), -u*(mode_s0 + mode_s1), u*(mode_s2 + mode_s3), -u*(mode_s2 + mode_s3), -u*(mode_s0 + mode_s2), -u*(mode_s1 + mode_s3), u*(-bsh*v*vsh*cos(a - ash) - gsh*v*vsh*sin(a - ash)), u*(bsh*v*sin(a - ash) - gsh*v*cos(a - ash) + 2*gsh*vsh), 1, u*(bsh*v*vsh*cos(a - ash) + gsh*v*vsh*sin(a - ash)), u*(bsh*vsh*sin(a - ash) - gsh*vsh*cos(a - ash)), -1, -1, -1/(v1 - v2), pdc/(v1 - v2)**2, -pdc/(v1 - v2)**2, (v1 - v2)**(-1.0), -pdc/(v1 - v2)**2, pdc/(v1 - v2)**2)


def ash_ia(a):
    return a


def vsh_ia(v0):
    return v0


def psh_ia(mode_s0, mode_s1, p0):
    return p0*(mode_s0 + mode_s1)


def qsh_ia(mode_s0, mode_s2, q0):
    return q0*(mode_s0 + mode_s2)


def pdc_ia():
    return 0


def gsh_svc(rsh, xsh):
    return (real(rsh) - imag(xsh))/((real(rsh) - imag(xsh))**2 + (real(xsh) + imag(rsh))**2)


def bsh_svc(rsh, xsh):
    return (-real(xsh) - imag(rsh))/((real(rsh) - imag(xsh))**2 + (real(xsh) + imag(rsh))**2)


# empty sns_update

f_args = []

g_args = ['a',
 'ash',
 'bsh',
 'gsh',
 'mode_s0',
 'mode_s1',
 'mode_s2',
 'mode_s3',
 'p0',
 'pdc',
 'psh',
 'q0',
 'qsh',
 'u',
 'v',
 'v0',
 'v1',
 'v2',
 'vdc0',
 'vsh']

j_args = {'gy': ['a',
        'ash',
        'bsh',
        'gsh',
        'mode_s0',
        'mode_s1',
        'mode_s2',
        'mode_s3',
        'pdc',
        'u',
        'v',
        'v1',
        'v2',
        'vsh']}

s_args = OrderedDict([('gsh', ['rsh', 'xsh']), ('bsh', ['rsh', 'xsh'])])

sns_args = []

ia_args = OrderedDict([('ash', ['a']),
             ('vsh', ['v0']),
             ('psh', ['mode_s0', 'mode_s1', 'p0']),
             ('qsh', ['mode_s0', 'mode_s2', 'q0']),
             ('pdc', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [0, 1, 2, 3]),
             ('gy',
              [0,
               0,
               0,
               0,
               0,
               1,
               1,
               1,
               1,
               1,
               2,
               2,
               2,
               3,
               3,
               4,
               4,
               4,
               4,
               4,
               5,
               6,
               7,
               7,
               7,
               8,
               8,
               8])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [0, 1, 2, 3]),
             ('gy',
              [0,
               1,
               2,
               5,
               6,
               0,
               1,
               3,
               5,
               6,
               2,
               7,
               8,
               3,
               6,
               0,
               1,
               4,
               5,
               6,
               2,
               3,
               4,
               7,
               8,
               4,
               7,
               8])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [1e-08, 1e-08, 1e-08, 1e-08]),
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

j_names = ['gy']

init_seq = ['a', 'ash', 'vsh', 'psh', 'qsh', 'pdc', 'v', 'v1', 'v2']

need_diag_eps = ['ash', 'vsh', 'psh', 'qsh', 'ash', 'vsh', 'psh', 'qsh']
