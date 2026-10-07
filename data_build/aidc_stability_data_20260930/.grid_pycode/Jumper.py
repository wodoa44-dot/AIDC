from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "90bab6e4bb67e59ad5a46a2ae5f07f70"

# empty f_update

def g_update(a1, a2, p, q, u, v1, v2):
    return (p*(1 - u) + u*(a1 - a2), p*(1 - u) + u*(v1 - v2), p, -p, q, -q,)


def gy_update(u):
    return (1 - u, u, -u, 1 - u, u, -u, 1, -1, 1, -1)


# empty sns_update

f_args = []

g_args = ['a1', 'a2', 'p', 'q', 'u', 'v1', 'v2']

j_args = {'gy': ['u']}

s_args = OrderedDict()

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
             ('gyc', [0, 1]),
             ('gy', [0, 0, 0, 1, 1, 1, 2, 3, 4, 5])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [0, 1]),
             ('gy', [0, 2, 3, 0, 4, 5, 0, 0, 1, 1])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [1e-08, 1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['gy']

init_seq = ['p', 'q', 'a1', 'a2', 'v1', 'v2']

need_diag_eps = ['p', 'q', 'p', 'q']
