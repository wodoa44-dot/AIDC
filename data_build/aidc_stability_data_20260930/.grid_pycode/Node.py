from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "e4765b0985ab67766546bb4d820f9e74"

# empty f_update

# empty g_update

def v_ia(flat_start, v0):
    return flat_start + v0*(1 - flat_start)


# empty sns_update

f_args = []

g_args = []

j_args = {}

s_args = OrderedDict()

sns_args = []

ia_args = OrderedDict([('v', ['flat_start', 'v0'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [0]),
             ('gy', [])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [0]),
             ('gy', [])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [1e-08]),
             ('gy', [])])

j_names = []

init_seq = ['v']

need_diag_eps = ['v', 'v']
