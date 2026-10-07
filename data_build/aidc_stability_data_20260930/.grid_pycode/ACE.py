from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "bc6664fbf8717364993e4d8d491fcc32"

# empty f_update

def g_update(ace, bias, fs_v, imva, sys_f):
    return (-ace + 10*bias*imva*sys_f*(fs_v - 1), 0,)


def gy_update():
    return (-1,)


def imva_svc(sys_mva):
    return sys_mva**(-1.0)


# empty sns_update

f_args = []

g_args = ['ace', 'bias', 'fs_v', 'imva', 'sys_f']

j_args = {'gy': []}

s_args = OrderedDict([('imva', ['sys_mva'])])

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
             ('gy', [0])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0])])

j_names = ['gy']

init_seq = ['ace', 'f']

need_diag_eps = []
