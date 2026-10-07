from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "007af25c149c19f99d7049f9e13f7427"

# empty f_update

def g_update(Idc, u, v, voltage):
    return (u*(v - voltage), -Idc,)


def gy_update(u):
    return (u, -1)


def Idc_ia():
    return 0


# empty sns_update

f_args = []

g_args = ['Idc', 'u', 'v', 'voltage']

j_args = {'gy': ['u']}

s_args = OrderedDict()

sns_args = []

ia_args = OrderedDict([('Idc', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [0]),
             ('gy', [0, 1])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [0]),
             ('gy', [1, 0])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', [1e-08]),
             ('gy', [0, 0])])

j_names = ['gy']

init_seq = ['Idc', 'v']

need_diag_eps = ['Idc', 'Idc']
