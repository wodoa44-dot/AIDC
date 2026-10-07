from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "1b3cd033271141430044d8357aab02be"

# empty f_update

def g_update(u, v, vcomp, vct):
    return (-u*v - vcomp + vct, 0, 0, 0, 0, 0, vcomp,)


def gy_update(u):
    return (-1, -u, 1)


def vcomp_ia(u, v, vct):
    return -u*v + vct


def Eterm_ia(vcomp):
    return vcomp


def vct_svc(Id, Iq, rc, u, vd, vq, xc):
    return u*abs(vd + 1j*vq + (Id + 1j*Iq)*(rc + 1j*xc))


# empty sns_update

f_args = []

g_args = ['u', 'v', 'vcomp', 'vct']

j_args = {'gy': ['u']}

s_args = OrderedDict([('vct', ['Id', 'Iq', 'rc', 'u', 'vd', 'vq', 'xc'])])

sns_args = []

ia_args = OrderedDict([('vcomp', ['u', 'v', 'vct']), ('Eterm', ['vcomp'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0, 6])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 1, 0])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0, 0])])

j_names = ['gy']

init_seq = ['v', 'vcomp', 'vd', 'vq', 'Id', 'Iq', 'Eterm']

need_diag_eps = []
