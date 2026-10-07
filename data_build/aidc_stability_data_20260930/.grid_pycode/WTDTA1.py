from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "3eaeb35e4b6ff3c14114017d94246327"

def f_update(DAMP, Kshaft, Pe, Pm, pd, s1_y, s2_y, s3_y, w0):
    return (1.0*Pm/s1_y - 1.0*pd - 1.0*s3_y, -1.0*DAMP*(s2_y - w0) - 1.0*Pe/s2_y + 1.0*pd + 1.0*s3_y, Kshaft*(s1_y - s2_y), 0, 0,)


def g_update(Dshaft, Pe0, Pm, pd, s1_y, s2_y, w0, wr0):
    return (w0 - wr0, Pe0 - Pm, Dshaft*(s1_y - s2_y) - pd, s2_y - 1.0, 0,)


def fx_update(DAMP, Kshaft, Pe, Pm, s1_y, s2_y):
    return (-1.0*Pm/s1_y**2, -1.0, -1.0*DAMP + 1.0*Pe/s2_y**2, 1.0, Kshaft, -Kshaft)


def fy_update(s1_y, s2_y):
    return (1.0/s1_y, -1.0, 1.0, -1.0/s2_y)


def gy_update():
    return (-1, -1, -1)


def gx_update(Dshaft):
    return (Dshaft, -Dshaft, 1)


def wr0_ia(w0):
    return w0


def s1_y_ia(wr0):
    return wr0


def s2_y_ia(wr0):
    return wr0


def s3_y_ia(Pe0, wr0):
    return Pe0/wr0


def Pm_ia(Pe0):
    return Pe0


def pd_ia():
    return 0


def Ht2_svc(H, Htfrac):
    return 2*H*Htfrac


def Hg2_svc(H, Htfrac):
    return 2*H*(1 - Htfrac)


def Kshaft_svc(Freq1, H, Hg2, Ht2):
    return 0.5*Freq1**2*Hg2*Ht2/H


# empty sns_update

f_args = ['DAMP', 'Kshaft', 'Pe', 'Pm', 'pd', 's1_y', 's2_y', 's3_y', 'w0']

g_args = ['Dshaft', 'Pe0', 'Pm', 'pd', 's1_y', 's2_y', 'w0', 'wr0']

j_args = {'fx': ['DAMP', 'Kshaft', 'Pe', 'Pm', 's1_y', 's2_y'],
 'fy': ['s1_y', 's2_y'],
 'gx': ['Dshaft'],
 'gy': []}

s_args = OrderedDict([('Ht2', ['H', 'Htfrac']),
             ('Hg2', ['H', 'Htfrac']),
             ('Kshaft', ['Freq1', 'H', 'Hg2', 'Ht2'])])

sns_args = []

ia_args = OrderedDict([('wr0', ['w0']),
             ('s1_y', ['wr0']),
             ('s2_y', ['wr0']),
             ('s3_y', ['Pe0', 'wr0']),
             ('Pm', ['Pe0']),
             ('pd', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 1, 1, 2, 2]),
             ('fyc', []),
             ('fy', [0, 0, 1, 1]),
             ('gxc', []),
             ('gx', [2, 2, 3]),
             ('gyc', []),
             ('gy', [0, 1, 2])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 2, 1, 2, 0, 1]),
             ('fyc', []),
             ('fy', [6, 7, 7, 9]),
             ('gxc', []),
             ('gx', [0, 1, 1]),
             ('gyc', []),
             ('gy', [5, 6, 7])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0]),
             ('gyc', []),
             ('gy', [0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['wr0', 's1_y', 's2_y', 's3_y', 'wt', 'wg', 'Pm', 'pd', 'wge', 'Pe']

need_diag_eps = []
