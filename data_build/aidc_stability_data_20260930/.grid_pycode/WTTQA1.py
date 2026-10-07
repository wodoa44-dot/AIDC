from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "e882e4cdf8fdccd3652a2fe67a17c2f0"

def f_update(Kip, Pe, Tsel, fPe_y, s1_y, s2_y):
    return (1.0*Pe - s1_y, 1.0*fPe_y - s2_y, Kip*Tsel, 0, 0, 0,)


def g_update(Kpp, PI_hl_zi, PI_hl_zl, PI_hl_zu, PI_xi, PI_y, PI_yul, Pe, Pref0, SWT_s0, SWT_s1, Temax, Temin, Tsel, fPe_y, kp1, kp2, kp3, p1, p2, p3, p4, s1_y, s2_y, sp1, sp2, sp3, sp4, w0, wg, wge, __zeros, __ones, __falses, __trues):
    return (-fPe_y + select([less_equal(s1_y, p1),less_equal(s1_y, p2),less_equal(s1_y, p3),less_equal(s1_y, p4),greater(s1_y, p4),__trues], [sp1,kp1*(-p1 + s1_y) + sp1,kp2*(-p2 + s1_y) + sp2,kp3*(-p3 + s1_y) + sp3,sp4,__zeros], default=nan), SWT_s0*(s2_y - wg) + SWT_s1*(Pe - Pref0)/wg - Tsel, Kpp*Tsel + PI_xi - PI_yul, PI_hl_zi*PI_yul + PI_hl_zl*Temin + PI_hl_zu*Temax - PI_y, 0, fPe_y - w0, 1 - fPe_y, PI_y*wg - Pref0/wge,)


def fx_update():
    return (-1, -1)


def fy_update(Kip):
    return (1.0, 1.0, Kip)


def gx_update(PI_y, Pe, Pref0, SWT_s0, SWT_s1, kp1, kp2, kp3, p1, p2, p3, p4, s1_y, wg, __zeros, __ones, __falses, __trues):
    return (select([greater_equal(p1, s1_y),greater_equal(p2, s1_y),greater_equal(p3, s1_y),greater_equal(p4, s1_y),__trues], [__zeros,kp1,kp2,kp3,__zeros], default=nan), SWT_s0, -SWT_s0 - SWT_s1*(Pe - Pref0)/wg**2, 1, PI_y)


def gy_update(Kpp, PI_hl_zi, Pref0, SWT_s1, wg, wge):
    return (-1, -1, SWT_s1/wg, Kpp, -1, PI_hl_zi, -1, 1, -1, wg, Pref0/wge**2)


def s1_y_ia(Pe):
    return 1.0*Pe


def fPe_y_ia(kp1, kp2, kp3, p1, p2, p3, p4, s1_y, sp1, sp2, sp3, sp4, __zeros, __ones, __falses, __trues):
    return select([less_equal(s1_y, p1),less_equal(s1_y, p2),less_equal(s1_y, p3),less_equal(s1_y, p4),greater(s1_y, p4),__trues], [sp1,kp1*(-p1 + s1_y) + sp1,kp2*(-p2 + s1_y) + sp2,kp3*(-p3 + s1_y) + sp3,sp4,__zeros], default=nan)


def s2_y_ia(fPe_y):
    return 1.0*fPe_y


def PI_xi_ia(Pref0, fPe_y):
    return Pref0/fPe_y


def wg_ia(fPe_y):
    return fPe_y


def wt_ia(fPe_y):
    return fPe_y


def s3_y_ia(Pref0, wg):
    return Pref0/wg


def Tsel_ia(Pe, Pref0, SWT_s0, SWT_s1, s2_y, wg):
    return SWT_s0*(s2_y - wg) + SWT_s1*(Pe - Pref0)/wg


def PI_yul_ia(Kpp, Pref0, Tsel, fPe_y):
    return Kpp*Tsel + Pref0/fPe_y


def PI_y_ia(PI_hl_zi, PI_hl_zl, PI_hl_zu, PI_yul, Temax, Temin):
    return PI_hl_zi*PI_yul + PI_hl_zl*Temin + PI_hl_zu*Temax


def wr0_ia(fPe_y):
    return fPe_y


def wge_ia():
    return 1.00000000000000


def Pref_ia(PI_y, wg):
    return PI_y*wg


def kp1_svc(p1, p2, sp1, sp2):
    return (-sp1 + sp2)/(-p1 + p2)


def kp2_svc(p2, p3, sp2, sp3):
    return (-sp2 + sp3)/(-p2 + p3)


def kp3_svc(p3, p4, sp3, sp4):
    return (-sp3 + sp4)/(-p3 + p4)


# empty sns_update

f_args = ['Kip', 'Pe', 'Tsel', 'fPe_y', 's1_y', 's2_y']

g_args = ['Kpp',
 'PI_hl_zi',
 'PI_hl_zl',
 'PI_hl_zu',
 'PI_xi',
 'PI_y',
 'PI_yul',
 'Pe',
 'Pref0',
 'SWT_s0',
 'SWT_s1',
 'Temax',
 'Temin',
 'Tsel',
 'fPe_y',
 'kp1',
 'kp2',
 'kp3',
 'p1',
 'p2',
 'p3',
 'p4',
 's1_y',
 's2_y',
 'sp1',
 'sp2',
 'sp3',
 'sp4',
 'w0',
 'wg',
 'wge',
 '__zeros',
 '__ones',
 '__falses',
 '__trues']

j_args = {'fx': [],
 'fy': ['Kip'],
 'gx': ['PI_y',
        'Pe',
        'Pref0',
        'SWT_s0',
        'SWT_s1',
        'kp1',
        'kp2',
        'kp3',
        'p1',
        'p2',
        'p3',
        'p4',
        's1_y',
        'wg',
        '__zeros',
        '__ones',
        '__falses',
        '__trues'],
 'gy': ['Kpp', 'PI_hl_zi', 'Pref0', 'SWT_s1', 'wg', 'wge']}

s_args = OrderedDict([('kp1', ['p1', 'p2', 'sp1', 'sp2']),
             ('kp2', ['p2', 'p3', 'sp2', 'sp3']),
             ('kp3', ['p3', 'p4', 'sp3', 'sp4'])])

sns_args = []

ia_args = OrderedDict([('s1_y', ['Pe']),
             ('fPe_y',
              ['kp1',
               'kp2',
               'kp3',
               'p1',
               'p2',
               'p3',
               'p4',
               's1_y',
               'sp1',
               'sp2',
               'sp3',
               'sp4',
               '__zeros',
               '__ones',
               '__falses',
               '__trues']),
             ('s2_y', ['fPe_y']),
             ('PI_xi', ['Pref0', 'fPe_y']),
             ('wg', ['fPe_y']),
             ('wt', ['fPe_y']),
             ('s3_y', ['Pref0', 'wg']),
             ('Tsel', ['Pe', 'Pref0', 'SWT_s0', 'SWT_s1', 's2_y', 'wg']),
             ('PI_yul', ['Kpp', 'Pref0', 'Tsel', 'fPe_y']),
             ('PI_y',
              ['PI_hl_zi', 'PI_hl_zl', 'PI_hl_zu', 'PI_yul', 'Temax', 'Temin']),
             ('wr0', ['fPe_y']),
             ('wge', []),
             ('Pref', ['PI_y', 'wg'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1]),
             ('fyc', []),
             ('fy', [0, 1, 2]),
             ('gxc', []),
             ('gx', [0, 1, 1, 2, 7]),
             ('gyc', []),
             ('gy', [0, 1, 1, 2, 2, 3, 3, 5, 6, 7, 7])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1]),
             ('fyc', []),
             ('fy', [10, 6, 7]),
             ('gxc', []),
             ('gx', [0, 1, 3, 2, 3]),
             ('gyc', []),
             ('gy', [6, 7, 10, 7, 8, 8, 9, 6, 6, 9, 12])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0]),
             ('gyc', []),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gx', 'gy']

init_seq = ['Pe',
 's1_y',
 'fPe_y',
 's2_y',
 'PI_xi',
 'wg',
 'wt',
 's3_y',
 'Tsel',
 'PI_yul',
 'PI_y',
 'wr0',
 'wge',
 'Pref']

need_diag_eps = []
