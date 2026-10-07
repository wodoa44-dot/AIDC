from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "044861cf27813d0e96d3a47d40a523d5"

def f_update(Ipcmd, Iqcmd, S0_y, S1_y, S2_y, v):
    return (-Iqcmd - S1_y, -S2_y + 1.0*v, Ipcmd - S0_y, 0,)


def g_update(Brkpt, HVG_lim_zi, HVG_x, HVG_y, Iolim, Ipcmd, Ipcmd0_LVG, Ipout, Iqcmd, Iqcmd0, Iqout_lim_zi, Iqout_lim_zl, Iqout_x, Iqout_y, Khv, LVG_y, LVPL_y, Lvplsw, Lvpnt0, Lvpnt1, Pe, Qe, S0_y, S1_y, S2_y, Volim, Zerox, a, am, kLVG, kLVPL, v, vd, vq, zam, __zeros, __ones, __falses, __trues):
    return (-LVG_y + select([less_equal(v, Lvpnt0),less_equal(v, Lvpnt1),greater(v, Lvpnt1),__trues], [__zeros,kLVG*(-Lvpnt0 + v),__ones,__zeros], default=nan), -Ipcmd + Ipcmd0_LVG, -Iqcmd + Iqcmd0, -LVPL_y + select([less_equal(S2_y, Zerox),less_equal(S2_y, Brkpt),greater(S2_y, Brkpt),__trues], [9999 - 9999*Lvplsw,-9999*Lvplsw + kLVPL*(S2_y - Zerox) + 9999,9999*__ones,__zeros], default=nan), -Ipout + LVG_y*S0_y, -HVG_x + Khv*(-Volim + v), HVG_lim_zi*HVG_x - HVG_y, -HVG_y - Iqout_x + S1_y, Iolim*Iqout_lim_zl + Iqout_lim_zi*Iqout_x - Iqout_y, Ipout*vd + Iqout_y*vq - Pe, -Ipout*vq + Iqout_y*vd - Qe, v*cos(zam*(a - am)) - vd, -v*sin(zam*(a - am)) - vq, -Pe, -Qe,)


def fx_update():
    return (-1, -1, -1)


def fy_update():
    return (-1, 1.0, 1)


def gy_update(HVG_lim_zi, Ipout, Iqout_lim_zi, Iqout_y, Khv, Lvpnt0, Lvpnt1, S0_y, a, am, kLVG, v, vd, vq, zam, __zeros, __ones, __falses, __trues):
    return (-1, select([greater_equal(Lvpnt0, v),greater_equal(Lvpnt1, v),__trues], [__zeros,kLVG,__zeros], default=nan), -1, -1, -1, S0_y, -1, -1, Khv, HVG_lim_zi, -1, -1, -1, Iqout_lim_zi, -1, vd, vq, -1, Ipout, Iqout_y, -vq, vd, -1, Iqout_y, -Ipout, -1, -v*zam*sin(zam*(a - am)), cos(zam*(a - am)), -1, -v*zam*cos(zam*(a - am)), -sin(zam*(a - am)), -1, -1)


def gx_update(Brkpt, LVG_y, S2_y, Zerox, a, am, kLVPL, v, zam, __zeros, __ones, __falses, __trues):
    return (select([less_equal(S2_y, Zerox),greater_equal(Brkpt, S2_y),__trues], [__zeros,kLVPL,__zeros], default=nan), LVG_y, 1, v*zam*sin(zam*(a - am)), v*zam*cos(zam*(a - am)))


def Iqcmd_ia(Iqcmd0):
    return Iqcmd0


def S1_y_ia(Iqcmd):
    return -Iqcmd


def S2_y_ia(v):
    return 1.0*v


def LVG_y_ia(Lvpnt0, Lvpnt1, kLVG, v, __zeros, __ones, __falses, __trues):
    return select([less_equal(v, Lvpnt0),less_equal(v, Lvpnt1),greater(v, Lvpnt1),__trues], [__zeros,kLVG*(-Lvpnt0 + v),__ones,__zeros], default=nan)


def Ipcmd_ia(Ipcmd0, LVG_y):
    return Ipcmd0*(greater(LVG_y, 0))/LVG_y + (less_equal(LVG_y, 0))


def S0_y_ia(Ipcmd):
    return Ipcmd


def LVPL_y_ia(Brkpt, Lvplsw, S2_y, Zerox, kLVPL, __zeros, __ones, __falses, __trues):
    return select([less_equal(S2_y, Zerox),less_equal(S2_y, Brkpt),greater(S2_y, Brkpt),__trues], [9999 - 9999*Lvplsw,-9999*Lvplsw + kLVPL*(S2_y - Zerox) + 9999,9999*__ones,__zeros], default=nan)


def Ipout_ia(Ipcmd, LVG_y):
    return Ipcmd*LVG_y


def HVG_x_ia(Khv, Volim, v):
    return Khv*(-Volim + v)


def HVG_y_ia(HVG_lim_zi, HVG_x):
    return HVG_lim_zi*HVG_x


def Iqout_x_ia(HVG_y, S1_y):
    return -HVG_y + S1_y


def Iqout_y_ia(Iolim, Iqout_lim_zi, Iqout_lim_zl, Iqout_x):
    return Iolim*Iqout_lim_zl + Iqout_lim_zi*Iqout_x


def Pe_ia(p0):
    return p0


def Qe_ia(q0):
    return q0


def vd_ia(v):
    return v


def vq_ia():
    return 0


def p0_svc(gammap, p0s):
    return gammap*p0s


def q0_svc(gammaq, q0s):
    return gammaq*q0s


def q0gt0_svc(q0):
    return (greater(q0, 0))


def q0lt0_svc(q0):
    return (less(q0, 0))


def Ipcmd0_svc(p0, v):
    return p0/v


def Iqcmd0_svc(q0, v):
    return -q0/v


def Ipcmd0_LVG_svc(Ipcmd):
    return Ipcmd


def kLVG_svc(Lvpnt0, Lvpnt1):
    return (-Lvpnt0 + Lvpnt1)**(-1.0)


def kLVPL_svc(Brkpt, Lvpl1, Lvplsw, Zerox):
    return Lvpl1*Lvplsw/(Brkpt - Zerox)


# empty sns_update

f_args = ['Ipcmd', 'Iqcmd', 'S0_y', 'S1_y', 'S2_y', 'v']

g_args = ['Brkpt',
 'HVG_lim_zi',
 'HVG_x',
 'HVG_y',
 'Iolim',
 'Ipcmd',
 'Ipcmd0_LVG',
 'Ipout',
 'Iqcmd',
 'Iqcmd0',
 'Iqout_lim_zi',
 'Iqout_lim_zl',
 'Iqout_x',
 'Iqout_y',
 'Khv',
 'LVG_y',
 'LVPL_y',
 'Lvplsw',
 'Lvpnt0',
 'Lvpnt1',
 'Pe',
 'Qe',
 'S0_y',
 'S1_y',
 'S2_y',
 'Volim',
 'Zerox',
 'a',
 'am',
 'kLVG',
 'kLVPL',
 'v',
 'vd',
 'vq',
 'zam',
 '__zeros',
 '__ones',
 '__falses',
 '__trues']

j_args = {'fx': [],
 'fy': [],
 'gx': ['Brkpt',
        'LVG_y',
        'S2_y',
        'Zerox',
        'a',
        'am',
        'kLVPL',
        'v',
        'zam',
        '__zeros',
        '__ones',
        '__falses',
        '__trues'],
 'gy': ['HVG_lim_zi',
        'Ipout',
        'Iqout_lim_zi',
        'Iqout_y',
        'Khv',
        'Lvpnt0',
        'Lvpnt1',
        'S0_y',
        'a',
        'am',
        'kLVG',
        'v',
        'vd',
        'vq',
        'zam',
        '__zeros',
        '__ones',
        '__falses',
        '__trues']}

s_args = OrderedDict([('p0', ['gammap', 'p0s']),
             ('q0', ['gammaq', 'q0s']),
             ('q0gt0', ['q0']),
             ('q0lt0', ['q0']),
             ('Ipcmd0', ['p0', 'v']),
             ('Iqcmd0', ['q0', 'v']),
             ('Ipcmd0_LVG', ['Ipcmd']),
             ('kLVG', ['Lvpnt0', 'Lvpnt1']),
             ('kLVPL', ['Brkpt', 'Lvpl1', 'Lvplsw', 'Zerox'])])

sns_args = []

ia_args = OrderedDict([('Iqcmd', ['Iqcmd0']),
             ('S1_y', ['Iqcmd']),
             ('S2_y', ['v']),
             ('LVG_y',
              ['Lvpnt0',
               'Lvpnt1',
               'kLVG',
               'v',
               '__zeros',
               '__ones',
               '__falses',
               '__trues']),
             ('Ipcmd', ['Ipcmd0', 'LVG_y']),
             ('S0_y', ['Ipcmd']),
             ('LVPL_y',
              ['Brkpt',
               'Lvplsw',
               'S2_y',
               'Zerox',
               'kLVPL',
               '__zeros',
               '__ones',
               '__falses',
               '__trues']),
             ('Ipout', ['Ipcmd', 'LVG_y']),
             ('HVG_x', ['Khv', 'Volim', 'v']),
             ('HVG_y', ['HVG_lim_zi', 'HVG_x']),
             ('Iqout_x', ['HVG_y', 'S1_y']),
             ('Iqout_y', ['Iolim', 'Iqout_lim_zi', 'Iqout_lim_zl', 'Iqout_x']),
             ('Pe', ['p0']),
             ('Qe', ['q0']),
             ('vd', ['v']),
             ('vq', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2]),
             ('fyc', []),
             ('fy', [0, 1, 2]),
             ('gxc', []),
             ('gx', [3, 4, 7, 11, 12]),
             ('gyc', [1]),
             ('gy',
              [0,
               0,
               1,
               2,
               3,
               4,
               4,
               5,
               5,
               6,
               6,
               7,
               7,
               8,
               8,
               9,
               9,
               9,
               9,
               9,
               10,
               10,
               10,
               10,
               10,
               11,
               11,
               11,
               12,
               12,
               12,
               13,
               14])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2]),
             ('fyc', []),
             ('fy', [6, 18, 5]),
             ('gxc', []),
             ('gx', [1, 2, 0, 3, 3]),
             ('gyc', [5]),
             ('gy',
              [4,
               18,
               5,
               6,
               7,
               4,
               8,
               9,
               18,
               9,
               10,
               10,
               11,
               11,
               12,
               8,
               12,
               13,
               15,
               16,
               8,
               12,
               14,
               15,
               16,
               15,
               17,
               18,
               16,
               17,
               18,
               13,
               14])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0]),
             ('gyc', [1e-08]),
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
               0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['Iqcmd',
 'S1_y',
 'v',
 'S2_y',
 'LVG_y',
 'Ipcmd',
 'S0_y',
 'am',
 'LVPL_y',
 'Ipout',
 'HVG_x',
 'HVG_y',
 'Iqout_x',
 'Iqout_y',
 'Pe',
 'Qe',
 'vd',
 'vq',
 'a']

need_diag_eps = ['Ipcmd', 'Ipcmd']
