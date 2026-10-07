from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "49c991729c8f698898fe681b6ef7b971"

def f_update(ll_x, w_dmg):
    return (-ll_x + w_dmg, 0,)


def g_update(T1, T2, dbl, dbu, gain, ll_LT1_z1, ll_LT2_z1, ll_x, ll_y, omega, paux, paux0, plim_zi, plim_zl, plim_zu, pmax, pmin, pnl, pout, pref0, tm0, ue, w_d, w_db_zi, w_db_zlr, w_db_zur, w_dm, w_dmg, wref, wref0):
    return (-paux + paux0, plim_zi*pnl + plim_zl*pmin + plim_zu*pmax - pout, -wref + wref0, ue*(-omega + wref) - w_d, dbl*w_db_zlr + dbu*w_db_zur + w_d*(1 - w_db_zi) - w_dm, gain*w_dm - w_dmg, T1*(-ll_x + w_dmg) + T2*ll_x - T2*ll_y + ll_LT1_z1*ll_LT2_z1*(-ll_x + ll_y), ll_y - pnl + pref0, ue*(pout - tm0),)


def fx_update():
    return (-1,)


def fy_update():
    return (1,)


def gy_update(T1, T2, gain, ll_LT1_z1, ll_LT2_z1, plim_zi, ue, w_db_zi):
    return (-1, -1, plim_zi, -1, ue, -1, 1 - w_db_zi, -1, gain, -1, T1, -T2 + ll_LT1_z1*ll_LT2_z1, 1, -1, ue)


def gx_update(T1, T2, ll_LT1_z1, ll_LT2_z1, ue):
    return (-ue, -T1 + T2 - ll_LT1_z1*ll_LT2_z1)


def w_dmg_ia():
    return 0


def ll_x_ia(w_dmg):
    return w_dmg


def paux_ia(paux0):
    return paux0


def pout_ia(tm0, ue):
    return tm0*ue


def wref_ia(wref0):
    return wref0


def w_d_ia():
    return 0


def w_dm_ia():
    return 0


def ll_y_ia(w_dmg):
    return w_dmg


def pnl_ia(tm0):
    return tm0


def ue_svc(u, ug):
    return u*ug


def pref0_svc(tm0):
    return tm0


def paux0_svc():
    return 0


def gain_svc(R, u):
    return u/R


# empty sns_update

f_args = ['ll_x', 'w_dmg']

g_args = ['T1',
 'T2',
 'dbl',
 'dbu',
 'gain',
 'll_LT1_z1',
 'll_LT2_z1',
 'll_x',
 'll_y',
 'omega',
 'paux',
 'paux0',
 'plim_zi',
 'plim_zl',
 'plim_zu',
 'pmax',
 'pmin',
 'pnl',
 'pout',
 'pref0',
 'tm0',
 'ue',
 'w_d',
 'w_db_zi',
 'w_db_zlr',
 'w_db_zur',
 'w_dm',
 'w_dmg',
 'wref',
 'wref0']

j_args = {'fx': [],
 'fy': [],
 'gx': ['T1', 'T2', 'll_LT1_z1', 'll_LT2_z1', 'ue'],
 'gy': ['T1',
        'T2',
        'gain',
        'll_LT1_z1',
        'll_LT2_z1',
        'plim_zi',
        'ue',
        'w_db_zi']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('pref0', ['tm0']),
             ('paux0', []),
             ('gain', ['R', 'u'])])

sns_args = []

ia_args = OrderedDict([('w_dmg', []),
             ('ll_x', ['w_dmg']),
             ('paux', ['paux0']),
             ('pout', ['tm0', 'ue']),
             ('wref', ['wref0']),
             ('w_d', []),
             ('w_dm', []),
             ('ll_y', ['w_dmg']),
             ('pnl', ['tm0'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [0]),
             ('gxc', []),
             ('gx', [3, 6]),
             ('gyc', [6]),
             ('gy', [0, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [7]),
             ('gxc', []),
             ('gx', [1, 0]),
             ('gyc', [8]),
             ('gy', [2, 3, 9, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 3])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0]),
             ('fyc', []),
             ('fy', [0]),
             ('gxc', []),
             ('gx', [0, 0]),
             ('gyc', [1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['w_dmg',
 'll_x',
 'omega',
 'paux',
 'pout',
 'wref',
 'w_d',
 'w_dm',
 'll_y',
 'pnl',
 'tm']

need_diag_eps = ['ll_y', 'll_y']
