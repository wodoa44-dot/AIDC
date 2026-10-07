from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "ed0e06b4fea4fe8b9367f425e0878d26"

def f_update(LAG_y, LG_y, dg, pd, q_y):
    return (-LG_y + pd, LG_y, -LAG_y + dg, 1 - q_y**2/LAG_y**2, 0,)


def g_update(At, DB_db_zl, DB_db_zu, DB_y, Dt, LAG_y, LG_y, R, dbL, dbU, dg, gr, gtpos, h, omega, paux, paux0, pd, pout, pref, q0, qNL, q_y, tm0, ue, wd, wref, wref0):
    return (-paux + paux0, -pout + ue*(At*h*(-qNL + q_y) - Dt*LAG_y*wd), -wref + wref0, R*q0 - pref, ue*(omega - wref) - wd, -pd + ue*(-DB_y - R*dg + paux + pref), LG_y*gr - dg + gtpos, -h + q_y**2/LAG_y**2, 1.0*DB_db_zl*(-dbL + wd) + 1.0*DB_db_zu*(-dbU + wd) - DB_y, ue*(pout - tm0),)


def fx_update(LAG_y, q_y):
    return (-1, 1, -1, 2*q_y**2/LAG_y**3, -2*q_y/LAG_y**2)


def fy_update():
    return (1, 1)


def gy_update(At, DB_db_zl, DB_db_zu, Dt, LAG_y, R, qNL, q_y, ue):
    return (-1, -1, -Dt*LAG_y*ue, At*ue*(-qNL + q_y), -1, -1, -ue, -1, ue, ue, -1, -R*ue, -ue, -1, -1, 1.0*DB_db_zl + 1.0*DB_db_zu, -1, ue)


def gx_update(At, Dt, LAG_y, gr, h, q_y, ue, wd):
    return (-Dt*ue*wd, At*h*ue, ue, gr, 1, -2*q_y**2/LAG_y**3, 2*q_y/LAG_y**2)


def pd_ia():
    return 0


def LG_y_ia(pd):
    return pd


def gtpos_ia(q0):
    return q0


def dg_ia(q0):
    return q0


def LAG_y_ia(dg):
    return dg


def q_y_ia(q0):
    return q0


def paux_ia(paux0):
    return paux0


def pout_ia(tm0, ue):
    return tm0*ue


def wref_ia(wref0):
    return wref0


def pref_ia(R, q0):
    return R*q0


def wd_ia():
    return 0


def h_ia():
    return 1


def DB_y_ia(DB_db_zl, DB_db_zu, dbL, dbU, wd):
    return 1.0*DB_db_zl*(-dbL + wd) + 1.0*DB_db_zu*(-dbU + wd)


def ue_svc(u, ug):
    return u*ug


def pref0_svc(tm0):
    return tm0


def paux0_svc():
    return 0


def VELMn_svc(VELM):
    return -VELM


def tr_svc(Tr, r):
    return Tr*r


def gr_svc(r):
    return r**(-1.0)


def ratel_svc(VELM, gr):
    return -VELM - gr


def rateu_svc(VELM, gr):
    return VELM - gr


def q0_svc(At, qNL, tm0):
    return qNL + tm0/At


def dgl_svc(LG_y, VELM, gr):
    return -LG_y*gr - VELM


def dgu_svc(LG_y, VELM, gr):
    return -LG_y*gr + VELM


# empty sns_update

f_args = ['LAG_y', 'LG_y', 'dg', 'pd', 'q_y']

g_args = ['At',
 'DB_db_zl',
 'DB_db_zu',
 'DB_y',
 'Dt',
 'LAG_y',
 'LG_y',
 'R',
 'dbL',
 'dbU',
 'dg',
 'gr',
 'gtpos',
 'h',
 'omega',
 'paux',
 'paux0',
 'pd',
 'pout',
 'pref',
 'q0',
 'qNL',
 'q_y',
 'tm0',
 'ue',
 'wd',
 'wref',
 'wref0']

j_args = {'fx': ['LAG_y', 'q_y'],
 'fy': [],
 'gx': ['At', 'Dt', 'LAG_y', 'gr', 'h', 'q_y', 'ue', 'wd'],
 'gy': ['At', 'DB_db_zl', 'DB_db_zu', 'Dt', 'LAG_y', 'R', 'qNL', 'q_y', 'ue']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('pref0', ['tm0']),
             ('paux0', []),
             ('VELMn', ['VELM']),
             ('tr', ['Tr', 'r']),
             ('gr', ['r']),
             ('ratel', ['VELM', 'gr']),
             ('rateu', ['VELM', 'gr']),
             ('q0', ['At', 'qNL', 'tm0']),
             ('dgl', ['LG_y', 'VELM', 'gr']),
             ('dgu', ['LG_y', 'VELM', 'gr'])])

sns_args = []

ia_args = OrderedDict([('pd', []),
             ('LG_y', ['pd']),
             ('gtpos', ['q0']),
             ('dg', ['q0']),
             ('LAG_y', ['dg']),
             ('q_y', ['q0']),
             ('paux', ['paux0']),
             ('pout', ['tm0', 'ue']),
             ('wref', ['wref0']),
             ('pref', ['R', 'q0']),
             ('wd', []),
             ('h', []),
             ('DB_y', ['DB_db_zl', 'DB_db_zu', 'dbL', 'dbU', 'wd'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2, 3, 3]),
             ('fyc', []),
             ('fy', [0, 2]),
             ('gxc', []),
             ('gx', [1, 1, 4, 6, 6, 7, 7]),
             ('gyc', []),
             ('gy', [0, 1, 1, 1, 2, 3, 4, 4, 5, 5, 5, 5, 5, 6, 7, 8, 8, 9])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 2, 2, 3]),
             ('fyc', []),
             ('fy', [10, 11]),
             ('gxc', []),
             ('gx', [2, 3, 4, 0, 1, 2, 3]),
             ('gyc', []),
             ('gy',
              [5, 6, 9, 12, 7, 8, 7, 9, 5, 8, 10, 11, 13, 11, 12, 9, 13, 6])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0]),
             ('gyc', []),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['pd',
 'LG_y',
 'gtpos',
 'dg',
 'LAG_y',
 'q_y',
 'omega',
 'paux',
 'pout',
 'wref',
 'pref',
 'wd',
 'h',
 'DB_y',
 'tm']

need_diag_eps = []
