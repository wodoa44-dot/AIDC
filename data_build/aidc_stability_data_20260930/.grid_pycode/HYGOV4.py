from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "dc77f46d597bfe4bcef4e91264870a14"

def f_update(GATE_y, Hdam, LAG_y, Psum, SV_y, WO_x, trhead):
    return (SV_y, GATE_y - WO_x, -LAG_y + Psum, Hdam - trhead, 0,)


def g_update(At, Dturb, GATE_y, Hdam, LAG_y, Psum, Rperm, SV_lim_zi, SV_lim_zl, SV_lim_zu, SV_x, SV_y, Tr, TrRtemp, UC, UO, WO_x, WO_y, iTg, omega, paux, paux0, pout, pref, q0, qNL, q_y, tm0, trhead, ue, wd, wref, wref0):
    return (-paux + paux0, -pout + ue*(At*trhead*(-qNL + q_y) - Dturb*GATE_y*wd), -wref + wref0, Hdam**(-0.5)*Rperm*q0 - pref, ue*(omega - wref) - wd, LAG_y*iTg - SV_x, SV_lim_zi*SV_x + SV_lim_zl*UC + SV_lim_zu*UO - SV_y, -Tr*WO_y + TrRtemp*(GATE_y - WO_x), -Psum + ue*(-GATE_y*Rperm - WO_y + paux + pref - wd), -trhead + q_y**2/GATE_y**2, ue*(pout - tm0),)


def fy_update():
    return (1, 1, -1)


def fx_update():
    return (1, -1, -1)


def gy_update(At, Dturb, GATE_y, SV_lim_zi, Tr, qNL, q_y, ue):
    return (-1, -1, -Dturb*GATE_y*ue, At*ue*(-qNL + q_y), -1, -1, -ue, -1, -1, SV_lim_zi, -1, -Tr, ue, ue, -ue, -ue, -1, -1, ue)


def gx_update(At, Dturb, GATE_y, Rperm, TrRtemp, iTg, q_y, trhead, ue, wd):
    return (-Dturb*ue*wd, At*trhead*ue, ue, iTg, TrRtemp, -TrRtemp, -Rperm*ue, -2*q_y**2/GATE_y**3, 2*q_y/GATE_y**2)


def GATE_y_ia(Hdam, q0):
    return Hdam**(-0.5)*q0


def WO_x_ia(GATE_y):
    return GATE_y


def Psum_ia():
    return 0


def LAG_y_ia(Psum):
    return Psum


def q_y_ia(q0):
    return q0


def paux_ia(paux0):
    return paux0


def pout_ia(tm0, ue):
    return tm0*ue


def wref_ia(wref0):
    return wref0


def pref_ia(Hdam, Rperm, q0):
    return Hdam**(-0.5)*Rperm*q0


def wd_ia():
    return 0


def SV_x_ia(LAG_y, iTg):
    return LAG_y*iTg


def SV_y_ia(SV_lim_zi, SV_lim_zl, SV_lim_zu, SV_x, UC, UO):
    return SV_lim_zi*SV_x + SV_lim_zl*UC + SV_lim_zu*UO


def WO_y_ia():
    return 0


def trhead_ia(Hdam):
    return Hdam


def ue_svc(u, ug):
    return u*ug


def pref0_svc(tm0):
    return tm0


def paux0_svc():
    return 0


def iTg_svc(Tg, u):
    return u/Tg


def R_svc(Rperm, Rtemp):
    return Rperm + Rtemp


def TrRtemp_svc(Rtemp, Tr):
    return Rtemp*Tr


def q0_svc(At, Hdam, qNL, tm0):
    return qNL + tm0/(At*Hdam)


# empty sns_update

f_args = ['GATE_y', 'Hdam', 'LAG_y', 'Psum', 'SV_y', 'WO_x', 'trhead']

g_args = ['At',
 'Dturb',
 'GATE_y',
 'Hdam',
 'LAG_y',
 'Psum',
 'Rperm',
 'SV_lim_zi',
 'SV_lim_zl',
 'SV_lim_zu',
 'SV_x',
 'SV_y',
 'Tr',
 'TrRtemp',
 'UC',
 'UO',
 'WO_x',
 'WO_y',
 'iTg',
 'omega',
 'paux',
 'paux0',
 'pout',
 'pref',
 'q0',
 'qNL',
 'q_y',
 'tm0',
 'trhead',
 'ue',
 'wd',
 'wref',
 'wref0']

j_args = {'fx': [],
 'fy': [],
 'gx': ['At',
        'Dturb',
        'GATE_y',
        'Rperm',
        'TrRtemp',
        'iTg',
        'q_y',
        'trhead',
        'ue',
        'wd'],
 'gy': ['At', 'Dturb', 'GATE_y', 'SV_lim_zi', 'Tr', 'qNL', 'q_y', 'ue']}

s_args = OrderedDict([('ue', ['u', 'ug']),
             ('pref0', ['tm0']),
             ('paux0', []),
             ('iTg', ['Tg', 'u']),
             ('R', ['Rperm', 'Rtemp']),
             ('TrRtemp', ['Rtemp', 'Tr']),
             ('q0', ['At', 'Hdam', 'qNL', 'tm0'])])

sns_args = []

ia_args = OrderedDict([('GATE_y', ['Hdam', 'q0']),
             ('WO_x', ['GATE_y']),
             ('Psum', []),
             ('LAG_y', ['Psum']),
             ('q_y', ['q0']),
             ('paux', ['paux0']),
             ('pout', ['tm0', 'ue']),
             ('wref', ['wref0']),
             ('pref', ['Hdam', 'Rperm', 'q0']),
             ('wd', []),
             ('SV_x', ['LAG_y', 'iTg']),
             ('SV_y',
              ['SV_lim_zi', 'SV_lim_zl', 'SV_lim_zu', 'SV_x', 'UC', 'UO']),
             ('WO_y', []),
             ('trhead', ['Hdam'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [1, 1, 2]),
             ('fyc', []),
             ('fy', [0, 2, 3]),
             ('gxc', []),
             ('gx', [1, 1, 4, 5, 7, 7, 8, 9, 9]),
             ('gyc', [7]),
             ('gy',
              [0, 1, 1, 1, 2, 3, 4, 4, 5, 6, 6, 7, 8, 8, 8, 8, 8, 9, 10])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2]),
             ('fyc', []),
             ('fy', [11, 13, 14]),
             ('gxc', []),
             ('gx', [0, 3, 4, 2, 0, 1, 0, 0, 3]),
             ('gyc', [12]),
             ('gy',
              [5,
               6,
               9,
               14,
               7,
               8,
               7,
               9,
               10,
               10,
               11,
               12,
               5,
               8,
               9,
               12,
               13,
               14,
               6])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0, 0, 0]),
             ('gyc', [1e-08]),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fy', 'fx', 'gy', 'gx']

init_seq = ['GATE_y',
 'WO_x',
 'Psum',
 'LAG_y',
 'q_y',
 'omega',
 'paux',
 'pout',
 'wref',
 'pref',
 'wd',
 'SV_x',
 'SV_y',
 'WO_y',
 'trhead',
 'tm']

need_diag_eps = ['WO_y', 'WO_y']
