from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "c2922960d5cfea18119270f9f822c105"

def f_update(Kcc, Kic, Kiw, LG_y, PIc_y, PIw_y, Pord, Pref, wref, wt):
    return (Kic*(Pord - Pref), Kiw*(Kcc*(Pord - Pref) - wref + wt), -LG_y + 1.0*PIc_y + 1.0*PIw_y, 0, 0,)


def g_update(Kcc, Kpc, Kpw, LG_y, PIc_hl_zi, PIc_hl_zl, PIc_hl_zu, PIc_xi, PIc_y, PIc_yul, PIw_hl_zi, PIw_hl_zl, PIw_hl_zu, PIw_xi, PIw_y, PIw_yul, Pord, Pref, theta0, thmax, thmin, wref, wref0, wt):
    return (Kpc*(Pord - Pref) + PIc_xi - PIc_yul, PIc_hl_zi*PIc_yul + PIc_hl_zl*thmin + PIc_hl_zu*thmax - PIc_y, -wref + wref0, Kpw*(Kcc*(Pord - Pref) - wref + wt) + PIw_xi - PIw_yul, PIw_hl_zi*PIw_yul + PIw_hl_zl*thmin + PIw_hl_zu*thmax - PIw_y, LG_y - theta0, 0,)


def fx_update(Kcc, Kic, Kiw):
    return (Kic, Kiw, Kcc*Kiw, -1)


def fy_update(Kcc, Kic, Kiw):
    return (-Kic, -Kiw, -Kcc*Kiw, 1.0, 1.0)


def gx_update(Kcc, Kpc, Kpw):
    return (1, Kpc, 1, Kpw, Kcc*Kpw, 1)


def gy_update(Kcc, Kpc, Kpw, PIc_hl_zi, PIw_hl_zi):
    return (-1, -Kpc, PIc_hl_zi, -1, -1, -Kpw, -1, -Kcc*Kpw, PIw_hl_zi, -1)


def PIc_xi_ia():
    return 0.0


def PIw_xi_ia():
    return 0.0


def PIc_yul_ia(Kpc, Pord, Pref):
    return Kpc*(Pord - Pref)


def PIc_y_ia(PIc_hl_zi, PIc_hl_zl, PIc_hl_zu, PIc_yul, thmax, thmin):
    return PIc_hl_zi*PIc_yul + PIc_hl_zl*thmin + PIc_hl_zu*thmax


def wref_ia(wt):
    return wt


def PIw_yul_ia(Kcc, Kpw, Pord, Pref, wref, wt):
    return Kpw*(Kcc*(Pord - Pref) - wref + wt)


def PIw_y_ia(PIw_hl_zi, PIw_hl_zl, PIw_hl_zu, PIw_yul, thmax, thmin):
    return PIw_hl_zi*PIw_yul + PIw_hl_zl*thmin + PIw_hl_zu*thmax


def LG_y_ia(PIc_y, PIw_y):
    return 1.0*PIc_y + 1.0*PIw_y


def wref0_svc(wref):
    return wref


# empty sns_update

f_args = ['Kcc', 'Kic', 'Kiw', 'LG_y', 'PIc_y', 'PIw_y', 'Pord', 'Pref', 'wref', 'wt']

g_args = ['Kcc',
 'Kpc',
 'Kpw',
 'LG_y',
 'PIc_hl_zi',
 'PIc_hl_zl',
 'PIc_hl_zu',
 'PIc_xi',
 'PIc_y',
 'PIc_yul',
 'PIw_hl_zi',
 'PIw_hl_zl',
 'PIw_hl_zu',
 'PIw_xi',
 'PIw_y',
 'PIw_yul',
 'Pord',
 'Pref',
 'theta0',
 'thmax',
 'thmin',
 'wref',
 'wref0',
 'wt']

j_args = {'fx': ['Kcc', 'Kic', 'Kiw'],
 'fy': ['Kcc', 'Kic', 'Kiw'],
 'gx': ['Kcc', 'Kpc', 'Kpw'],
 'gy': ['Kcc', 'Kpc', 'Kpw', 'PIc_hl_zi', 'PIw_hl_zi']}

s_args = OrderedDict([('wref0', ['wref'])])

sns_args = []

ia_args = OrderedDict([('PIc_xi', []),
             ('PIw_xi', []),
             ('PIc_yul', ['Kpc', 'Pord', 'Pref']),
             ('PIc_y',
              ['PIc_hl_zi',
               'PIc_hl_zl',
               'PIc_hl_zu',
               'PIc_yul',
               'thmax',
               'thmin']),
             ('wref', ['wt']),
             ('PIw_yul', ['Kcc', 'Kpw', 'Pord', 'Pref', 'wref', 'wt']),
             ('PIw_y',
              ['PIw_hl_zi',
               'PIw_hl_zl',
               'PIw_hl_zu',
               'PIw_yul',
               'thmax',
               'thmin']),
             ('LG_y', ['PIc_y', 'PIw_y'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 1, 2]),
             ('fyc', []),
             ('fy', [0, 1, 1, 2, 2]),
             ('gxc', []),
             ('gx', [0, 0, 3, 3, 3, 5]),
             ('gyc', []),
             ('gy', [0, 0, 1, 1, 2, 3, 3, 3, 4, 4])])

jjac = OrderedDict([('fxc', []),
             ('fx', [4, 3, 4, 2]),
             ('fyc', []),
             ('fy', [11, 7, 11, 6, 9]),
             ('gxc', []),
             ('gx', [0, 4, 1, 3, 4, 2]),
             ('gyc', []),
             ('gy', [5, 11, 5, 6, 7, 7, 8, 11, 8, 9])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0]),
             ('gyc', []),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['fx', 'fy', 'gx', 'gy']

init_seq = ['PIc_xi',
 'PIw_xi',
 'Pord',
 'Pref',
 'PIc_yul',
 'PIc_y',
 'wt',
 'wref',
 'PIw_yul',
 'PIw_y',
 'LG_y',
 'theta']

need_diag_eps = []
