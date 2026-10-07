from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "a6e48a309498391e3d9c529271962511"

def f_update(Kc, Ki, Kig, Perr, Pline, Qline, SWVC_s0, SWVC_s1, Vcomp, eHld, feHL_zi, feHL_zl, feHL_zu, femax, femin, s0_y, s1_y, s2_y, s2_ys, s3_x, s4_y, s5_y, s5_ys, s6_y, v):
    return (SWVC_s0*(Kc*Qline + v) + SWVC_s1*Vcomp - s0_y, Qline - s1_y, Ki*(eHld + 2*s2_y - 2*s2_ys), s2_y - s3_x, Pline - s4_y, Kig*(Perr*feHL_zi + feHL_zl*femin + feHL_zu*femax + 2*s5_y - 2*s5_ys), s5_y - s6_y,)


def g_update(Ddn, Dup, Freq_ref, Kp, Kpg, Perr, Plant_pref, Plerr, Pline0, Pmax, Pmin, Qline0, Qlinef, Qmax, Qmin, Refsel, SWF_s1, SWPL_s1, SWRef_s0, SWRef_s1, Tft, Tfv, Vref, Vref0, dbd1, dbd2, dbd_db_zl, dbd_db_zu, dbd_y, eHL_zi, eHL_zl, eHL_zu, eHld, emax, emin, enf, f, fdbd1, fdbd2, fdbd_db_zl, fdbd_db_zu, fdbd_y, fdlt0_z0, fdlt0_z1, feHL_zi, feHL_zl, feHL_zu, femax, femin, ferr, s0_y, s1_y, s2_lim_zi, s2_lim_zl, s2_lim_zu, s2_xi, s2_y, s2_ys, s3_LT1_z1, s3_LT2_z1, s3_x, s3_y, s4_y, s5_lim_zi, s5_lim_zl, s5_lim_zu, s5_xi, s5_y, s5_ys, s6_y):
    return (-Vref + Vref0, Qline0 - Qlinef, -Refsel + SWRef_s0*(Qlinef - s1_y) + SWRef_s1*(Vref - s0_y), 1.0*dbd_db_zl*(Refsel - dbd1) + 1.0*dbd_db_zu*(Refsel - dbd2) - dbd_y, dbd_y*eHL_zi + eHL_zl*emin + eHL_zu*emax - enf, Kp*eHld + s2_xi - s2_ys, Qmax*s2_lim_zu + Qmin*s2_lim_zl + s2_lim_zi*s2_ys - s2_y, Tft*(s2_y - s3_x) + Tfv*s3_x - Tfv*s3_y + s3_LT1_z1*s3_LT2_z1*(-s3_x + s3_y), Freq_ref - f - ferr, 1.0*fdbd_db_zl*(-fdbd1 + ferr) + 1.0*fdbd_db_zu*(-fdbd2 + ferr) - fdbd_y, -Plant_pref + Pline0, Plant_pref - Plerr - s4_y, Ddn*fdbd_y*fdlt0_z1 + Dup*fdbd_y*fdlt0_z0 - Perr + Plerr*SWPL_s1, Kpg*(Perr*feHL_zi + feHL_zl*femin + feHL_zu*femax) + s5_xi - s5_ys, Pmax*s5_lim_zu + Pmin*s5_lim_zl + s5_lim_zi*s5_ys - s5_y, SWF_s1*s6_y, s3_y, 0, 0, 0, 0, 0, 0, 0,)


def fx_update():
    return (-1, -1, -1, -1, -1)


def fy_update(Ki, Kig, SWVC_s0, feHL_zi):
    return (SWVC_s0, -2*Ki, 2*Ki, 1, Kig*feHL_zi, -2*Kig, 2*Kig, 1)


def gy_update(Ddn, Dup, Kpg, SWPL_s1, SWRef_s0, SWRef_s1, Tft, Tfv, dbd_db_zl, dbd_db_zu, eHL_zi, fdbd_db_zl, fdbd_db_zu, fdlt0_z0, fdlt0_z1, feHL_zi, s2_lim_zi, s3_LT1_z1, s3_LT2_z1, s5_lim_zi):
    return (-1, -1, SWRef_s1, SWRef_s0, -1, 1.0*dbd_db_zl + 1.0*dbd_db_zu, -1, eHL_zi, -1, -1, s2_lim_zi, -1, Tft, -Tfv + s3_LT1_z1*s3_LT2_z1, -1, -1, 1.0*fdbd_db_zl + 1.0*fdbd_db_zu, -1, -1, 1, -1, Ddn*fdlt0_z1 + Dup*fdlt0_z0, SWPL_s1, -1, Kpg*feHL_zi, -1, s5_lim_zi, -1, 1)


def gx_update(SWF_s1, SWRef_s0, SWRef_s1, Tft, Tfv, s3_LT1_z1, s3_LT2_z1):
    return (-SWRef_s1, -SWRef_s0, 1, -Tft + Tfv - s3_LT1_z1*s3_LT2_z1, -1, 1, SWF_s1)


def s0_y_ia(Kc, Qline, SWVC_s0, SWVC_s1, Vcomp, v):
    return SWVC_s0*(Kc*Qline + v) + SWVC_s1*Vcomp


def s1_y_ia(Qline):
    return Qline


def s2_xi_ia():
    return 0.0


def s2_ys_ia(Kp, eHld):
    return Kp*eHld


def s2_y_ia(Qmax, Qmin, s2_lim_zi, s2_lim_zl, s2_lim_zu, s2_ys):
    return Qmax*s2_lim_zu + Qmin*s2_lim_zl + s2_lim_zi*s2_ys


def s3_x_ia(s2_y):
    return s2_y


def s4_y_ia(Pline):
    return Pline


def s5_xi_ia():
    return 0.0


def Plant_pref_ia(Pline0):
    return Pline0


def Plerr_ia(Plant_pref, s4_y):
    return Plant_pref - s4_y


def ferr_ia(Freq_ref, f):
    return Freq_ref - f


def fdbd_y_ia(fdbd1, fdbd2, fdbd_db_zl, fdbd_db_zu, ferr):
    return 1.0*fdbd_db_zl*(-fdbd1 + ferr) + 1.0*fdbd_db_zu*(-fdbd2 + ferr)


def Perr_ia(Ddn, Dup, Plerr, SWPL_s1, fdbd_y, fdlt0_z0, fdlt0_z1):
    return Ddn*fdbd_y*fdlt0_z1 + Dup*fdbd_y*fdlt0_z0 + Plerr*SWPL_s1


def s5_ys_ia(Kpg, Perr, feHL_zi, feHL_zl, feHL_zu, femax, femin):
    return Kpg*(Perr*feHL_zi + feHL_zl*femin + feHL_zu*femax)


def s5_y_ia(Pmax, Pmin, s5_lim_zi, s5_lim_zl, s5_lim_zu, s5_ys):
    return Pmax*s5_lim_zu + Pmin*s5_lim_zl + s5_lim_zi*s5_ys


def s6_y_ia(s5_y):
    return s5_y


def Vref_ia(Vref0):
    return Vref0


def Qlinef_ia(Qline0):
    return Qline0


def Refsel_ia(Qlinef, SWRef_s0, SWRef_s1, Vref, s0_y, s1_y):
    return SWRef_s0*(Qlinef - s1_y) + SWRef_s1*(Vref - s0_y)


def dbd_y_ia(Refsel, dbd1, dbd2, dbd_db_zl, dbd_db_zu):
    return 1.0*dbd_db_zl*(Refsel - dbd1) + 1.0*dbd_db_zu*(Refsel - dbd2)


def enf_ia(dbd_y, eHL_zi, eHL_zl, eHL_zu, emax, emin):
    return dbd_y*eHL_zi + eHL_zl*emin + eHL_zu*emax


def s3_y_ia(s2_y):
    return s2_y


def Isign_svc():
    return 0


def Iline_svc(Isign, a1, a2, r, v1, v2, x):
    return Isign*(v1*exp(1j*a1) - v2*exp(1j*a2))/(r + 1j*x)


def Iline0_svc(Iline):
    return Iline


def Pline_svc(Isign, a1, a2, r, v1, v2, x):
    return real(Isign*v1*conj((v1*exp(1j*a1) - v2*exp(1j*a2))/(r + 1j*x))*exp(1j*a1))


def Pline0_svc(Pline):
    return Pline


def Qline_svc(Isign, a1, a2, r, v1, v2, x):
    return imag(Isign*v1*conj((v1*exp(1j*a1) - v2*exp(1j*a2))/(r + 1j*x))*exp(1j*a1))


def Qline0_svc(Qline):
    return Qline


def Vcomp_svc(Iline, Rcs, Xcs, a, v):
    return abs(Iline*(Rcs + 1j*Xcs) - v*exp(1j*a))


def Vref0_svc(Kc, Qline0, SWVC_s0, SWVC_s1, Vcomp, v):
    return SWVC_s0*(Kc*Qline0 + v) + SWVC_s1*Vcomp


def zf_svc(Vfrz, freeze, v):
    return freeze*(less(v, Vfrz))


def eHld_svc():
    return 0


def Freq_ref_svc():
    return 1.00000000000000


# empty sns_update

f_args = ['Kc',
 'Ki',
 'Kig',
 'Perr',
 'Pline',
 'Qline',
 'SWVC_s0',
 'SWVC_s1',
 'Vcomp',
 'eHld',
 'feHL_zi',
 'feHL_zl',
 'feHL_zu',
 'femax',
 'femin',
 's0_y',
 's1_y',
 's2_y',
 's2_ys',
 's3_x',
 's4_y',
 's5_y',
 's5_ys',
 's6_y',
 'v']

g_args = ['Ddn',
 'Dup',
 'Freq_ref',
 'Kp',
 'Kpg',
 'Perr',
 'Plant_pref',
 'Plerr',
 'Pline0',
 'Pmax',
 'Pmin',
 'Qline0',
 'Qlinef',
 'Qmax',
 'Qmin',
 'Refsel',
 'SWF_s1',
 'SWPL_s1',
 'SWRef_s0',
 'SWRef_s1',
 'Tft',
 'Tfv',
 'Vref',
 'Vref0',
 'dbd1',
 'dbd2',
 'dbd_db_zl',
 'dbd_db_zu',
 'dbd_y',
 'eHL_zi',
 'eHL_zl',
 'eHL_zu',
 'eHld',
 'emax',
 'emin',
 'enf',
 'f',
 'fdbd1',
 'fdbd2',
 'fdbd_db_zl',
 'fdbd_db_zu',
 'fdbd_y',
 'fdlt0_z0',
 'fdlt0_z1',
 'feHL_zi',
 'feHL_zl',
 'feHL_zu',
 'femax',
 'femin',
 'ferr',
 's0_y',
 's1_y',
 's2_lim_zi',
 's2_lim_zl',
 's2_lim_zu',
 's2_xi',
 's2_y',
 's2_ys',
 's3_LT1_z1',
 's3_LT2_z1',
 's3_x',
 's3_y',
 's4_y',
 's5_lim_zi',
 's5_lim_zl',
 's5_lim_zu',
 's5_xi',
 's5_y',
 's5_ys',
 's6_y']

j_args = {'fx': [],
 'fy': ['Ki', 'Kig', 'SWVC_s0', 'feHL_zi'],
 'gx': ['SWF_s1',
        'SWRef_s0',
        'SWRef_s1',
        'Tft',
        'Tfv',
        's3_LT1_z1',
        's3_LT2_z1'],
 'gy': ['Ddn',
        'Dup',
        'Kpg',
        'SWPL_s1',
        'SWRef_s0',
        'SWRef_s1',
        'Tft',
        'Tfv',
        'dbd_db_zl',
        'dbd_db_zu',
        'eHL_zi',
        'fdbd_db_zl',
        'fdbd_db_zu',
        'fdlt0_z0',
        'fdlt0_z1',
        'feHL_zi',
        's2_lim_zi',
        's3_LT1_z1',
        's3_LT2_z1',
        's5_lim_zi']}

s_args = OrderedDict([('Isign', []),
             ('Iline', ['Isign', 'a1', 'a2', 'r', 'v1', 'v2', 'x']),
             ('Iline0', ['Iline']),
             ('Pline', ['Isign', 'a1', 'a2', 'r', 'v1', 'v2', 'x']),
             ('Pline0', ['Pline']),
             ('Qline', ['Isign', 'a1', 'a2', 'r', 'v1', 'v2', 'x']),
             ('Qline0', ['Qline']),
             ('Vcomp', ['Iline', 'Rcs', 'Xcs', 'a', 'v']),
             ('Vref0', ['Kc', 'Qline0', 'SWVC_s0', 'SWVC_s1', 'Vcomp', 'v']),
             ('zf', ['Vfrz', 'freeze', 'v']),
             ('eHld', []),
             ('Freq_ref', [])])

sns_args = []

ia_args = OrderedDict([('s0_y', ['Kc', 'Qline', 'SWVC_s0', 'SWVC_s1', 'Vcomp', 'v']),
             ('s1_y', ['Qline']),
             ('s2_xi', []),
             ('s2_ys', ['Kp', 'eHld']),
             ('s2_y',
              ['Qmax', 'Qmin', 's2_lim_zi', 's2_lim_zl', 's2_lim_zu', 's2_ys']),
             ('s3_x', ['s2_y']),
             ('s4_y', ['Pline']),
             ('s5_xi', []),
             ('Plant_pref', ['Pline0']),
             ('Plerr', ['Plant_pref', 's4_y']),
             ('ferr', ['Freq_ref', 'f']),
             ('fdbd_y', ['fdbd1', 'fdbd2', 'fdbd_db_zl', 'fdbd_db_zu', 'ferr']),
             ('Perr',
              ['Ddn',
               'Dup',
               'Plerr',
               'SWPL_s1',
               'fdbd_y',
               'fdlt0_z0',
               'fdlt0_z1']),
             ('s5_ys',
              ['Kpg',
               'Perr',
               'feHL_zi',
               'feHL_zl',
               'feHL_zu',
               'femax',
               'femin']),
             ('s5_y',
              ['Pmax', 'Pmin', 's5_lim_zi', 's5_lim_zl', 's5_lim_zu', 's5_ys']),
             ('s6_y', ['s5_y']),
             ('Vref', ['Vref0']),
             ('Qlinef', ['Qline0']),
             ('Refsel',
              ['Qlinef', 'SWRef_s0', 'SWRef_s1', 'Vref', 's0_y', 's1_y']),
             ('dbd_y', ['Refsel', 'dbd1', 'dbd2', 'dbd_db_zl', 'dbd_db_zu']),
             ('enf', ['dbd_y', 'eHL_zi', 'eHL_zl', 'eHL_zu', 'emax', 'emin']),
             ('s3_y', ['s2_y'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 3, 4, 6]),
             ('fyc', []),
             ('fy', [0, 2, 2, 3, 5, 5, 5, 6]),
             ('gxc', []),
             ('gx', [2, 2, 5, 7, 11, 13, 15]),
             ('gyc', [7]),
             ('gy',
              [0,
               1,
               2,
               2,
               2,
               3,
               3,
               4,
               4,
               5,
               6,
               6,
               7,
               7,
               8,
               8,
               9,
               9,
               10,
               11,
               11,
               12,
               12,
               12,
               13,
               13,
               14,
               14,
               16])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 3, 4, 6]),
             ('fyc', []),
             ('fy', [24, 12, 13, 13, 19, 20, 21, 21]),
             ('gxc', []),
             ('gx', [0, 1, 2, 3, 4, 5, 6]),
             ('gyc', [14]),
             ('gy',
              [7,
               8,
               7,
               8,
               9,
               9,
               10,
               10,
               11,
               12,
               12,
               13,
               13,
               14,
               15,
               26,
               15,
               16,
               17,
               17,
               18,
               16,
               18,
               19,
               19,
               20,
               20,
               21,
               14])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0]),
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
               0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['v',
 's0_y',
 's1_y',
 's2_xi',
 's2_ys',
 's2_y',
 's3_x',
 's4_y',
 's5_xi',
 'Plant_pref',
 'Plerr',
 'f',
 'ferr',
 'fdbd_y',
 'Perr',
 's5_ys',
 's5_y',
 's6_y',
 'Vref',
 'Qlinef',
 'Refsel',
 'dbd_y',
 'enf',
 's3_y',
 'Pext',
 'Qext',
 'a',
 'v1',
 'v2',
 'a1',
 'a2']

need_diag_eps = ['s3_y', 's3_y']
