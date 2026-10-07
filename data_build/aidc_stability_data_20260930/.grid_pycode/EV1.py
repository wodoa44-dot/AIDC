from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "f13337c6d3a28c4d70257f37258ea347"

def f_update(En, EtaC, EtaD, Ipcmd_y, Ipout_y, Iqcmd_y, Iqout_y, LTN_z0, LTN_z1, sys_mva, v):
    return (1.0*Ipcmd_y - Ipout_y, 1.0*Iqcmd_y - Iqout_y, (1/3600)*sys_mva*(-EtaC*Ipout_y*LTN_z1*v - Ipout_y*LTN_z0*v/EtaD)/En, 0,)


def g_update(DB_db_zl, DB_db_zu, DB_y, FL1_zi, FL1_zu, FL2_zi, FL2_zl, Fdev, Ffh, Ffl, Fvh, Fvl, Ipcmd_lim_zi, Ipcmd_lim_zu, Ipcmd_x, Ipcmd_y, Ipmax, Ipmaxsq, Ipmin, Ipout_y, Ipul, Iqcmd_lim_zi, Iqcmd_lim_zl, Iqcmd_lim_zu, Iqcmd_x, Iqcmd_y, Iqmax, Iqmaxsq, Iqout_y, Iqul, Kft01, Kft23, Kvt01, Kvt23, PHL_zi, PHL_zl, PHL_zu, Pext, Pext0, Pref, Psum, Qdrp, Qref, Qsum, SOClim_zl, SOClim_zu, SWPQ_s0, SWPQ_s1, VL1_zi, VL1_zu, VL2_zi, VL2_zl, VLo_zi, VLo_zl, VQ1_zi, VQ1_zl, VQ2_zi, VQ2_zu, Vcomp, Vqu, ddn, dqdv, f, fHz, fdbd, fn, ft0, ft2, ialim, pmn, pmx, pref0, qmn, qmx, qref0, recflag, u, v, v1, vp, vt0, vt2):
    return (f*fn - fHz, FL1_zi*Kft01*(fHz - ft0) + FL1_zu - Ffl, FL2_zi*(Kft23*(-fHz + ft2) + 1) + FL2_zl - Ffh, -Fdev - fHz + fn, -DB_y + ddn*(DB_db_zl*(Fdev - fdbd) + DB_db_zu*Fdev), -Fvl + Kvt01*VL1_zi*(v - vt0) + VL1_zu, -Fvh + VL2_zi*(Kvt23*(-v + vt2) + 1) + VL2_zl, VLo_zi*v + 0.01*VLo_zl - vp, -Pext + Pext0*u, -Pref + pref0*u, -Psum + u*(DB_y + Pext + Pref), -Qdrp + VQ1_zi*u*(dqdv*(-Vcomp + Vqu) + qmx) + VQ1_zl*qmx*u + VQ2_zi*dqdv*u*(-Vcomp + v1) + VQ2_zu*qmn, -Qref + qref0*u, -Qsum + u*(Qdrp + Qref), -Ipul + (PHL_zi*Psum + PHL_zl*pmn + PHL_zu*pmx)/vp, -Iqul + Qsum/vp, -Ipmax + (1 - SOClim_zl)*(sqrt(Ipmaxsq)*SWPQ_s0 + SWPQ_s1*ialim), -Iqmax + sqrt(Iqmaxsq)*SWPQ_s1 + SWPQ_s0*ialim, -Ipcmd_x + Ipul, Ipcmd_lim_zi*Ipcmd_x*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1) + Ipcmd_lim_zu*Ipmax*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1) - Ipcmd_y, -Iqcmd_x + Iqul, Iqcmd_lim_zi*Iqcmd_x*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1) - Iqcmd_lim_zl*Iqmax*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1) + Iqcmd_lim_zu*Iqmax*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1) - Iqcmd_y, -Ipmin + (SOClim_zu - 1)*(sqrt(Ipmaxsq)*SWPQ_s0 + SWPQ_s1*ialim), -Ipout_y*u*v, -Iqout_y*u*v, 0,)


def fx_update(En, EtaC, EtaD, LTN_z0, LTN_z1, sys_mva, v):
    return (-1, -1, (1/3600)*sys_mva*(-EtaC*LTN_z1*v - LTN_z0*v/EtaD)/En)


def fy_update(En, EtaC, EtaD, Ipout_y, LTN_z0, LTN_z1, sys_mva):
    return (1.0, 1.0, (1/3600)*sys_mva*(-EtaC*Ipout_y*LTN_z1 - Ipout_y*LTN_z0/EtaD)/En)


def gy_update(DB_db_zl, DB_db_zu, FL1_zi, FL2_zi, Ffh, Ffl, Fvh, Fvl, Ipcmd_lim_zi, Ipcmd_lim_zu, Ipcmd_x, Ipmax, Ipout_y, Iqcmd_lim_zi, Iqcmd_lim_zl, Iqcmd_lim_zu, Iqcmd_x, Iqmax, Iqout_y, Kft01, Kft23, Kvt01, Kvt23, PHL_zi, PHL_zl, PHL_zu, Psum, Qsum, VL1_zi, VL2_zi, VLo_zi, ddn, fn, pmn, pmx, recflag, u, vp):
    return (-1, fn, FL1_zi*Kft01, -1, -FL2_zi*Kft23, -1, -1, -1, ddn*(DB_db_zl + DB_db_zu), -1, -1, Kvt01*VL1_zi, -1, -Kvt23*VL2_zi, -1, VLo_zi, -1, -1, u, u, u, -1, -1, -1, u, u, -1, -(PHL_zi*Psum + PHL_zl*pmn + PHL_zu*pmx)/vp**2, PHL_zi/vp, -1, -Qsum/vp**2, vp**(-1.0), -1, -1, -1, 1, -1, Ffh*Fvh*Fvl*Ipcmd_lim_zi*Ipcmd_x*recflag + Ffh*Fvh*Fvl*Ipcmd_lim_zu*Ipmax*recflag, Ffl*Fvh*Fvl*Ipcmd_lim_zi*Ipcmd_x*recflag + Ffl*Fvh*Fvl*Ipcmd_lim_zu*Ipmax*recflag, Ffh*Ffl*Fvh*Ipcmd_lim_zi*Ipcmd_x*recflag + Ffh*Ffl*Fvh*Ipcmd_lim_zu*Ipmax*recflag, Ffh*Ffl*Fvl*Ipcmd_lim_zi*Ipcmd_x*recflag + Ffh*Ffl*Fvl*Ipcmd_lim_zu*Ipmax*recflag, Ipcmd_lim_zu*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1), Ipcmd_lim_zi*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1), -1, 1, -1, Ffh*Fvh*Fvl*Iqcmd_lim_zi*Iqcmd_x*recflag - Ffh*Fvh*Fvl*Iqcmd_lim_zl*Iqmax*recflag + Ffh*Fvh*Fvl*Iqcmd_lim_zu*Iqmax*recflag, Ffl*Fvh*Fvl*Iqcmd_lim_zi*Iqcmd_x*recflag - Ffl*Fvh*Fvl*Iqcmd_lim_zl*Iqmax*recflag + Ffl*Fvh*Fvl*Iqcmd_lim_zu*Iqmax*recflag, Ffh*Ffl*Fvh*Iqcmd_lim_zi*Iqcmd_x*recflag - Ffh*Ffl*Fvh*Iqcmd_lim_zl*Iqmax*recflag + Ffh*Ffl*Fvh*Iqcmd_lim_zu*Iqmax*recflag, Ffh*Ffl*Fvl*Iqcmd_lim_zi*Iqcmd_x*recflag - Ffh*Ffl*Fvl*Iqcmd_lim_zl*Iqmax*recflag + Ffh*Ffl*Fvl*Iqcmd_lim_zu*Iqmax*recflag, -Iqcmd_lim_zl*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1) + Iqcmd_lim_zu*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1), Iqcmd_lim_zi*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1), -1, -1, -Ipout_y*u, -Iqout_y*u)


def gx_update(u, v):
    return (-u*v, -u*v)


def fHz_ia(f, fn):
    return f*fn


def Ffh_ia(FL2_zi, FL2_zl, Kft23, fHz, ft2):
    return FL2_zi*(Kft23*(-fHz + ft2) + 1) + FL2_zl


def Ffl_ia(FL1_zi, FL1_zu, Kft01, fHz, ft0):
    return FL1_zi*Kft01*(fHz - ft0) + FL1_zu


def Fvh_ia(Kvt23, VL2_zi, VL2_zl, v, vt2):
    return VL2_zi*(Kvt23*(-v + vt2) + 1) + VL2_zl


def Fvl_ia(Kvt01, VL1_zi, VL1_zu, v, vt0):
    return Kvt01*VL1_zi*(v - vt0) + VL1_zu


def Fdev_ia(fHz, fn):
    return -fHz + fn


def DB_y_ia(DB_db_zl, DB_db_zu, Fdev, ddn, fdbd):
    return ddn*(DB_db_zl*(Fdev - fdbd) + DB_db_zu*Fdev)


def Pext_ia(Pext0, u):
    return Pext0*u


def Pref_ia(pref0, u):
    return pref0*u


def Psum_ia(DB_y, Pext, Pref, u):
    return u*(DB_y + Pext + Pref)


def vp_ia(VLo_zi, VLo_zl, v):
    return VLo_zi*v + 0.01*VLo_zl


def Ipul_ia(PHL_zi, PHL_zl, PHL_zu, Psum, pmn, pmx, vp):
    return (PHL_zi*Psum + PHL_zl*pmn + PHL_zu*pmx)/vp


def Ipcmd_x_ia(Ipul):
    return Ipul


def Ipmax_ia(Ipmaxsq0, SOClim_zl, SWPQ_s0, SWPQ_s1, ialim):
    return (1 - SOClim_zl)*(sqrt(Ipmaxsq0)*SWPQ_s0 + SWPQ_s1*ialim)


def Ipmin_ia(Ipmaxsq0, SOClim_zu, SWPQ_s0, SWPQ_s1, ialim):
    return (SOClim_zu - 1)*(sqrt(Ipmaxsq0)*SWPQ_s0 + SWPQ_s1*ialim)


def Ipcmd_y_ia(Ffh, Ffl, Fvh, Fvl, Ipcmd_lim_zi, Ipcmd_lim_zu, Ipcmd_x, Ipmax, recflag):
    return Ipcmd_lim_zi*Ipcmd_x*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1) + Ipcmd_lim_zu*Ipmax*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1)


def Ipout_y_ia(Ipcmd_y):
    return 1.0*Ipcmd_y


def Qsum_ia(VQ1_zi, VQ1_zl, VQ2_zi, VQ2_zu, Vcomp, Vqu, dqdv, qmn, qmx, qref0, u, v1):
    return u*(VQ1_zi*u*(dqdv*(-Vcomp + Vqu) + qmx) + VQ1_zl*qmx*u + VQ2_zi*dqdv*u*(-Vcomp + v1) + VQ2_zu*qmn + qref0)


def Iqul_ia(Qsum, vp):
    return Qsum/vp


def Iqcmd_x_ia(Iqul):
    return Iqul


def Iqmax_ia(Iqmaxsq0, SWPQ_s0, SWPQ_s1, ialim):
    return sqrt(Iqmaxsq0)*SWPQ_s1 + SWPQ_s0*ialim


def Iqcmd_y_ia(Ffh, Ffl, Fvh, Fvl, Iqcmd_lim_zi, Iqcmd_lim_zl, Iqcmd_lim_zu, Iqcmd_x, Iqmax, recflag):
    return Iqcmd_lim_zi*Iqcmd_x*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1) - Iqcmd_lim_zl*Iqmax*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1) + Iqcmd_lim_zu*Iqmax*(Ffh*Ffl*Fvh*Fvl*recflag - recflag + 1)


def Iqout_y_ia(Iqcmd_y):
    return 1.0*Iqcmd_y


def pIG_y_ia(SOCinit):
    return SOCinit


def Qdrp_ia(VQ1_zi, VQ1_zl, VQ2_zi, VQ2_zu, Vcomp, Vqu, dqdv, qmn, qmx, u, v1):
    return VQ1_zi*u*(dqdv*(-Vcomp + Vqu) + qmx) + VQ1_zl*qmx*u + VQ2_zi*dqdv*u*(-Vcomp + v1) + VQ2_zu*qmn


def Qref_ia(qref0, u):
    return qref0*u


def pref0_svc(gammap, p0s):
    return gammap*p0s


def qref0_svc(gammaq, q0s):
    return gammaq*q0s


def Kft01_svc(ft0, ft1):
    return (-ft0 + ft1)**(-1.0)


def Kft23_svc(ft2, ft3):
    return (-ft2 + ft3)**(-1.0)


def Kvt01_svc(vt0, vt1):
    return (-vt0 + vt1)**(-1.0)


def Kvt23_svc(vt2, vt3):
    return (-vt2 + vt3)**(-1.0)


def Pext0_svc():
    return 0


def Vcomp_svc(Ipout_y, Iqout_y, a, v, xc):
    return abs(v*exp(1j*a) + 1j*xc*(Ipout_y + 1j*Iqout_y))


def Vqu_svc(dqdv, qmn, qref0, v1):
    return v1 - (-qmn + qref0)/dqdv


def Vql_svc(dqdv, qmx, qref0, v0):
    return v0 + (qmx - qref0)/dqdv


def Ipmaxsq_svc(Iqcmd_y, ialim, __zeros, __ones, __falses, __trues):
    return select([greater_equal(Iqcmd_y**2 - ialim**2, 0),__trues], [__zeros,-Iqcmd_y**2 + ialim**2], default=nan)


def Ipmaxsq0_svc(ialim, qref0, u, v, __zeros, __ones, __falses, __trues):
    return select([less_equal(ialim**2 - qref0**2*u**2/v**2, 0),__trues], [__zeros,ialim**2 - qref0**2*u**2/v**2], default=nan)


def Iqmaxsq_svc(Ipcmd_y, ialim, __zeros, __ones, __falses, __trues):
    return select([greater_equal(Ipcmd_y**2 - ialim**2, 0),__trues], [__zeros,-Ipcmd_y**2 + ialim**2], default=nan)


def Iqmaxsq0_svc(ialim, pref0, u, v, __zeros, __ones, __falses, __trues):
    return select([less_equal(ialim**2 - pref0**2*u**2/v**2, 0),__trues], [__zeros,ialim**2 - pref0**2*u**2/v**2], default=nan)


# empty sns_update

f_args = ['En',
 'EtaC',
 'EtaD',
 'Ipcmd_y',
 'Ipout_y',
 'Iqcmd_y',
 'Iqout_y',
 'LTN_z0',
 'LTN_z1',
 'sys_mva',
 'v']

g_args = ['DB_db_zl',
 'DB_db_zu',
 'DB_y',
 'FL1_zi',
 'FL1_zu',
 'FL2_zi',
 'FL2_zl',
 'Fdev',
 'Ffh',
 'Ffl',
 'Fvh',
 'Fvl',
 'Ipcmd_lim_zi',
 'Ipcmd_lim_zu',
 'Ipcmd_x',
 'Ipcmd_y',
 'Ipmax',
 'Ipmaxsq',
 'Ipmin',
 'Ipout_y',
 'Ipul',
 'Iqcmd_lim_zi',
 'Iqcmd_lim_zl',
 'Iqcmd_lim_zu',
 'Iqcmd_x',
 'Iqcmd_y',
 'Iqmax',
 'Iqmaxsq',
 'Iqout_y',
 'Iqul',
 'Kft01',
 'Kft23',
 'Kvt01',
 'Kvt23',
 'PHL_zi',
 'PHL_zl',
 'PHL_zu',
 'Pext',
 'Pext0',
 'Pref',
 'Psum',
 'Qdrp',
 'Qref',
 'Qsum',
 'SOClim_zl',
 'SOClim_zu',
 'SWPQ_s0',
 'SWPQ_s1',
 'VL1_zi',
 'VL1_zu',
 'VL2_zi',
 'VL2_zl',
 'VLo_zi',
 'VLo_zl',
 'VQ1_zi',
 'VQ1_zl',
 'VQ2_zi',
 'VQ2_zu',
 'Vcomp',
 'Vqu',
 'ddn',
 'dqdv',
 'f',
 'fHz',
 'fdbd',
 'fn',
 'ft0',
 'ft2',
 'ialim',
 'pmn',
 'pmx',
 'pref0',
 'qmn',
 'qmx',
 'qref0',
 'recflag',
 'u',
 'v',
 'v1',
 'vp',
 'vt0',
 'vt2']

j_args = {'fx': ['En', 'EtaC', 'EtaD', 'LTN_z0', 'LTN_z1', 'sys_mva', 'v'],
 'fy': ['En', 'EtaC', 'EtaD', 'Ipout_y', 'LTN_z0', 'LTN_z1', 'sys_mva'],
 'gx': ['u', 'v'],
 'gy': ['DB_db_zl',
        'DB_db_zu',
        'FL1_zi',
        'FL2_zi',
        'Ffh',
        'Ffl',
        'Fvh',
        'Fvl',
        'Ipcmd_lim_zi',
        'Ipcmd_lim_zu',
        'Ipcmd_x',
        'Ipmax',
        'Ipout_y',
        'Iqcmd_lim_zi',
        'Iqcmd_lim_zl',
        'Iqcmd_lim_zu',
        'Iqcmd_x',
        'Iqmax',
        'Iqout_y',
        'Kft01',
        'Kft23',
        'Kvt01',
        'Kvt23',
        'PHL_zi',
        'PHL_zl',
        'PHL_zu',
        'Psum',
        'Qsum',
        'VL1_zi',
        'VL2_zi',
        'VLo_zi',
        'ddn',
        'fn',
        'pmn',
        'pmx',
        'recflag',
        'u',
        'vp']}

s_args = OrderedDict([('pref0', ['gammap', 'p0s']),
             ('qref0', ['gammaq', 'q0s']),
             ('Kft01', ['ft0', 'ft1']),
             ('Kft23', ['ft2', 'ft3']),
             ('Kvt01', ['vt0', 'vt1']),
             ('Kvt23', ['vt2', 'vt3']),
             ('Pext0', []),
             ('Vcomp', ['Ipout_y', 'Iqout_y', 'a', 'v', 'xc']),
             ('Vqu', ['dqdv', 'qmn', 'qref0', 'v1']),
             ('Vql', ['dqdv', 'qmx', 'qref0', 'v0']),
             ('Ipmaxsq',
              ['Iqcmd_y', 'ialim', '__zeros', '__ones', '__falses', '__trues']),
             ('Ipmaxsq0',
              ['ialim',
               'qref0',
               'u',
               'v',
               '__zeros',
               '__ones',
               '__falses',
               '__trues']),
             ('Iqmaxsq',
              ['Ipcmd_y', 'ialim', '__zeros', '__ones', '__falses', '__trues']),
             ('Iqmaxsq0',
              ['ialim',
               'pref0',
               'u',
               'v',
               '__zeros',
               '__ones',
               '__falses',
               '__trues'])])

sns_args = []

ia_args = OrderedDict([('fHz', ['f', 'fn']),
             ('Ffh', ['FL2_zi', 'FL2_zl', 'Kft23', 'fHz', 'ft2']),
             ('Ffl', ['FL1_zi', 'FL1_zu', 'Kft01', 'fHz', 'ft0']),
             ('Fvh', ['Kvt23', 'VL2_zi', 'VL2_zl', 'v', 'vt2']),
             ('Fvl', ['Kvt01', 'VL1_zi', 'VL1_zu', 'v', 'vt0']),
             ('Fdev', ['fHz', 'fn']),
             ('DB_y', ['DB_db_zl', 'DB_db_zu', 'Fdev', 'ddn', 'fdbd']),
             ('Pext', ['Pext0', 'u']),
             ('Pref', ['pref0', 'u']),
             ('Psum', ['DB_y', 'Pext', 'Pref', 'u']),
             ('vp', ['VLo_zi', 'VLo_zl', 'v']),
             ('Ipul',
              ['PHL_zi', 'PHL_zl', 'PHL_zu', 'Psum', 'pmn', 'pmx', 'vp']),
             ('Ipcmd_x', ['Ipul']),
             ('Ipmax',
              ['Ipmaxsq0', 'SOClim_zl', 'SWPQ_s0', 'SWPQ_s1', 'ialim']),
             ('Ipmin',
              ['Ipmaxsq0', 'SOClim_zu', 'SWPQ_s0', 'SWPQ_s1', 'ialim']),
             ('Ipcmd_y',
              ['Ffh',
               'Ffl',
               'Fvh',
               'Fvl',
               'Ipcmd_lim_zi',
               'Ipcmd_lim_zu',
               'Ipcmd_x',
               'Ipmax',
               'recflag']),
             ('Ipout_y', ['Ipcmd_y']),
             ('Qsum',
              ['VQ1_zi',
               'VQ1_zl',
               'VQ2_zi',
               'VQ2_zu',
               'Vcomp',
               'Vqu',
               'dqdv',
               'qmn',
               'qmx',
               'qref0',
               'u',
               'v1']),
             ('Iqul', ['Qsum', 'vp']),
             ('Iqcmd_x', ['Iqul']),
             ('Iqmax', ['Iqmaxsq0', 'SWPQ_s0', 'SWPQ_s1', 'ialim']),
             ('Iqcmd_y',
              ['Ffh',
               'Ffl',
               'Fvh',
               'Fvl',
               'Iqcmd_lim_zi',
               'Iqcmd_lim_zl',
               'Iqcmd_lim_zu',
               'Iqcmd_x',
               'Iqmax',
               'recflag']),
             ('Iqout_y', ['Iqcmd_y']),
             ('pIG_y', ['SOCinit']),
             ('Qdrp',
              ['VQ1_zi',
               'VQ1_zl',
               'VQ2_zi',
               'VQ2_zu',
               'Vcomp',
               'Vqu',
               'dqdv',
               'qmn',
               'qmx',
               'u',
               'v1']),
             ('Qref', ['qref0', 'u'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 2]),
             ('fyc', []),
             ('fy', [0, 1, 2]),
             ('gxc', []),
             ('gx', [23, 24]),
             ('gyc', []),
             ('gy',
              [0,
               0,
               1,
               1,
               2,
               2,
               3,
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
               9,
               10,
               10,
               10,
               10,
               11,
               12,
               13,
               13,
               13,
               14,
               14,
               14,
               15,
               15,
               15,
               16,
               17,
               18,
               18,
               19,
               19,
               19,
               19,
               19,
               19,
               19,
               20,
               20,
               21,
               21,
               21,
               21,
               21,
               21,
               21,
               22,
               23,
               24])])

jjac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 0]),
             ('fyc', []),
             ('fy', [23, 25, 28]),
             ('gxc', []),
             ('gx', [0, 1]),
             ('gyc', []),
             ('gy',
              [4,
               29,
               4,
               5,
               4,
               6,
               4,
               7,
               7,
               8,
               9,
               28,
               10,
               28,
               11,
               28,
               12,
               13,
               8,
               12,
               13,
               14,
               15,
               16,
               15,
               16,
               17,
               11,
               14,
               18,
               11,
               17,
               19,
               20,
               21,
               18,
               22,
               5,
               6,
               9,
               10,
               20,
               22,
               23,
               19,
               24,
               5,
               6,
               9,
               10,
               21,
               24,
               25,
               26,
               28,
               28])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0]),
             ('gyc', []),
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

init_seq = ['f',
 'fHz',
 'Ffh',
 'Ffl',
 'v',
 'Fvh',
 'Fvl',
 'Fdev',
 'DB_y',
 'Pext',
 'Pref',
 'Psum',
 'vp',
 'Ipul',
 'Ipcmd_x',
 'Ipmax',
 'Ipmin',
 'Ipcmd_y',
 'Ipout_y',
 'Qsum',
 'Iqul',
 'Iqcmd_x',
 'Iqmax',
 'Iqcmd_y',
 'Iqout_y',
 'pIG_y',
 'SOC',
 'Qdrp',
 'Qref',
 'a']

need_diag_eps = []
