from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "96659a404aa8b2622c24dbc0f97849f7"

def f_update(Kqi, Kvi, PFsel, PIQ_y, PIQ_ys, PIV_y, PIV_ys, Pe, Pref, Psel, Qerr, S1_y, SWQ_s1, SWV_s0, SWV_s1, Volt_dip, Vsel_y, pfilt_y, s0_y, s4_y, s5_y, v, vp):
    return (-s0_y + v, Pe - S1_y, Kqi*(1 - Volt_dip)*(2*PIQ_y - 2*PIQ_ys + Qerr*SWV_s1), (1 - Volt_dip)*(PFsel/vp - s4_y), Pref - pfilt_y, (1 - Volt_dip)*(Psel - s5_y), Kvi*(1 - Volt_dip)*(2*PIV_y - 2*PIV_ys + SWQ_s1*(-SWV_s0*s0_y + Vsel_y)), 0, 0,)


def g_update(Imaxr, Ip1, Ip2, Ip3, Ip4, IpHL_lim_zi, IpHL_lim_zl, IpHL_lim_zu, IpHL_x, IpHL_y, Ipcmd0, Ipmax, Ipmax2sq, Ipmaxh, Ipmin, Iq1, Iq2, Iq3, Iq4, IqHL_lim_zi, IqHL_lim_zl, IqHL_lim_zu, IqHL_x, IqHL_y, Iqcmd0, Iqfrz, Iqinj, Iqmax, Iqmax2sq, Iqmin, Kf, Kqp, Kqv, Kvp, PFlim_zi, PFlim_zl, PFlim_zu, PFsel, PIQ_lim_zi, PIQ_lim_zl, PIQ_lim_zu, PIQ_xi, PIQ_y, PIQ_ys, PIV_lim_zi, PIV_lim_zl, PIV_lim_zu, PIV_xi, PIV_y, PIV_ys, Pref, Psel, QMax, QMin, Qcpf, Qe, Qerr, Qref, Qsel, S1_y, SWPF_s0, SWPF_s1, SWPQ_s0, SWPQ_s1, SWP_s0, SWP_s1, SWQ_s0, SWQ_s1, SWV_s0, SWV_s1, VDL1_y, VDL1c, VDL2_y, VDL2c, VLower_zi, VLower_zl, VMAX, VMIN, Verr, Volt_dip, Vp1, Vp2, Vp3, Vp4, Vq1, Vq2, Vq3, Vq4, Vref0, Vref1, Vsel_lim_zi, Vsel_lim_zl, Vsel_lim_zu, Vsel_x, Vsel_y, dbV_db_zl, dbV_db_zu, dbV_y, dbd1, dbd2, fThld, fThld2, kVp12, kVp23, kVp34, kVq12, kVq23, kVq34, nThld, omega, p0, pThld, pfaref, pfaref0, pfilt_y, qref0, s0_y, s4_y, s5_y, v, vp, wg, zVDL1, zVDL2, zp_z1, __zeros, __ones, __falses, __trues):
    return (VLower_zi*v + 0.01*VLower_zl - vp, -pfaref + pfaref0, -Qref - qref0, (1 - zp_z1)*(-Qcpf + S1_y*tan(pfaref)), -PFsel + Qcpf*SWPF_s1 + Qref*SWPF_s0, PFlim_zi*PFsel + PFlim_zl*QMin + PFlim_zu*QMax - Qe - Qerr, (1 - Volt_dip)*(Kqp*Qerr*SWV_s1 + PIQ_xi - PIQ_ys), (1 - Volt_dip)*(PIQ_lim_zi*PIQ_ys + PIQ_lim_zl*VMIN + PIQ_lim_zu*VMAX - PIQ_y), PIQ_y*SWV_s1 + SWV_s0*(Qcpf*SWPF_s1 + Qref*SWPF_s0 + Vref1) - Vsel_x, VMAX*Vsel_lim_zu + VMIN*Vsel_lim_zl + Vsel_lim_zi*Vsel_x - Vsel_y, -Verr + Vref0 - s0_y, 1.0*dbV_db_zl*(Verr - dbd1) + 1.0*dbV_db_zu*(Verr - dbd2) - dbV_y, -Iqinj + Kqv*Volt_dip*dbV_y + fThld*(1 - Volt_dip)*(Iqfrz*pThld + Kqv*dbV_y*nThld), 1.0 - wg, -Kf*(omega - 1) - Pref + p0/wg, -Psel + SWP_s0*pfilt_y + SWP_s1*pfilt_y*wg, -VDL1_y + select([less_equal(s0_y, Vq1),less_equal(s0_y, Vq2),less_equal(s0_y, Vq3),less_equal(s0_y, Vq4),greater(s0_y, Vq4),__trues], [Iq1,Iq1 + kVq12*(-Vq1 + s0_y),Iq2 + kVq23*(-Vq2 + s0_y),Iq3 + kVq34*(-Vq3 + s0_y),Iq4,__zeros], default=nan), -VDL2_y + select([less_equal(s0_y, Vp1),less_equal(s0_y, Vp2),less_equal(s0_y, Vp3),less_equal(s0_y, Vp4),greater(s0_y, Vp4),__trues], [Ip1,Ip1 + kVp12*(-Vp1 + s0_y),Ip2 + kVp23*(-Vp2 + s0_y),Ip3 + kVp34*(-Vp3 + s0_y),Ip4,__zeros], default=nan), -Ipmax + Ipmaxh*fThld2 + (1 - fThld2)*(sqrt(Ipmax2sq)*SWPQ_s0 + SWPQ_s1*(zVDL2*(Imaxr*(1 - VDL2c) + VDL2_y*VDL2c) - 100000000.0*zVDL2 + 100000000.0)), -Iqmax + sqrt(Iqmax2sq)*SWPQ_s1 + SWPQ_s0*(zVDL1*(Imaxr*(1 - VDL1c) + VDL1_y*VDL1c) - 100000000.0*zVDL1 + 100000000.0), (1 - Volt_dip)*(Kvp*SWQ_s1*(-SWV_s0*s0_y + Vsel_y) + PIV_xi - PIV_ys), (1 - Volt_dip)*(Iqmax*PIV_lim_zu + Iqmin*PIV_lim_zl + PIV_lim_zi*PIV_ys - PIV_y), PIV_y*SWQ_s1 - Qsel + SWQ_s0*s4_y, -IpHL_x + s5_y/vp, IpHL_lim_zi*IpHL_x + IpHL_lim_zl*Ipmin + IpHL_lim_zu*Ipmax - IpHL_y, -IqHL_x + Iqinj + Qsel, IqHL_lim_zi*IqHL_x + IqHL_lim_zl*Iqmin + IqHL_lim_zu*Iqmax - IqHL_y, 0, 0, 0, 0, IpHL_y - Ipcmd0, -IqHL_y - Iqcmd0,)


def fx_update(Kvi, SWQ_s1, SWV_s0, Volt_dip):
    return (-1, -1, Volt_dip - 1, -1, Volt_dip - 1, -Kvi*SWQ_s1*SWV_s0*(1 - Volt_dip))


def fy_update(Kqi, Kvi, PFsel, SWQ_s1, SWV_s1, Volt_dip, vp):
    return (1, 1, Kqi*SWV_s1*(1 - Volt_dip), -2*Kqi*(1 - Volt_dip), 2*Kqi*(1 - Volt_dip), -PFsel*(1 - Volt_dip)/vp**2, (1 - Volt_dip)/vp, 1, 1 - Volt_dip, Kvi*SWQ_s1*(1 - Volt_dip), -2*Kvi*(1 - Volt_dip), 2*Kvi*(1 - Volt_dip))


def gy_update(IpHL_lim_zi, IpHL_lim_zu, IqHL_lim_zi, IqHL_lim_zu, Kqp, Kqv, Kvp, PFlim_zi, PIQ_lim_zi, PIV_lim_zi, PIV_lim_zu, S1_y, SWPF_s0, SWPF_s1, SWPQ_s0, SWPQ_s1, SWP_s1, SWQ_s1, SWV_s0, SWV_s1, VDL1c, VDL2c, VLower_zi, Volt_dip, Vsel_lim_zi, dbV_db_zl, dbV_db_zu, fThld, fThld2, nThld, p0, pfaref, pfilt_y, s5_y, vp, wg, zVDL1, zVDL2, zp_z1):
    return (-1, VLower_zi, -1, -1, S1_y*(1 - zp_z1)*(tan(pfaref)**2 + 1), zp_z1 - 1, SWPF_s0, SWPF_s1, -1, PFlim_zi, -1, -1, Kqp*SWV_s1*(1 - Volt_dip), Volt_dip - 1, PIQ_lim_zi*(1 - Volt_dip), Volt_dip - 1, SWPF_s0*SWV_s0, SWPF_s1*SWV_s0, SWV_s1, -1, Vsel_lim_zi, -1, -1, 1.0*dbV_db_zl + 1.0*dbV_db_zu, -1, Kqv*Volt_dip + Kqv*fThld*nThld*(1 - Volt_dip), -1, -1, -p0/wg**2, -1, SWP_s1*pfilt_y, -1, -1, -1, SWPQ_s1*VDL2c*zVDL2*(1 - fThld2), -1, SWPQ_s0*VDL1c*zVDL1, -1, Kvp*SWQ_s1*(1 - Volt_dip), Volt_dip - 1, PIV_lim_zu*(1 - Volt_dip), PIV_lim_zi*(1 - Volt_dip), Volt_dip - 1, SWQ_s1, -1, -s5_y/vp**2, -1, IpHL_lim_zu, IpHL_lim_zi, -1, 1, 1, -1, IqHL_lim_zu, IqHL_lim_zi, -1, 1, -1)


def gx_update(Kf, Kvp, SWP_s0, SWP_s1, SWQ_s0, SWQ_s1, SWV_s0, Volt_dip, Vp1, Vp2, Vp3, Vp4, Vq1, Vq2, Vq3, Vq4, kVp12, kVp23, kVp34, kVq12, kVq23, kVq34, pfaref, s0_y, vp, wg, zp_z1, __zeros, __ones, __falses, __trues):
    return ((1 - zp_z1)*tan(pfaref), 1 - Volt_dip, -1, -Kf, SWP_s0 + SWP_s1*wg, select([greater_equal(Vq1, s0_y),greater_equal(Vq2, s0_y),greater_equal(Vq3, s0_y),greater_equal(Vq4, s0_y),__trues], [__zeros,kVq12,kVq23,kVq34,__zeros], default=nan), select([greater_equal(Vp1, s0_y),greater_equal(Vp2, s0_y),greater_equal(Vp3, s0_y),greater_equal(Vp4, s0_y),__trues], [__zeros,kVp12,kVp23,kVp34,__zeros], default=nan), -Kvp*SWQ_s1*SWV_s0*(1 - Volt_dip), 1 - Volt_dip, SWQ_s0, vp**(-1.0))


def s0_y_ia(v):
    return v


def S1_y_ia(Pe):
    return Pe


def PIQ_xi_ia():
    return 0.0


def Qcpf_ia(q0):
    return q0


def Qref_ia(qref0):
    return -qref0


def PFsel_ia(Qcpf, Qref, SWPF_s0, SWPF_s1):
    return Qcpf*SWPF_s1 + Qref*SWPF_s0


def vp_ia(VLower_zi, VLower_zl, v):
    return VLower_zi*v + 0.01*VLower_zl


def s4_y_ia(PFsel, vp):
    return PFsel/vp


def wg_ia():
    return 1.00000000000000


def Pref_ia(p0, wg):
    return p0/wg


def pfilt_y_ia(Pref):
    return Pref


def Psel_ia(SWP_s0, SWP_s1, pfilt_y, wg):
    return SWP_s0*pfilt_y + SWP_s1*pfilt_y*wg


def s5_y_ia(Psel):
    return Psel


def PIV_xi_ia(Iqcmd0, SWQ_s1):
    return -Iqcmd0*SWQ_s1


def pfaref_ia(pfaref0):
    return pfaref0


def Qerr_ia(PFlim_zi, PFlim_zl, PFlim_zu, PFsel, QMax, QMin, Qe):
    return PFlim_zi*PFsel + PFlim_zl*QMin + PFlim_zu*QMax - Qe


def PIQ_ys_ia(Kqp, Qerr, SWV_s1):
    return Kqp*Qerr*SWV_s1


def PIQ_y_ia(PIQ_lim_zi, PIQ_lim_zl, PIQ_lim_zu, PIQ_ys, VMAX, VMIN):
    return PIQ_lim_zi*PIQ_ys + PIQ_lim_zl*VMIN + PIQ_lim_zu*VMAX


def Vsel_x_ia(PIQ_y, Qcpf, Qref, SWPF_s0, SWPF_s1, SWV_s0, SWV_s1, Vref1):
    return PIQ_y*SWV_s1 + SWV_s0*(Qcpf*SWPF_s1 + Qref*SWPF_s0 + Vref1)


def Vsel_y_ia(VMAX, VMIN, Vsel_lim_zi, Vsel_lim_zl, Vsel_lim_zu, Vsel_x):
    return VMAX*Vsel_lim_zu + VMIN*Vsel_lim_zl + Vsel_lim_zi*Vsel_x


def Verr_ia(Vref0, s0_y):
    return Vref0 - s0_y


def dbV_y_ia(Verr, dbV_db_zl, dbV_db_zu, dbd1, dbd2):
    return 1.0*dbV_db_zl*(Verr - dbd1) + 1.0*dbV_db_zu*(Verr - dbd2)


def Iqinj_ia(Iqfrz, Kqv, Volt_dip, dbV_y, fThld, nThld, pThld):
    return Kqv*Volt_dip*dbV_y + fThld*(1 - Volt_dip)*(Iqfrz*pThld + Kqv*dbV_y*nThld)


def VDL1_y_ia(Iq1, Iq2, Iq3, Iq4, Vq1, Vq2, Vq3, Vq4, kVq12, kVq23, kVq34, s0_y, __zeros, __ones, __falses, __trues):
    return select([less_equal(s0_y, Vq1),less_equal(s0_y, Vq2),less_equal(s0_y, Vq3),less_equal(s0_y, Vq4),greater(s0_y, Vq4),__trues], [Iq1,Iq1 + kVq12*(-Vq1 + s0_y),Iq2 + kVq23*(-Vq2 + s0_y),Iq3 + kVq34*(-Vq3 + s0_y),Iq4,__zeros], default=nan)


def VDL2_y_ia(Ip1, Ip2, Ip3, Ip4, Vp1, Vp2, Vp3, Vp4, kVp12, kVp23, kVp34, s0_y, __zeros, __ones, __falses, __trues):
    return select([less_equal(s0_y, Vp1),less_equal(s0_y, Vp2),less_equal(s0_y, Vp3),less_equal(s0_y, Vp4),greater(s0_y, Vp4),__trues], [Ip1,Ip1 + kVp12*(-Vp1 + s0_y),Ip2 + kVp23*(-Vp2 + s0_y),Ip3 + kVp34*(-Vp3 + s0_y),Ip4,__zeros], default=nan)


def Ipmax_ia(Imaxr, Ipmax2sq0, SWPQ_s0, SWPQ_s1, VDL2_y, VDL2c, fThld2, zVDL2):
    return (1 - fThld2)*(sqrt(Ipmax2sq0)*SWPQ_s0 + SWPQ_s1*(zVDL2*(Imaxr*(1 - VDL2c) + VDL2_y*VDL2c) - 100000000.0*zVDL2 + 100000000.0))


def Iqmax_ia(Imaxr, Iqmax2sq0, SWPQ_s0, SWPQ_s1, VDL1_y, VDL1c, zVDL1):
    return sqrt(Iqmax2sq0)*SWPQ_s1 + SWPQ_s0*(zVDL1*(Imaxr*(1 - VDL1c) + VDL1_y*VDL1c) - 100000000.0*zVDL1 + 100000000.0)


def PIV_ys_ia(Iqcmd0, Kvp, SWQ_s1, SWV_s0, Vsel_y, s0_y):
    return -Iqcmd0*SWQ_s1 + Kvp*SWQ_s1*(-SWV_s0*s0_y + Vsel_y)


def PIV_y_ia(Iqmax, Iqmin, PIV_lim_zi, PIV_lim_zl, PIV_lim_zu, PIV_ys):
    return Iqmax*PIV_lim_zu + Iqmin*PIV_lim_zl + PIV_lim_zi*PIV_ys


def Qsel_ia(PIV_y, SWQ_s0, SWQ_s1, s4_y):
    return PIV_y*SWQ_s1 + SWQ_s0*s4_y


def IpHL_x_ia(s5_y, vp):
    return s5_y/vp


def IpHL_y_ia(IpHL_lim_zi, IpHL_lim_zl, IpHL_lim_zu, IpHL_x, Ipmax, Ipmin):
    return IpHL_lim_zi*IpHL_x + IpHL_lim_zl*Ipmin + IpHL_lim_zu*Ipmax


def IqHL_x_ia(Iqinj, Qsel):
    return Iqinj + Qsel


def IqHL_y_ia(IqHL_lim_zi, IqHL_lim_zl, IqHL_lim_zu, IqHL_x, Iqmax, Iqmin):
    return IqHL_lim_zi*IqHL_x + IqHL_lim_zl*Iqmin + IqHL_lim_zu*Iqmax


def Ipcmd0_svc(p0, v):
    return p0/v


def Iqcmd0_svc(q0, v):
    return -q0/v


def pfaref0_svc(p0, q0):
    return arctan2(q0, p0)


def Volt_dip_svc(Vcmp_zi):
    return 1 - Vcmp_zi


def qref0_svc(Iqcmd0, SWQ_s0, SWQ_s1, VLower_zi, VLower_zl, Vref1, v):
    return Iqcmd0*SWQ_s0*(VLower_zi*v + 0.01*VLower_zl) + SWQ_s1*(-Vref1 + v)


def PIQ_flag_svc():
    return 0


def s4_flag_svc():
    return 0


def pThld_svc(Thld):
    return (greater(Thld, 0))


def nThld_svc(Thld):
    return (less(Thld, 0))


def Thld_abs_svc(Thld):
    return abs(Thld)


def fThld_svc():
    return 0


def s5_flag_svc():
    return 0


def kVq12_svc(Iq1, Iq2, Vq1, Vq2):
    return (-Iq1 + Iq2)/(-Vq1 + Vq2)


def kVq23_svc(Iq2, Iq3, Vq2, Vq3):
    return (-Iq2 + Iq3)/(-Vq2 + Vq3)


def kVq34_svc(Iq3, Iq4, Vq3, Vq4):
    return (-Iq3 + Iq4)/(-Vq3 + Vq4)


def zVDL1_svc(Iq1, Iq2, Iq3, Iq4, Vq1, Vq2, Vq3, Vq4):
    return logical_and.reduce((less_equal(Iq1, Iq2),less_equal(Iq2, Iq3),less_equal(Iq3, Iq4),less_equal(Vq1, Vq2),less_equal(Vq2, Vq3),less_equal(Vq3, Vq4)))


def kVp12_svc(Ip1, Ip2, Vp1, Vp2):
    return (-Ip1 + Ip2)/(-Vp1 + Vp2)


def kVp23_svc(Ip2, Ip3, Vp2, Vp3):
    return (-Ip2 + Ip3)/(-Vp2 + Vp3)


def kVp34_svc(Ip3, Ip4, Vp3, Vp4):
    return (-Ip3 + Ip4)/(-Vp3 + Vp4)


def zVDL2_svc(Ip1, Ip2, Ip3, Ip4, Vp1, Vp2, Vp3, Vp4):
    return logical_and.reduce((less_equal(Ip1, Ip2),less_equal(Ip2, Ip3),less_equal(Ip3, Ip4),less_equal(Vp1, Vp2),less_equal(Vp2, Vp3),less_equal(Vp3, Vp4)))


def fThld2_svc():
    return 0


def VDL1c_svc(Imaxr, VDL1_y):
    return less(VDL1_y, Imaxr)


def VDL2c_svc(Imaxr, VDL2_y):
    return less(VDL2_y, Imaxr)


def Ipmax2sq0_svc(Imax, Iqcmd0, __zeros, __ones, __falses, __trues):
    return select([less_equal(Imax**2 - Iqcmd0**2, 0.0),__trues], [__zeros,Imax**2 - Iqcmd0**2], default=nan)


def Ipmax2sq_svc(Imax, IqHL_y, __zeros, __ones, __falses, __trues):
    return select([less_equal(Imax**2 - IqHL_y**2, 0.0),__trues], [__zeros,Imax**2 - IqHL_y**2], default=nan)


def Ipmaxh_svc():
    return 0


def Iqmax2sq0_svc(Imax, Ipcmd0, __zeros, __ones, __falses, __trues):
    return select([less_equal(Imax**2 - Ipcmd0**2, 0.0),__trues], [__zeros,Imax**2 - Ipcmd0**2], default=nan)


def Iqmax2sq_svc(Imax, IpHL_y, __zeros, __ones, __falses, __trues):
    return select([less_equal(Imax**2 - IpHL_y**2, 0.0),__trues], [__zeros,Imax**2 - IpHL_y**2], default=nan)


def Ipmin_svc():
    return 0.0


def PIV_flag_svc():
    return 0


# empty sns_update

f_args = ['Kqi',
 'Kvi',
 'PFsel',
 'PIQ_y',
 'PIQ_ys',
 'PIV_y',
 'PIV_ys',
 'Pe',
 'Pref',
 'Psel',
 'Qerr',
 'S1_y',
 'SWQ_s1',
 'SWV_s0',
 'SWV_s1',
 'Volt_dip',
 'Vsel_y',
 'pfilt_y',
 's0_y',
 's4_y',
 's5_y',
 'v',
 'vp']

g_args = ['Imaxr',
 'Ip1',
 'Ip2',
 'Ip3',
 'Ip4',
 'IpHL_lim_zi',
 'IpHL_lim_zl',
 'IpHL_lim_zu',
 'IpHL_x',
 'IpHL_y',
 'Ipcmd0',
 'Ipmax',
 'Ipmax2sq',
 'Ipmaxh',
 'Ipmin',
 'Iq1',
 'Iq2',
 'Iq3',
 'Iq4',
 'IqHL_lim_zi',
 'IqHL_lim_zl',
 'IqHL_lim_zu',
 'IqHL_x',
 'IqHL_y',
 'Iqcmd0',
 'Iqfrz',
 'Iqinj',
 'Iqmax',
 'Iqmax2sq',
 'Iqmin',
 'Kf',
 'Kqp',
 'Kqv',
 'Kvp',
 'PFlim_zi',
 'PFlim_zl',
 'PFlim_zu',
 'PFsel',
 'PIQ_lim_zi',
 'PIQ_lim_zl',
 'PIQ_lim_zu',
 'PIQ_xi',
 'PIQ_y',
 'PIQ_ys',
 'PIV_lim_zi',
 'PIV_lim_zl',
 'PIV_lim_zu',
 'PIV_xi',
 'PIV_y',
 'PIV_ys',
 'Pref',
 'Psel',
 'QMax',
 'QMin',
 'Qcpf',
 'Qe',
 'Qerr',
 'Qref',
 'Qsel',
 'S1_y',
 'SWPF_s0',
 'SWPF_s1',
 'SWPQ_s0',
 'SWPQ_s1',
 'SWP_s0',
 'SWP_s1',
 'SWQ_s0',
 'SWQ_s1',
 'SWV_s0',
 'SWV_s1',
 'VDL1_y',
 'VDL1c',
 'VDL2_y',
 'VDL2c',
 'VLower_zi',
 'VLower_zl',
 'VMAX',
 'VMIN',
 'Verr',
 'Volt_dip',
 'Vp1',
 'Vp2',
 'Vp3',
 'Vp4',
 'Vq1',
 'Vq2',
 'Vq3',
 'Vq4',
 'Vref0',
 'Vref1',
 'Vsel_lim_zi',
 'Vsel_lim_zl',
 'Vsel_lim_zu',
 'Vsel_x',
 'Vsel_y',
 'dbV_db_zl',
 'dbV_db_zu',
 'dbV_y',
 'dbd1',
 'dbd2',
 'fThld',
 'fThld2',
 'kVp12',
 'kVp23',
 'kVp34',
 'kVq12',
 'kVq23',
 'kVq34',
 'nThld',
 'omega',
 'p0',
 'pThld',
 'pfaref',
 'pfaref0',
 'pfilt_y',
 'qref0',
 's0_y',
 's4_y',
 's5_y',
 'v',
 'vp',
 'wg',
 'zVDL1',
 'zVDL2',
 'zp_z1',
 '__zeros',
 '__ones',
 '__falses',
 '__trues']

j_args = {'fx': ['Kvi', 'SWQ_s1', 'SWV_s0', 'Volt_dip'],
 'fy': ['Kqi', 'Kvi', 'PFsel', 'SWQ_s1', 'SWV_s1', 'Volt_dip', 'vp'],
 'gx': ['Kf',
        'Kvp',
        'SWP_s0',
        'SWP_s1',
        'SWQ_s0',
        'SWQ_s1',
        'SWV_s0',
        'Volt_dip',
        'Vp1',
        'Vp2',
        'Vp3',
        'Vp4',
        'Vq1',
        'Vq2',
        'Vq3',
        'Vq4',
        'kVp12',
        'kVp23',
        'kVp34',
        'kVq12',
        'kVq23',
        'kVq34',
        'pfaref',
        's0_y',
        'vp',
        'wg',
        'zp_z1',
        '__zeros',
        '__ones',
        '__falses',
        '__trues'],
 'gy': ['IpHL_lim_zi',
        'IpHL_lim_zu',
        'IqHL_lim_zi',
        'IqHL_lim_zu',
        'Kqp',
        'Kqv',
        'Kvp',
        'PFlim_zi',
        'PIQ_lim_zi',
        'PIV_lim_zi',
        'PIV_lim_zu',
        'S1_y',
        'SWPF_s0',
        'SWPF_s1',
        'SWPQ_s0',
        'SWPQ_s1',
        'SWP_s1',
        'SWQ_s1',
        'SWV_s0',
        'SWV_s1',
        'VDL1c',
        'VDL2c',
        'VLower_zi',
        'Volt_dip',
        'Vsel_lim_zi',
        'dbV_db_zl',
        'dbV_db_zu',
        'fThld',
        'fThld2',
        'nThld',
        'p0',
        'pfaref',
        'pfilt_y',
        's5_y',
        'vp',
        'wg',
        'zVDL1',
        'zVDL2',
        'zp_z1']}

s_args = OrderedDict([('Ipcmd0', ['p0', 'v']),
             ('Iqcmd0', ['q0', 'v']),
             ('pfaref0', ['p0', 'q0']),
             ('Volt_dip', ['Vcmp_zi']),
             ('qref0',
              ['Iqcmd0',
               'SWQ_s0',
               'SWQ_s1',
               'VLower_zi',
               'VLower_zl',
               'Vref1',
               'v']),
             ('PIQ_flag', []),
             ('s4_flag', []),
             ('pThld', ['Thld']),
             ('nThld', ['Thld']),
             ('Thld_abs', ['Thld']),
             ('fThld', []),
             ('s5_flag', []),
             ('kVq12', ['Iq1', 'Iq2', 'Vq1', 'Vq2']),
             ('kVq23', ['Iq2', 'Iq3', 'Vq2', 'Vq3']),
             ('kVq34', ['Iq3', 'Iq4', 'Vq3', 'Vq4']),
             ('zVDL1',
              ['Iq1', 'Iq2', 'Iq3', 'Iq4', 'Vq1', 'Vq2', 'Vq3', 'Vq4']),
             ('kVp12', ['Ip1', 'Ip2', 'Vp1', 'Vp2']),
             ('kVp23', ['Ip2', 'Ip3', 'Vp2', 'Vp3']),
             ('kVp34', ['Ip3', 'Ip4', 'Vp3', 'Vp4']),
             ('zVDL2',
              ['Ip1', 'Ip2', 'Ip3', 'Ip4', 'Vp1', 'Vp2', 'Vp3', 'Vp4']),
             ('fThld2', []),
             ('VDL1c', ['Imaxr', 'VDL1_y']),
             ('VDL2c', ['Imaxr', 'VDL2_y']),
             ('Ipmax2sq0',
              ['Imax', 'Iqcmd0', '__zeros', '__ones', '__falses', '__trues']),
             ('Ipmax2sq',
              ['Imax', 'IqHL_y', '__zeros', '__ones', '__falses', '__trues']),
             ('Ipmaxh', []),
             ('Iqmax2sq0',
              ['Imax', 'Ipcmd0', '__zeros', '__ones', '__falses', '__trues']),
             ('Iqmax2sq',
              ['Imax', 'IpHL_y', '__zeros', '__ones', '__falses', '__trues']),
             ('Ipmin', []),
             ('PIV_flag', [])])

sns_args = []

ia_args = OrderedDict([('s0_y', ['v']),
             ('S1_y', ['Pe']),
             ('PIQ_xi', []),
             ('Qcpf', ['q0']),
             ('Qref', ['qref0']),
             ('PFsel', ['Qcpf', 'Qref', 'SWPF_s0', 'SWPF_s1']),
             ('vp', ['VLower_zi', 'VLower_zl', 'v']),
             ('s4_y', ['PFsel', 'vp']),
             ('wg', []),
             ('Pref', ['p0', 'wg']),
             ('pfilt_y', ['Pref']),
             ('Psel', ['SWP_s0', 'SWP_s1', 'pfilt_y', 'wg']),
             ('s5_y', ['Psel']),
             ('PIV_xi', ['Iqcmd0', 'SWQ_s1']),
             ('pfaref', ['pfaref0']),
             ('Qerr',
              ['PFlim_zi',
               'PFlim_zl',
               'PFlim_zu',
               'PFsel',
               'QMax',
               'QMin',
               'Qe']),
             ('PIQ_ys', ['Kqp', 'Qerr', 'SWV_s1']),
             ('PIQ_y',
              ['PIQ_lim_zi',
               'PIQ_lim_zl',
               'PIQ_lim_zu',
               'PIQ_ys',
               'VMAX',
               'VMIN']),
             ('Vsel_x',
              ['PIQ_y',
               'Qcpf',
               'Qref',
               'SWPF_s0',
               'SWPF_s1',
               'SWV_s0',
               'SWV_s1',
               'Vref1']),
             ('Vsel_y',
              ['VMAX',
               'VMIN',
               'Vsel_lim_zi',
               'Vsel_lim_zl',
               'Vsel_lim_zu',
               'Vsel_x']),
             ('Verr', ['Vref0', 's0_y']),
             ('dbV_y', ['Verr', 'dbV_db_zl', 'dbV_db_zu', 'dbd1', 'dbd2']),
             ('Iqinj',
              ['Iqfrz', 'Kqv', 'Volt_dip', 'dbV_y', 'fThld', 'nThld', 'pThld']),
             ('VDL1_y',
              ['Iq1',
               'Iq2',
               'Iq3',
               'Iq4',
               'Vq1',
               'Vq2',
               'Vq3',
               'Vq4',
               'kVq12',
               'kVq23',
               'kVq34',
               's0_y',
               '__zeros',
               '__ones',
               '__falses',
               '__trues']),
             ('VDL2_y',
              ['Ip1',
               'Ip2',
               'Ip3',
               'Ip4',
               'Vp1',
               'Vp2',
               'Vp3',
               'Vp4',
               'kVp12',
               'kVp23',
               'kVp34',
               's0_y',
               '__zeros',
               '__ones',
               '__falses',
               '__trues']),
             ('Ipmax',
              ['Imaxr',
               'Ipmax2sq0',
               'SWPQ_s0',
               'SWPQ_s1',
               'VDL2_y',
               'VDL2c',
               'fThld2',
               'zVDL2']),
             ('Iqmax',
              ['Imaxr',
               'Iqmax2sq0',
               'SWPQ_s0',
               'SWPQ_s1',
               'VDL1_y',
               'VDL1c',
               'zVDL1']),
             ('PIV_ys',
              ['Iqcmd0', 'Kvp', 'SWQ_s1', 'SWV_s0', 'Vsel_y', 's0_y']),
             ('PIV_y',
              ['Iqmax',
               'Iqmin',
               'PIV_lim_zi',
               'PIV_lim_zl',
               'PIV_lim_zu',
               'PIV_ys']),
             ('Qsel', ['PIV_y', 'SWQ_s0', 'SWQ_s1', 's4_y']),
             ('IpHL_x', ['s5_y', 'vp']),
             ('IpHL_y',
              ['IpHL_lim_zi',
               'IpHL_lim_zl',
               'IpHL_lim_zu',
               'IpHL_x',
               'Ipmax',
               'Ipmin']),
             ('IqHL_x', ['Iqinj', 'Qsel']),
             ('IqHL_y',
              ['IqHL_lim_zi',
               'IqHL_lim_zl',
               'IqHL_lim_zu',
               'IqHL_x',
               'Iqmax',
               'Iqmin'])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', [3, 5]),
             ('fx', [0, 1, 3, 4, 5, 6]),
             ('fyc', []),
             ('fy', [0, 1, 2, 2, 2, 3, 3, 4, 5, 6, 6, 6]),
             ('gxc', []),
             ('gx', [3, 6, 10, 14, 15, 16, 17, 20, 20, 22, 23]),
             ('gyc', [3, 6, 7, 18, 20, 21]),
             ('gy',
              [0,
               0,
               1,
               2,
               3,
               3,
               4,
               4,
               4,
               5,
               5,
               5,
               6,
               6,
               7,
               7,
               8,
               8,
               8,
               8,
               9,
               9,
               10,
               11,
               11,
               12,
               12,
               13,
               14,
               14,
               15,
               15,
               16,
               17,
               18,
               18,
               19,
               19,
               20,
               20,
               21,
               21,
               21,
               22,
               22,
               23,
               23,
               24,
               24,
               24,
               25,
               25,
               25,
               26,
               26,
               26,
               31,
               32])])

jjac = OrderedDict([('fxc', [3, 5]),
             ('fx', [0, 1, 3, 4, 5, 0]),
             ('fyc', []),
             ('fy', [37, 38, 14, 15, 16, 9, 13, 23, 24, 18, 29, 30]),
             ('gxc', []),
             ('gx', [1, 2, 0, 8, 4, 0, 0, 0, 6, 3, 5]),
             ('gyc', [12, 15, 16, 27, 29, 30]),
             ('gy',
              [9,
               37,
               10,
               11,
               10,
               12,
               11,
               12,
               13,
               13,
               14,
               39,
               14,
               15,
               15,
               16,
               11,
               12,
               16,
               17,
               17,
               18,
               19,
               19,
               20,
               20,
               21,
               22,
               22,
               23,
               22,
               24,
               25,
               26,
               26,
               27,
               25,
               28,
               18,
               29,
               28,
               29,
               30,
               30,
               31,
               9,
               32,
               27,
               32,
               33,
               21,
               31,
               34,
               28,
               34,
               35,
               33,
               35])])

vjac = OrderedDict([('fxc', [1e-08, 1e-08]),
             ('fx', [0, 0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08]),
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
               0,
               0,
               0])])

j_names = ['fx', 'fy', 'gy', 'gx']

init_seq = ['v',
 's0_y',
 'Pe',
 'S1_y',
 'PIQ_xi',
 'Qcpf',
 'Qref',
 'PFsel',
 'vp',
 's4_y',
 'wg',
 'Pref',
 'pfilt_y',
 'Psel',
 's5_y',
 'PIV_xi',
 'Pord',
 'omega',
 'pfaref',
 'Qe',
 'Qerr',
 'PIQ_ys',
 'PIQ_y',
 'Vsel_x',
 'Vsel_y',
 'Verr',
 'dbV_y',
 'Iqinj',
 'VDL1_y',
 'VDL2_y',
 'Ipmax',
 'Iqmax',
 'PIV_ys',
 'PIV_y',
 'Qsel',
 'IpHL_x',
 'IpHL_y',
 'IqHL_x',
 'IqHL_y',
 'a',
 'Ipcmd',
 'Iqcmd']

need_diag_eps = ['s4_y',
 's5_y',
 'Qcpf',
 'PIQ_ys',
 'PIQ_y',
 'Ipmax',
 'PIV_ys',
 'PIV_y',
 's4_y',
 's5_y',
 'Qcpf',
 'PIQ_ys',
 'PIQ_y',
 'Ipmax',
 'PIV_ys',
 'PIV_y']
