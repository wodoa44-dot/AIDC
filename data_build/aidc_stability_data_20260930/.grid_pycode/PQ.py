from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "6608e6cd298172d6ed5ea4e9ad00afe7"

# empty f_update

def g_update(Ipeq, Iqeq, Ppf, Qpf, Req, Rlb, Rub, Xeq, Xlb, Xub, dae_t, p0, p2i, p2p, p2z, q0, q2i, q2q, q2z, u, v, vcmp_zi, vcmp_zl, vcmp_zu):
    return (u*(Ipeq*p2i*v + Ppf*p2p + Req*p2z*v**2)*(greater_equal(dae_t, 0)) + u*(Rlb*v**2*vcmp_zl + Rub*v**2*vcmp_zu + p0*vcmp_zi)*(less(dae_t, 0)), u*(Iqeq*q2i*v + Qpf*q2q + Xeq*q2z*v**2)*(greater_equal(dae_t, 0)) + u*(Xlb*v**2*vcmp_zl + Xub*v**2*vcmp_zu + q0*vcmp_zi)*(less(dae_t, 0)),)


def gy_update(Ipeq, Iqeq, Req, Rlb, Rub, Xeq, Xlb, Xub, dae_t, p2i, p2z, q2i, q2z, u, v, vcmp_zl, vcmp_zu):
    return (u*(Ipeq*p2i + 2*Req*p2z*v)*(greater_equal(dae_t, 0)) + u*(2*Rlb*v*vcmp_zl + 2*Rub*v*vcmp_zu)*(less(dae_t, 0)), u*(Iqeq*q2i + 2*Xeq*q2z*v)*(greater_equal(dae_t, 0)) + u*(2*Xlb*v*vcmp_zl + 2*Xub*v*vcmp_zu)*(less(dae_t, 0)))


def Rub_svc(p0, vmax):
    return p0/vmax**2


def Xub_svc(q0, vmax):
    return q0/vmax**2


def Rlb_svc(p0, vmin):
    return p0/vmin**2


def Xlb_svc(q0, vmin):
    return q0/vmin**2


def Ppf_svc(Rlb, Rub, p0, v0, vcmp_zi, vcmp_zl, vcmp_zu):
    return Rlb*v0**2*vcmp_zl + Rub*v0**2*vcmp_zu + p0*vcmp_zi


def Qpf_svc(Xlb, Xub, q0, v0, vcmp_zi, vcmp_zl, vcmp_zu):
    return Xlb*v0**2*vcmp_zl + Xub*v0**2*vcmp_zu + q0*vcmp_zi


def Req_svc(Ppf, v0):
    return Ppf/v0**2


def Xeq_svc(Qpf, v0):
    return Qpf/v0**2


def Ipeq_svc(Ppf, v0):
    return Ppf/v0


def Iqeq_svc(Qpf, v0):
    return Qpf/v0


# empty sns_update

f_args = []

g_args = ['Ipeq',
 'Iqeq',
 'Ppf',
 'Qpf',
 'Req',
 'Rlb',
 'Rub',
 'Xeq',
 'Xlb',
 'Xub',
 'dae_t',
 'p0',
 'p2i',
 'p2p',
 'p2z',
 'q0',
 'q2i',
 'q2q',
 'q2z',
 'u',
 'v',
 'vcmp_zi',
 'vcmp_zl',
 'vcmp_zu']

j_args = {'gy': ['Ipeq',
        'Iqeq',
        'Req',
        'Rlb',
        'Rub',
        'Xeq',
        'Xlb',
        'Xub',
        'dae_t',
        'p2i',
        'p2z',
        'q2i',
        'q2z',
        'u',
        'v',
        'vcmp_zl',
        'vcmp_zu']}

s_args = OrderedDict([('Rub', ['p0', 'vmax']),
             ('Xub', ['q0', 'vmax']),
             ('Rlb', ['p0', 'vmin']),
             ('Xlb', ['q0', 'vmin']),
             ('Ppf',
              ['Rlb', 'Rub', 'p0', 'v0', 'vcmp_zi', 'vcmp_zl', 'vcmp_zu']),
             ('Qpf',
              ['Xlb', 'Xub', 'q0', 'v0', 'vcmp_zi', 'vcmp_zl', 'vcmp_zu']),
             ('Req', ['Ppf', 'v0']),
             ('Xeq', ['Qpf', 'v0']),
             ('Ipeq', ['Ppf', 'v0']),
             ('Iqeq', ['Qpf', 'v0'])])

sns_args = []

ia_args = OrderedDict()

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 1])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [1, 1])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0])])

j_names = ['gy']

init_seq = ['a', 'v']

need_diag_eps = []
