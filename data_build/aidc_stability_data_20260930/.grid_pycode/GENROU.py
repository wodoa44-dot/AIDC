from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "796d01b718d3cf4d2a916c6a04abd7d3"

def f_update(D, Id, Iq, XadIfd, XaqI1q, e1d, e1q, e2d, e2q, fn, omega, te, tm, u, vf, xd1, xl, xq1):
    return (2*pi*fn*u*(omega - 1), u*(-D*(omega - 1) - te + tm), -XadIfd + vf, -XaqI1q, -Id*(xd1 - xl) + e1q - e2d, Iq*(-xl + xq1) + e1d - e2q,)


def g_update(Id, Iq, Pe, Qe, SAT_A, SAT_B, SL_z0, Se, XadIfd, XaqI1q, a, delta, e1d, e1q, e2d, e2q, gd1, gd2, gq1, gq2, gqd, psi2, psi2d, psi2q, psid, psiq, ra, te, tm, tm0, u, v, vd, vf, vf0, vq, xd, xd1, xd2, xl, xq, xq1, xq2):
    return (Id*xd2 - psi2d + psid, Iq*xq2 + psi2q + psiq, -u*v*sin(a - delta) - vd, u*v*cos(a - delta) - vq, -tm + tm0, -te + u*(-Id*psiq + Iq*psid), u*vf0 - vf, -XadIfd + u*(Se*psi2d + e1q + (xd - xd1)*(Id*gd1 + e1q*gd2 - e2d*gd2)), -Pe + u*(Id*vd + Iq*vq), -Qe + u*(Id*vq - Iq*vd), -psid + u*(Iq*ra + vq), psiq + u*(Id*ra + vd), e1d*gq1 + e2q*(1 - gq1) - psi2q, e1q*gd1 + e2d*gd2*(xd1 - xl) - psi2d, -psi2**2 + psi2d**2 + psi2q**2, SAT_B*SL_z0*(-SAT_A + psi2)**2 - Se*psi2, Se*gqd*psi2q - XaqI1q + e1d + (xq - xq1)*(-Iq*gq1 + e1d*gq2 - e2q*gq2), -u*(Id*vd + Iq*vq), -u*(Id*vq - Iq*vd),)


def fx_update(D, fn, u):
    return (2*pi*fn*u, -D*u, 1, -1, 1, -1)


def fy_update(u, xd1, xl, xq1):
    return (u, -u, 1, -1, -1, -xd1 + xl, -xl + xq1)


def gy_update(Id, Iq, SAT_A, SAT_B, SL_z0, Se, a, delta, gd1, gq1, gqd, psi2, psi2d, psi2q, psid, psiq, ra, u, v, vd, vq, xd, xd1, xd2, xq, xq1, xq2):
    return (xd2, 1, -1, xq2, 1, 1, -1, -u*v*cos(a - delta), -u*sin(a - delta), -1, -u*v*sin(a - delta), u*cos(a - delta), -1, -psiq*u, psid*u, -1, Iq*u, -Id*u, -1, gd1*u*(xd - xd1), -1, Se*u, psi2d*u, u*vd, u*vq, Id*u, Iq*u, -1, u*vq, -u*vd, -Iq*u, Id*u, -1, ra*u, u, -1, ra*u, u, 1, -1, -1, 2*psi2q, 2*psi2d, -2*psi2, SAT_B*SL_z0*(-2*SAT_A + 2*psi2) - Se, -psi2, -gq1*(xq - xq1), Se*gqd, gqd*psi2q, -1, -u*vd, -u*vq, -Id*u, -Iq*u, -u*vq, u*vd, Iq*u, -Id*u)


def gx_update(a, delta, gd1, gd2, gq1, gq2, u, v, xd, xd1, xl, xq, xq1):
    return (u*v*cos(a - delta), u*v*sin(a - delta), u*(gd2*(xd - xd1) + 1), -gd2*u*(xd - xd1), gq1, 1 - gq1, gd1, gd2*(xd1 - xl), gq2*(xq - xq1) + 1, -gq2*(xq - xq1))


def delta_ia(delta0):
    return delta0


def omega_ia(u):
    return u


def e1q_ia(e1q0, u):
    return e1q0*u


def e1d_ia(e1d0, u):
    return e1d0*u


def e2d_ia(e2d0, u):
    return e2d0*u


def e2q_ia(e2q0, u):
    return e2q0*u


def Id_ia(Id0, u):
    return Id0*u


def Iq_ia(Iq0, u):
    return Iq0*u


def vd_ia(u, vd0):
    return u*vd0


def vq_ia(u, vq0):
    return u*vq0


def tm_ia(tm0):
    return tm0


def te_ia(tm0, u):
    return tm0*u


def vf_ia(u, vf0):
    return u*vf0


def XadIfd_ia(u, vf0):
    return u*vf0


def Pe_ia(Id0, Iq0, u, vd0, vq0):
    return u*(Id0*vd0 + Iq0*vq0)


def Qe_ia(Id0, Iq0, u, vd0, vq0):
    return u*(Id0*vq0 - Iq0*vd0)


def psid_ia(psid0, u):
    return psid0*u


def psiq_ia(psiq0, u):
    return psiq0*u


def psi2q_ia(psi2q0, u):
    return psi2q0*u


def psi2d_ia(psi2d0, u):
    return psi2d0*u


def psi2_ia(psi20_dq, u):
    return u*abs(psi20_dq)


def Se_ia(Se0, u):
    return Se0*u


def XaqI1q_ia():
    return 0


def p0_svc(gammap, p0s):
    return gammap*p0s


def q0_svc(gammaq, q0s):
    return gammaq*q0s


def gd1_svc(xd1, xd2, xl):
    return (xd2 - xl)/(xd1 - xl)


def gq1_svc(xl, xq1, xq2):
    return (-xl + xq2)/(-xl + xq1)


def gd2_svc(xd1, xd2, xl):
    return (xd1 - xd2)/(xd1 - xl)**2


def gq2_svc(xl, xq1, xq2):
    return (xq1 - xq2)/(-xl + xq1)**2


def gqd_svc(xd, xl, xq):
    return (-xl + xq)/(xd - xl)


def _S12_svc(S12, _fS12):
    return S12 - _fS12 + 1


def SAT_E1_svc():
    return 1.00000000000000


def SAT_E2_svc(SAT_zSE2):
    return 3.2 - 2*SAT_zSE2


def SAT_SE1_svc(S10):
    return S10


def SAT_SE2_svc(S12, SAT_zSE2):
    return S12 - 2*SAT_zSE2 + 2


def SAT_a_svc(SAT_E1, SAT_E2, SAT_SE1, SAT_SE2):
    return sqrt(SAT_E1*SAT_SE1/(SAT_E2*SAT_SE2))*((greater(SAT_SE2, 0)) + (less(SAT_SE2, 0)))


def SAT_A_svc(SAT_E1, SAT_E2, SAT_a):
    return SAT_E2 - (SAT_E1 - SAT_E2)/(SAT_a - 1)


def SAT_B_svc(SAT_E1, SAT_E2, SAT_SE2, SAT_a):
    return SAT_E2*SAT_SE2*(SAT_a - 1)**2*((greater(SAT_a, 0)) + (less(SAT_a, 0)))/(SAT_E1 - SAT_E2)**2


def _V_svc(a, v):
    return v*exp(1j*a)


def _S_svc(p0, q0):
    return p0 - 1j*q0


def _Zs_svc(ra, xd2):
    return ra + 1j*xd2


def _It_svc(_S, _V):
    return _S/conj(_V)


def _Is_svc(_It, _V, _Zs):
    return _It + _V/_Zs


def psi20_svc(_Is, _Zs):
    return _Is*_Zs


def psi20_arg_svc(psi20):
    return angle(psi20)


def psi20_abs_svc(psi20):
    return abs(psi20)


def _It_arg_svc(_It):
    return angle(_It)


def _psi20_It_arg_svc(_It_arg, psi20_arg):
    return -_It_arg + psi20_arg


def Se0_svc(SAT_A, SAT_B, psi20_abs):
    return SAT_B*(-SAT_A + psi20_abs)**2*(greater_equal(psi20_abs, SAT_A))/psi20_abs


def _a_svc(Se0, gqd, psi20_abs):
    return psi20_abs*(Se0*gqd + 1)


def _b_svc(_It, xq, xq2):
    return (-xq + xq2)*abs(_It)


def delta0_svc(_a, _b, _psi20_It_arg, psi20_arg):
    return psi20_arg + arctan(_b*cos(_psi20_It_arg)/(-_a + _b*sin(_psi20_It_arg)))


def _Tdq_svc(delta0):
    return -1j*sin(delta0) + cos(delta0)


def psi20_dq_svc(_Tdq, psi20):
    return _Tdq*psi20


def It_dq_svc(_It, _Tdq):
    return conj(_It*_Tdq)


def psi2d0_svc(psi20_dq):
    return real(psi20_dq)


def psi2q0_svc(psi20_dq):
    return -imag(psi20_dq)


def Id0_svc(It_dq):
    return imag(It_dq)


def Iq0_svc(It_dq):
    return real(It_dq)


def vd0_svc(Id0, Iq0, psi2q0, ra, xq2):
    return -Id0*ra + Iq0*xq2 + psi2q0


def vq0_svc(Id0, Iq0, psi2d0, ra, xd2):
    return -Id0*xd2 - Iq0*ra + psi2d0


def tm0_svc(Id0, Iq0, ra, u, vd0, vq0):
    return u*(Id0*(Id0*ra + vd0) + Iq0*(Iq0*ra + vq0))


def vf0_svc(Id0, Se0, psi2d0, xd, xd2):
    return Id0*(xd - xd2) + psi2d0*(Se0 + 1)


def psid0_svc(Iq0, ra, u, vq0):
    return Iq0*ra*u + vq0


def psiq0_svc(Id0, ra, u, vd0):
    return -Id0*ra*u - vd0


def e1q0_svc(Id0, Se0, psi2d0, vf0, xd, xd1):
    return Id0*(-xd + xd1) - Se0*psi2d0 + vf0


def e1d0_svc(Iq0, Se0, gqd, psi2q0, xq, xq1):
    return Iq0*(xq - xq1) - Se0*gqd*psi2q0


def e2d0_svc(Id0, Se0, psi2d0, vf0, xd, xl):
    return Id0*(-xd + xl) - Se0*psi2d0 + vf0


def e2q0_svc(Iq0, Se0, gqd, psi2q0, xl, xq):
    return -Iq0*(xl - xq) - Se0*gqd*psi2q0


# empty sns_update

f_args = ['D',
 'Id',
 'Iq',
 'XadIfd',
 'XaqI1q',
 'e1d',
 'e1q',
 'e2d',
 'e2q',
 'fn',
 'omega',
 'te',
 'tm',
 'u',
 'vf',
 'xd1',
 'xl',
 'xq1']

g_args = ['Id',
 'Iq',
 'Pe',
 'Qe',
 'SAT_A',
 'SAT_B',
 'SL_z0',
 'Se',
 'XadIfd',
 'XaqI1q',
 'a',
 'delta',
 'e1d',
 'e1q',
 'e2d',
 'e2q',
 'gd1',
 'gd2',
 'gq1',
 'gq2',
 'gqd',
 'psi2',
 'psi2d',
 'psi2q',
 'psid',
 'psiq',
 'ra',
 'te',
 'tm',
 'tm0',
 'u',
 'v',
 'vd',
 'vf',
 'vf0',
 'vq',
 'xd',
 'xd1',
 'xd2',
 'xl',
 'xq',
 'xq1',
 'xq2']

j_args = {'fx': ['D', 'fn', 'u'],
 'fy': ['u', 'xd1', 'xl', 'xq1'],
 'gx': ['a',
        'delta',
        'gd1',
        'gd2',
        'gq1',
        'gq2',
        'u',
        'v',
        'xd',
        'xd1',
        'xl',
        'xq',
        'xq1'],
 'gy': ['Id',
        'Iq',
        'SAT_A',
        'SAT_B',
        'SL_z0',
        'Se',
        'a',
        'delta',
        'gd1',
        'gq1',
        'gqd',
        'psi2',
        'psi2d',
        'psi2q',
        'psid',
        'psiq',
        'ra',
        'u',
        'v',
        'vd',
        'vq',
        'xd',
        'xd1',
        'xd2',
        'xq',
        'xq1',
        'xq2']}

s_args = OrderedDict([('p0', ['gammap', 'p0s']),
             ('q0', ['gammaq', 'q0s']),
             ('gd1', ['xd1', 'xd2', 'xl']),
             ('gq1', ['xl', 'xq1', 'xq2']),
             ('gd2', ['xd1', 'xd2', 'xl']),
             ('gq2', ['xl', 'xq1', 'xq2']),
             ('gqd', ['xd', 'xl', 'xq']),
             ('_S12', ['S12', '_fS12']),
             ('SAT_E1', []),
             ('SAT_E2', ['SAT_zSE2']),
             ('SAT_SE1', ['S10']),
             ('SAT_SE2', ['S12', 'SAT_zSE2']),
             ('SAT_a', ['SAT_E1', 'SAT_E2', 'SAT_SE1', 'SAT_SE2']),
             ('SAT_A', ['SAT_E1', 'SAT_E2', 'SAT_a']),
             ('SAT_B', ['SAT_E1', 'SAT_E2', 'SAT_SE2', 'SAT_a']),
             ('_V', ['a', 'v']),
             ('_S', ['p0', 'q0']),
             ('_Zs', ['ra', 'xd2']),
             ('_It', ['_S', '_V']),
             ('_Is', ['_It', '_V', '_Zs']),
             ('psi20', ['_Is', '_Zs']),
             ('psi20_arg', ['psi20']),
             ('psi20_abs', ['psi20']),
             ('_It_arg', ['_It']),
             ('_psi20_It_arg', ['_It_arg', 'psi20_arg']),
             ('Se0', ['SAT_A', 'SAT_B', 'psi20_abs']),
             ('_a', ['Se0', 'gqd', 'psi20_abs']),
             ('_b', ['_It', 'xq', 'xq2']),
             ('delta0', ['_a', '_b', '_psi20_It_arg', 'psi20_arg']),
             ('_Tdq', ['delta0']),
             ('psi20_dq', ['_Tdq', 'psi20']),
             ('It_dq', ['_It', '_Tdq']),
             ('psi2d0', ['psi20_dq']),
             ('psi2q0', ['psi20_dq']),
             ('Id0', ['It_dq']),
             ('Iq0', ['It_dq']),
             ('vd0', ['Id0', 'Iq0', 'psi2q0', 'ra', 'xq2']),
             ('vq0', ['Id0', 'Iq0', 'psi2d0', 'ra', 'xd2']),
             ('tm0', ['Id0', 'Iq0', 'ra', 'u', 'vd0', 'vq0']),
             ('vf0', ['Id0', 'Se0', 'psi2d0', 'xd', 'xd2']),
             ('psid0', ['Iq0', 'ra', 'u', 'vq0']),
             ('psiq0', ['Id0', 'ra', 'u', 'vd0']),
             ('e1q0', ['Id0', 'Se0', 'psi2d0', 'vf0', 'xd', 'xd1']),
             ('e1d0', ['Iq0', 'Se0', 'gqd', 'psi2q0', 'xq', 'xq1']),
             ('e2d0', ['Id0', 'Se0', 'psi2d0', 'vf0', 'xd', 'xl']),
             ('e2q0', ['Iq0', 'Se0', 'gqd', 'psi2q0', 'xl', 'xq'])])

sns_args = []

ia_args = OrderedDict([('delta', ['delta0']),
             ('omega', ['u']),
             ('e1q', ['e1q0', 'u']),
             ('e1d', ['e1d0', 'u']),
             ('e2d', ['e2d0', 'u']),
             ('e2q', ['e2q0', 'u']),
             ('Id', ['Id0', 'u']),
             ('Iq', ['Iq0', 'u']),
             ('vd', ['u', 'vd0']),
             ('vq', ['u', 'vq0']),
             ('tm', ['tm0']),
             ('te', ['tm0', 'u']),
             ('vf', ['u', 'vf0']),
             ('XadIfd', ['u', 'vf0']),
             ('Pe', ['Id0', 'Iq0', 'u', 'vd0', 'vq0']),
             ('Qe', ['Id0', 'Iq0', 'u', 'vd0', 'vq0']),
             ('psid', ['psid0', 'u']),
             ('psiq', ['psiq0', 'u']),
             ('psi2q', ['psi2q0', 'u']),
             ('psi2d', ['psi2d0', 'u']),
             ('psi2', ['psi20_dq', 'u']),
             ('Se', ['Se0', 'u']),
             ('XaqI1q', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', [0, 1, 4, 4, 5, 5]),
             ('fyc', []),
             ('fy', [1, 1, 2, 2, 3, 4, 5]),
             ('gxc', []),
             ('gx', [2, 3, 7, 7, 12, 12, 13, 13, 16, 16]),
             ('gyc', [14, 15]),
             ('gy',
              [0,
               0,
               0,
               1,
               1,
               1,
               2,
               2,
               2,
               3,
               3,
               3,
               4,
               5,
               5,
               5,
               5,
               5,
               6,
               7,
               7,
               7,
               7,
               8,
               8,
               8,
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
               11,
               11,
               11,
               12,
               13,
               14,
               14,
               14,
               15,
               15,
               16,
               16,
               16,
               16,
               17,
               17,
               17,
               17,
               18,
               18,
               18,
               18])])

jjac = OrderedDict([('fxc', []),
             ('fx', [1, 1, 2, 4, 3, 5]),
             ('fyc', []),
             ('fy', [10, 11, 12, 13, 22, 6, 7]),
             ('gxc', []),
             ('gx', [0, 0, 2, 4, 3, 5, 2, 4, 3, 5]),
             ('gyc', [20, 21]),
             ('gy',
              [6,
               16,
               19,
               7,
               17,
               18,
               8,
               23,
               24,
               9,
               23,
               24,
               10,
               6,
               7,
               11,
               16,
               17,
               12,
               6,
               13,
               19,
               21,
               6,
               7,
               8,
               9,
               14,
               6,
               7,
               8,
               9,
               15,
               7,
               9,
               16,
               6,
               8,
               17,
               18,
               19,
               18,
               19,
               20,
               20,
               21,
               7,
               18,
               21,
               22,
               6,
               7,
               8,
               9,
               6,
               7,
               8,
               9])])

vjac = OrderedDict([('fxc', []),
             ('fx', [0, 0, 0, 0, 0, 0]),
             ('fyc', []),
             ('fy', [0, 0, 0, 0, 0, 0, 0]),
             ('gxc', []),
             ('gx', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
             ('gyc', [1e-08, 1e-08]),
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

init_seq = ['delta',
 'omega',
 'e1q',
 'e1d',
 'e2d',
 'e2q',
 'Id',
 'Iq',
 'vd',
 'vq',
 'tm',
 'te',
 'vf',
 'XadIfd',
 'Pe',
 'Qe',
 'psid',
 'psiq',
 'psi2q',
 'psi2d',
 'psi2',
 'Se',
 'XaqI1q',
 'a',
 'v']

need_diag_eps = ['psi2', 'Se', 'psi2', 'Se']
