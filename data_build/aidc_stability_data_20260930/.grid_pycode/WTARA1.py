from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "caad914aa8fc9aa8f22b7a31e23576f4"

# empty f_update

def g_update(Ka, theta, theta0, theta0r):
    return (-theta + theta0r, -Ka*theta*(theta - theta0),)


def gy_update(Ka, theta, theta0):
    return (-1, -Ka*theta - Ka*(theta - theta0))


def theta_ia(theta0r):
    return theta0r


def theta0r_svc(theta0):
    return (1/180)*pi*theta0


# empty sns_update

f_args = []

g_args = ['Ka', 'theta', 'theta0', 'theta0r']

j_args = {'gy': ['Ka', 'theta', 'theta0']}

s_args = OrderedDict([('theta0r', ['theta0'])])

sns_args = []

ia_args = OrderedDict([('theta', ['theta0r'])])

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
             ('gy', [0, 0])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0])])

j_names = ['gy']

init_seq = ['theta', 'Pmg']

need_diag_eps = []
