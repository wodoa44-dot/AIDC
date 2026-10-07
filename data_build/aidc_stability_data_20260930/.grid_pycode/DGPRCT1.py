from collections import OrderedDict  # NOQA

from numpy import ones_like, zeros_like, full, array                # NOQA
from numpy import nan, pi, sin, cos, tan, sqrt, exp, select         # NOQA
from numpy import greater_equal, less_equal, greater, less, equal   # NOQA
from numpy import logical_and, logical_or, logical_not              # NOQA
from numpy import real, imag, conj, angle, radians, abs             # NOQA
from numpy import arcsin, arccos, arctan, arctan2                   # NOQA
from numpy import log                                               # NOQA

from andes.thirdparty.npfunc import *                               # NOQA


md5 = "f338c98879b9e067fcb582a93c658934"

def f_update(LVl1_zi, LVl2_zi, LVu1_zi, LVu2_zi, Lfl1_zi, Lfl2_zi, Lfu1_zi, Tfl1, Tfl2, Tfu1, Tfu2, Tres, Tvl1, Tvl2, Tvl3, Tvu1, Tvu2, res):
    return (Lfl1_zi*(1 - res) - Tfl1*res/Tres, Lfl2_zi*(1 - res) - Tfl2*res/Tres, Lfu1_zi*(1 - res) - Tfu1*res/Tres, Lfl2_zi*(1 - res) - Tfu2*res/Tres, LVl1_zi*(1 - res) - Tvl1*res/Tres, LVl2_zi*(1 - res) - Tvl2*res/Tres, LVl2_zi*(1 - res) - Tvl3*res/Tres, LVu1_zi*(1 - res) - Tvu1*res/Tres, LVu2_zi*(1 - res) - Tvu2*res/Tres,)


def g_update(IAWVl1_lim_zu, IAWVl2_lim_zu, IAWVl3_lim_zu, IAWVu1_lim_zu, IAWVu2_lim_zu, IAWfl1_lim_zu, IAWfl2_lim_zu, IAWfu1_lim_zu, IAWfu2_lim_zu, LVl1_zi, LVl2_zi, LVl3_zi, LVu1_zi, LVu2_zi, Ldsum_zu, Lfl1_zi, Lfl2_zi, Lfu1_zi, Lfu2_zi, Pdrp, Pext, Pref, Qdrp, Qref, Ven, dsum, f, fHz, fen, fn, ue):
    return (f*fn - fHz, Ven*(IAWVl1_lim_zu*LVl1_zi + IAWVl2_lim_zu*LVl2_zi + IAWVl3_lim_zu*LVl3_zi + IAWVu1_lim_zu*LVu1_zi + IAWVu2_lim_zu*LVu2_zi) - dsum + fen*(IAWfl1_lim_zu*Lfl1_zi + IAWfl2_lim_zu*Lfl2_zi + IAWfu1_lim_zu*Lfu1_zi + IAWfu2_lim_zu*Lfu2_zi), Ldsum_zu - ue, 0, 0, -f*fn*ue, 0, 0, 0, -ue*(Pdrp + Pext + Pref), 0, 0, -ue*(Qdrp + Qref), 0,)


def gy_update(Pdrp, Pext, Pref, Qdrp, Qref, f, fn, ue):
    return (-1, fn, -1, -1, -f*fn, -fn*ue, -Pdrp - Pext - Pref, -ue, -ue, -ue, -Qdrp - Qref, -ue, -ue)


def IAWfl1_y_ia():
    return 0


def IAWfl2_y_ia():
    return 0


def IAWfu1_y_ia():
    return 0


def IAWfu2_y_ia():
    return 0


def IAWVl1_y_ia():
    return 0


def IAWVl2_y_ia():
    return 0


def IAWVl3_y_ia():
    return 0


def IAWVu1_y_ia():
    return 0


def IAWVu2_y_ia():
    return 0


def fHz_ia(f, fn):
    return f*fn


def dsum_ia():
    return 0


def ue_ia():
    return 0


def ltu_svc():
    return 0.800000000000000


def ltl_svc():
    return 0.200000000000000


def zero_svc():
    return 0


def res_svc():
    return 0


# empty sns_update

f_args = ['LVl1_zi',
 'LVl2_zi',
 'LVu1_zi',
 'LVu2_zi',
 'Lfl1_zi',
 'Lfl2_zi',
 'Lfu1_zi',
 'Tfl1',
 'Tfl2',
 'Tfu1',
 'Tfu2',
 'Tres',
 'Tvl1',
 'Tvl2',
 'Tvl3',
 'Tvu1',
 'Tvu2',
 'res']

g_args = ['IAWVl1_lim_zu',
 'IAWVl2_lim_zu',
 'IAWVl3_lim_zu',
 'IAWVu1_lim_zu',
 'IAWVu2_lim_zu',
 'IAWfl1_lim_zu',
 'IAWfl2_lim_zu',
 'IAWfu1_lim_zu',
 'IAWfu2_lim_zu',
 'LVl1_zi',
 'LVl2_zi',
 'LVl3_zi',
 'LVu1_zi',
 'LVu2_zi',
 'Ldsum_zu',
 'Lfl1_zi',
 'Lfl2_zi',
 'Lfu1_zi',
 'Lfu2_zi',
 'Pdrp',
 'Pext',
 'Pref',
 'Qdrp',
 'Qref',
 'Ven',
 'dsum',
 'f',
 'fHz',
 'fen',
 'fn',
 'ue']

j_args = {'gy': ['Pdrp', 'Pext', 'Pref', 'Qdrp', 'Qref', 'f', 'fn', 'ue']}

s_args = OrderedDict([('ltu', []), ('ltl', []), ('zero', []), ('res', [])])

sns_args = []

ia_args = OrderedDict([('IAWfl1_y', []),
             ('IAWfl2_y', []),
             ('IAWfu1_y', []),
             ('IAWfu2_y', []),
             ('IAWVl1_y', []),
             ('IAWVl2_y', []),
             ('IAWVl3_y', []),
             ('IAWVu1_y', []),
             ('IAWVu2_y', []),
             ('fHz', ['f', 'fn']),
             ('dsum', []),
             ('ue', [])])

ii_args = OrderedDict()

ij_args = OrderedDict()

ijac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0, 1, 2, 5, 5, 9, 9, 9, 9, 12, 12, 12])])

jjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [9, 12, 10, 11, 11, 12, 11, 15, 16, 17, 11, 19, 20])])

vjac = OrderedDict([('fxc', []),
             ('fx', []),
             ('fyc', []),
             ('fy', []),
             ('gxc', []),
             ('gx', []),
             ('gyc', []),
             ('gy', [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])])

j_names = ['gy']

init_seq = ['IAWfl1_y',
 'IAWfl2_y',
 'IAWfu1_y',
 'IAWfu2_y',
 'IAWVl1_y',
 'IAWVl2_y',
 'IAWVl3_y',
 'IAWVu1_y',
 'IAWVu2_y',
 'f',
 'fHz',
 'dsum',
 'ue',
 'fin',
 'fHzl',
 'Pext',
 'Pref',
 'Pdrp',
 'Psum',
 'Qdrp',
 'Qref',
 'Qsum',
 'v']

need_diag_eps = []
