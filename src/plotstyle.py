"""Shared figure style: rc parameters, colour palette and paired marker shapes.

    from src import plotstyle as ps
    ps.apply()                       # once, before making figures
    ax.plot(x, y, **ps.style(i))     # i-th category: colour PALETTE[i], marker MARKERS[i % 4]

Colours and markers change together, so categories stay distinguishable in greyscale and for colour-blind
readers. `apply()` also sets the default property cycle to the same pairs, so plain `ax.plot` calls follow it.
Figure widths: SINGLE (one journal column) and DOUBLE (full page width), in inches.
"""
import matplotlib
from cycler import cycler
from matplotlib import font_manager

FONTS = ['Arial', 'Liberation Sans', 'Helvetica', 'DejaVu Sans']   # Liberation Sans is metric-compatible with Arial
PALETTE = ['#104485', '#f17f03', '#418c47', '#b63667', '#6b4c9a', '#2a8c8c', '#b5546b', '#7d8a2e']
MARKERS = ['o', 's', '^', 'D']
SINGLE, DOUBLE = 3.5, 7.2


def _maths_font():
    """Arial for maths text where installed; otherwise the first available sans-serif font in FONTS."""
    have = {f.name for f in font_manager.fontManager.ttflist}
    return next((f for f in FONTS if f in have), 'DejaVu Sans')


def rc_params():
    m = _maths_font()
    return {
        'font.family': 'sans-serif', 'font.sans-serif': FONTS, 'font.size': 9, 'font.weight': 'bold',
        'axes.labelweight': 'bold', 'axes.labelsize': 7, 'axes.titleweight': 'bold', 'axes.titlesize': 9,
        'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': False,
        'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.minor.size': 1.8, 'ytick.minor.size': 1.8, 'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'legend.frameon': False,
        'lines.linewidth': 1.6, 'lines.markersize': 5,
        'mathtext.fontset': 'custom', 'mathtext.rm': m, 'mathtext.it': f'{m}:italic', 'mathtext.bf': f'{m}:bold',
        'mathtext.sf': m, 'mathtext.default': 'it',
        'savefig.dpi': 300,          # use 600 for line art if a journal asks (raster minimum usually 300-350)
        'savefig.bbox': 'tight',
        'pdf.fonttype': 42,          # embed TrueType so PDF text stays editable
        'axes.prop_cycle': cycler(color=PALETTE, marker=(MARKERS * 2)[:len(PALETTE)]),
    }


RC_PARAMS = rc_params()


def apply():
    matplotlib.rcParams.update(rc_params())


def color(i):
    return PALETTE[i % len(PALETTE)]


def marker(i):
    return MARKERS[i % len(MARKERS)]


def style(i, hollow=False):
    """Plot keywords for the i-th category; hollow=True gives an open marker (e.g. for a comparison arm)."""
    s = dict(color=color(i), marker=marker(i))
    if hollow:
        s['markerfacecolor'] = 'white'
    return s


def plain_log(ax, axis='y'):
    """Log scale with plain tick labels at 1, 2 and 5 per decade (0.1, 0.2, 0.5, 1 ...), not powers of ten."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    a = ax.yaxis if axis == 'y' else ax.xaxis
    a.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    a.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}')); a.set_minor_formatter(NullFormatter())


def save(fig, path, pdf=True):
    """Save as PNG (300 dpi) and, by default, a PDF with editable text alongside."""
    fig.savefig(path)
    if pdf and path.endswith('.png'):
        fig.savefig(path[:-4] + '.pdf')
