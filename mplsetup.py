import matplotlib.pyplot as plt

def rcsetup():
    plt.rc("figure", dpi=150, facecolor=(1, 1, 1))
    plt.rc("font", family='palatino linotype', size=14)
    plt.rc("axes", facecolor=(1, .99, .95), titlesize=18)
    plt.rc("mathtext", fontset='cm')

def make_plot(ax, title, xlabel, ylabel, drawax=True, drawgrid=True, ylim=None, xlim=None):    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if drawax:
        ax.axvline(0, lw=1, color=(.38, .36, .33), zorder=1)
        ax.axhline(0, lw=1, color=(.38, .36, .33), zorder=1)
    if drawgrid:
        ax.set_axisbelow(True)
        ax.grid(ls=":")
    if ylim:
        ax.set_ylim(*ylim)
    if xlim:
        ax.set_xlim(*xlim)