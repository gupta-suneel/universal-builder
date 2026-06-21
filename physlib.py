"""
physlib.py
==========
Vetted physics-figure drawing toolkit for Universal Builder.

The problem this solves: an LLM writing matplotlib code "blind" gets the geometry
of schematic diagrams (pulleys, free-body diagrams, inclines) subtly wrong. These
functions have the correct geometry baked in, so the model just CALLS them with
parameters instead of hand-placing coordinates.

Every function takes a matplotlib Axes `ax` (or uses the current axes) and draws
onto it. Geometry-critical figures use equal aspect so angles and circles are true.

Conventions:
- SI-style schematic, not to physical scale unless noted.
- Force arrows point the correct way; for one ideal rope the tension is equal
  on both sides (Atwood); mechanical advantage = number of supporting strands.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrow, Arc, Polygon


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def _ax(ax):
    return ax if ax is not None else plt.gca()


def _clean(ax, equal=True):
    if equal:
        ax.set_aspect("equal")
    ax.axis("off")


def _arrow(ax, x, y, dx, dy, color="k", label=None, lw=2.0, label_dx=0.0, label_dy=0.0):
    ax.annotate("", xy=(x + dx, y + dy), xytext=(x, y),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=color,
                                mutation_scale=16))
    if label:
        ax.text(x + dx + label_dx, y + dy + label_dy, label, color=color,
                fontsize=12, ha="center", va="center")


def _ground(ax, x0, x1, y, depth=0.18, n=14):
    """Hatched ground/support line."""
    ax.plot([x0, x1], [y, y], "k", lw=2)
    for xi in np.linspace(x0, x1, n):
        ax.plot([xi, xi - depth], [y, y - depth], "k", lw=1)


# ---------------------------------------------------------------------------
# MECHANICS SCHEMATICS
# ---------------------------------------------------------------------------

def free_body_diagram(ax=None, forces=None, body="m", title=None):
    """
    Free-body diagram. `forces` is a list of (label, angle_deg, magnitude),
    angle measured CCW from the +x axis. Arrow lengths scale with magnitude.
    Example: forces=[("N",90,1.0),("mg",270,1.0),("f",180,0.5),("F",0,0.8)]
    """
    ax = _ax(ax)
    forces = forces or [("N", 90, 1.0), ("mg", 270, 1.0)]
    mags = [m for _, _, m in forces] or [1.0]
    mmax = max(mags) or 1.0
    L = 1.0  # longest arrow length
    ax.add_patch(Rectangle((-0.22, -0.22), 0.44, 0.44, fc="#cfe3ff", ec="k", lw=1.5))
    ax.text(0, 0, body, ha="center", va="center", fontsize=12)
    for label, ang, mag in forces:
        a = np.radians(ang)
        ln = L * (mag / mmax)
        dx, dy = ln * np.cos(a), ln * np.sin(a)
        _arrow(ax, 0, 0, dx, dy, label=label,
               label_dx=0.16 * np.cos(a), label_dy=0.16 * np.sin(a))
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    if title:
        ax.set_title(title)
    _clean(ax)
    return ax


def inclined_plane(ax=None, angle_deg=30.0, show_forces=True, block_label="m",
                   mu_friction=True, title=None):
    """Inclined plane with a block on the slope and (optionally) resolved forces."""
    ax = _ax(ax)
    th = np.radians(angle_deg)
    L = 3.0
    H = L * np.tan(th)
    # Triangle: A bottom-left, B bottom-right, C top-right. Slope = A->C.
    A, B, C = np.array([0, 0]), np.array([L, 0]), np.array([L, H])
    ax.add_patch(Polygon([A, B, C], closed=True, fc="#e9e9e9", ec="k", lw=2))
    _ground(ax, -0.4, L + 0.4, 0)
    # angle arc at A
    ax.add_patch(Arc(A, 1.2, 1.2, angle=0, theta1=0, theta2=angle_deg, color="k"))
    ax.text(0.95, 0.16, f"{angle_deg:.0f}", fontsize=11)
    # block centre on the slope (at 60% up), oriented along slope
    t = 0.55
    base = A + t * (C - A)
    along = (C - A) / np.linalg.norm(C - A)          # up the slope
    normal = np.array([-along[1], along[0]])         # outward normal
    s = 0.5
    centre = base + normal * s * 0.6
    corners = [centre + along * s / 2 + normal * s / 2,
               centre - along * s / 2 + normal * s / 2,
               centre - along * s / 2 - normal * s / 2,
               centre + along * s / 2 - normal * s / 2]
    ax.add_patch(Polygon(corners, closed=True, fc="#cfe3ff", ec="k", lw=1.5))
    ax.text(*centre, block_label, ha="center", va="center", fontsize=11)
    if show_forces:
        g = 1.0
        _arrow(ax, centre[0], centre[1], 0, -g, color="tab:red",
               label="mg", label_dy=-0.18)
        _arrow(ax, centre[0], centre[1], normal[0] * g, normal[1] * g,
               color="tab:blue", label="N",
               label_dx=normal[0] * 0.2, label_dy=normal[1] * 0.2)
        if mu_friction:
            _arrow(ax, centre[0], centre[1], along[0] * g * 0.6, along[1] * g * 0.6,
                   color="tab:green", label="f",
                   label_dx=along[0] * 0.25, label_dy=along[1] * 0.25)
    ax.set_xlim(-0.8, L + 1.2)
    ax.set_ylim(-0.8, H + 1.4)
    if title:
        ax.set_title(title)
    _clean(ax)
    return ax


def pulley(ax, cx, cy, r=0.35):
    """Draw a single pulley wheel with a centre axle (primitive)."""
    ax.add_patch(Circle((cx, cy), r, fc="#dddddd", ec="k", lw=2))
    ax.add_patch(Circle((cx, cy), 0.05, fc="k"))


def atwood(ax=None, m1=3.0, m2=5.0, title=None):
    """
    Atwood machine: one fixed pulley, two masses on ONE rope (equal tension T
    on both sides). Heavier mass is drawn lower to suggest its descent.
    """
    ax = _ax(ax)
    r = 0.5
    cx, cy = 0.0, 4.0
    ceiling = cy + r + 0.5
    _ground(ax, -1.2, 1.2, ceiling)
    ax.plot([cx, cx], [cy + r, ceiling], "k", lw=2)        # mount bracket
    pulley(ax, cx, cy, r)
    # rope arc over the top of the wheel, tangent at the sides (-r, +r)
    th = np.linspace(0, np.pi, 60)
    ax.plot(cx + r * np.cos(th), cy + r * np.sin(th), color="saddlebrown", lw=2)
    # vertical strands from tangent points
    heavier_right = m2 >= m1
    yL = 1.6 if not heavier_right else 1.0
    yR = 1.0 if heavier_right else 1.6
    ax.plot([cx - r, cx - r], [yL, cy], color="saddlebrown", lw=2)
    ax.plot([cx + r, cx + r], [yR, cy], color="saddlebrown", lw=2)
    # blocks
    for x, ytop, label in [(cx - r, yL, f"$m_1$={m1:g}"), (cx + r, yR, f"$m_2$={m2:g}")]:
        ax.add_patch(Rectangle((x - 0.35, ytop - 0.7), 0.7, 0.7, fc="#cfe3ff", ec="k", lw=1.5))
        ax.text(x, ytop - 0.35, label, ha="center", va="center", fontsize=10)
    # equal tension label on both strands
    ax.text(cx - r - 0.28, (yL + cy) / 2, "T", fontsize=11, ha="right")
    ax.text(cx + r + 0.28, (yR + cy) / 2, "T", fontsize=11, ha="left")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(0.0, ceiling + 0.6)
    ax.set_title(title or "Atwood machine (one rope, equal tension T)")
    _clean(ax)
    return ax


def movable_pulley(ax=None, advantage=2, load_label="W", title=None):
    """
    Movable-pulley lift. Draws exactly `advantage` vertical supporting strands,
    so the figure literally shows the mechanical advantage (force = W/advantage).
    """
    ax = _ax(ax)
    advantage = max(1, int(advantage))
    ceiling = 5.0
    _ground(ax, -1.5, 2.5, ceiling)
    load_y = 1.0
    px, py, r = 0.5, load_y + 1.2, 0.45      # movable pulley
    xs = np.linspace(px - r, px + r, advantage)
    for x in xs:
        ax.plot([x, x], [py, ceiling], color="saddlebrown", lw=2)
    pulley(ax, px, py, r)
    # load hanging from the movable pulley
    ax.plot([px, px], [load_y + 0.35, py - r], color="saddlebrown", lw=2)
    ax.add_patch(Rectangle((px - 0.5, load_y - 0.35), 1.0, 0.7, fc="#cfe3ff", ec="k", lw=1.5))
    ax.text(px, load_y, load_label, ha="center", va="center", fontsize=11)
    ax.text(px, load_y - 0.7, f"force = {load_label}/{advantage}", ha="center", fontsize=10)
    ax.set_xlim(-1.8, 2.8)
    ax.set_ylim(0.0, ceiling + 0.6)
    ax.set_title(title or f"Movable pulley (mechanical advantage {advantage})")
    _clean(ax)
    return ax


def spring_mass(ax=None, block_label="m", coils=8, stretch=0.0, title=None):
    """Horizontal spring fixed to a wall on the left, attached to a block."""
    ax = _ax(ax)
    wall_x = 0.0
    ax.plot([wall_x, wall_x], [-0.8, 0.8], "k", lw=3)
    for yi in np.linspace(-0.8, 0.8, 9):
        ax.plot([wall_x, wall_x - 0.18], [yi, yi - 0.18], "k", lw=1)
    x0, x1 = 0.0, 2.2 + stretch
    xs = np.linspace(x0, x1, coils * 2 + 2)
    ys = np.zeros_like(xs)
    ys[1:-1] = 0.25 * np.tile([1, -1], coils)[:len(xs) - 2]
    ax.plot(xs, ys, "k", lw=1.8)
    ax.add_patch(Rectangle((x1, -0.4), 0.8, 0.8, fc="#cfe3ff", ec="k", lw=1.5))
    ax.text(x1 + 0.4, 0, block_label, ha="center", va="center", fontsize=11)
    ax.set_xlim(-0.6, x1 + 1.2)
    ax.set_ylim(-1.2, 1.2)
    if title:
        ax.set_title(title)
    _clean(ax)
    return ax


def pendulum(ax=None, angle_deg=25.0, L=1.0, title=None):
    """Simple pendulum: pivot, rod, bob, vertical reference and angle arc."""
    ax = _ax(ax)
    th = np.radians(angle_deg)
    px, py = 0.0, 0.0
    bx, by = px + L * np.sin(th), py - L * np.cos(th)
    ax.plot([px - 0.4, px + 0.4], [py, py], "k", lw=3)   # support
    ax.plot([px, px], [py, py - L - 0.3], "k--", lw=1)   # vertical reference
    ax.plot([px, bx], [py, by], "k", lw=2)               # rod
    ax.add_patch(Circle((bx, by), 0.12, fc="#cfe3ff", ec="k", lw=1.5))
    ax.add_patch(Arc((px, py), 0.7, 0.7, angle=-90, theta1=0, theta2=angle_deg, color="k"))
    ax.text(0.16, -0.5, r"$\theta$", fontsize=12)
    ax.set_xlim(-1.0, L + 0.4)
    ax.set_ylim(-L - 0.6, 0.4)
    if title:
        ax.set_title(title)
    _clean(ax)
    return ax


# ---------------------------------------------------------------------------
# VECTORS & FIELDS
# ---------------------------------------------------------------------------

def vector(ax=None, origin=(0, 0), comp=(1, 0), label=None, color="k"):
    """Draw a single labeled vector arrow from `origin` with components `comp`."""
    ax = _ax(ax)
    ox, oy = origin
    dx, dy = comp
    _arrow(ax, ox, oy, dx, dy, color=color, label=label,
           label_dx=0.12 * np.sign(dx or 1), label_dy=0.12 * np.sign(dy or 1))
    return ax


def vector_field(ax=None, fx=None, fy=None, xlim=(-3, 3), ylim=(-3, 3),
                 density=1.2, title=None):
    """
    Stream/quiver plot of a 2-D field. fx, fy are functions fx(x, y), fy(x, y)
    that accept numpy arrays. Default: a simple rotational field.
    """
    ax = _ax(ax)
    fx = fx or (lambda x, y: -y)
    fy = fy or (lambda x, y: x)
    x = np.linspace(xlim[0], xlim[1], 25)
    y = np.linspace(ylim[0], ylim[1], 25)
    X, Y = np.meshgrid(x, y)
    U, V = fx(X, Y), fy(X, Y)
    ax.streamplot(X, Y, U, V, density=density, color="tab:blue")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    if title:
        ax.set_title(title)
    return ax


def dipole_field(ax=None, sep=1.0, title=None):
    """Electric field lines of a +/- charge pair (field lines never cross)."""
    ax = _ax(ax)
    x = np.linspace(-3, 3, 40)
    y = np.linspace(-3, 3, 40)
    X, Y = np.meshgrid(x, y)
    px, nx = sep, -sep

    def field(qx, sign):
        dx, dy = X - qx, Y - 0.0
        r2 = dx**2 + dy**2 + 1e-3
        r = np.sqrt(r2)
        return sign * dx / r**3, sign * dy / r**3

    ex1, ey1 = field(px, +1)
    ex2, ey2 = field(nx, -1)
    Ex, Ey = ex1 + ex2, ey1 + ey2
    ax.streamplot(X, Y, Ex, Ey, density=1.4, color="tab:blue", linewidth=0.8)
    ax.plot(px, 0, "o", color="red", ms=14)
    ax.text(px, 0, "+", color="white", ha="center", va="center", fontsize=12)
    ax.plot(nx, 0, "o", color="blue", ms=14)
    ax.text(nx, 0, "-", color="white", ha="center", va="center", fontsize=14)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect("equal")
    ax.set_title(title or "Electric dipole field")
    return ax


# ---------------------------------------------------------------------------
# WAVES & OSCILLATIONS
# ---------------------------------------------------------------------------

def wave(ax=None, amplitude=1.0, wavelength=2.0, phase=0.0, cycles=2.0,
         label=None, color="tab:blue", title=None):
    """A single sinusoidal wave y = A sin(kx + phase)."""
    ax = _ax(ax)
    k = 2 * np.pi / wavelength
    x = np.linspace(0, cycles * wavelength, 500)
    y = amplitude * np.sin(k * x + phase)
    ax.plot(x, y, color=color, label=label)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y")
    if label:
        ax.legend()
    if title:
        ax.set_title(title)
    return ax


def standing_wave(ax=None, n=2, length=1.0, title=None):
    """First few snapshots of the n-th standing-wave mode on a string of length L."""
    ax = _ax(ax)
    x = np.linspace(0, length, 400)
    for amp in (1.0, 0.5, -0.5, -1.0):
        ax.plot(x, amp * np.sin(n * np.pi * x / length),
                color="tab:blue", alpha=0.5)
    ax.plot(x, np.sin(n * np.pi * x / length), color="tab:blue", lw=2)
    ax.axhline(0, color="k", lw=0.8)
    for nodexp in np.linspace(0, length, n + 1):
        ax.plot(nodexp, 0, "ko", ms=5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("displacement")
    ax.set_title(title or f"Standing wave, mode n={n}")
    return ax


def superposition(ax=None, wavelength1=2.0, wavelength2=2.2, title=None):
    """Two waves and their sum (shows beats / interference)."""
    ax = _ax(ax)
    x = np.linspace(0, 12, 1000)
    y1 = np.sin(2 * np.pi * x / wavelength1)
    y2 = np.sin(2 * np.pi * x / wavelength2)
    ax.plot(x, y1, color="tab:blue", alpha=0.4, label="wave 1")
    ax.plot(x, y2, color="tab:green", alpha=0.4, label="wave 2")
    ax.plot(x, y1 + y2, color="tab:red", lw=1.5, label="sum")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y")
    ax.legend()
    if title:
        ax.set_title(title)
    return ax


# ---------------------------------------------------------------------------
# FUNCTION & DATA PLOTS
# ---------------------------------------------------------------------------

def potential_well(ax=None, V=None, xlim=(-3, 3), mark_minima=True, title=None):
    """Plot a potential energy curve V(x) and mark its minima (equilibria)."""
    ax = _ax(ax)
    V = V or (lambda x: 0.5 * x**2)
    x = np.linspace(xlim[0], xlim[1], 600)
    y = V(x)
    ax.plot(x, y, color="tab:blue", lw=2)
    if mark_minima:
        dy = np.gradient(y, x)
        sign_change = np.where(np.diff(np.sign(dy)) > 0)[0]
        for i in sign_change:
            ax.plot(x[i], y[i], "o", color="tab:red")
    ax.set_xlabel("x")
    ax.set_ylabel("V(x)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(title or "Potential energy well")
    return ax


def phase_portrait(ax=None, fx=None, fy=None, xlim=(-3, 3), ylim=(-3, 3),
                   title=None):
    """Phase portrait of a 2-D system x'=fx(x,y), y'=fy(x,y)."""
    ax = _ax(ax)
    fx = fx or (lambda x, y: y)
    fy = fy or (lambda x, y: -np.sin(x))
    x = np.linspace(xlim[0], xlim[1], 25)
    y = np.linspace(ylim[0], ylim[1], 25)
    X, Y = np.meshgrid(x, y)
    ax.streamplot(X, Y, fx(X, Y), fy(X, Y), density=1.2, color="tab:purple")
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title or "Phase portrait")
    return ax


# Catalogue used by the app to tell the model what's available.
CATALOG = """
phys.free_body_diagram(ax, forces=[(label, angle_deg, magnitude), ...], body="m")
phys.inclined_plane(ax, angle_deg=30, show_forces=True, block_label="m")
phys.atwood(ax, m1=3, m2=5)
phys.movable_pulley(ax, advantage=2, load_label="W")
phys.spring_mass(ax, block_label="m", stretch=0.0)
phys.pendulum(ax, angle_deg=25, L=1)
phys.pulley(ax, cx, cy, r=0.35)
phys.vector(ax, origin=(0,0), comp=(1,0), label="v")
phys.vector_field(ax, fx=lambda x,y:-y, fy=lambda x,y:x, xlim=(-3,3), ylim=(-3,3))
phys.dipole_field(ax)
phys.wave(ax, amplitude=1, wavelength=2, phase=0, cycles=2)
phys.standing_wave(ax, n=2, length=1)
phys.superposition(ax, wavelength1=2.0, wavelength2=2.2)
phys.potential_well(ax, V=lambda x: 0.5*x**2, xlim=(-3,3))
phys.phase_portrait(ax, fx=lambda x,y:y, fy=lambda x,y:-np.sin(x))
""".strip()
