import numpy as np
from psychopy import visual


class GaussianRDK:
    """
    Gaussian-masked Random Dot Kinematogram (RDK).

    - A proportion (coherence) of dots (signal) move in a common direction.
    - The remaining dots (noise) take a new random direction every frame.
    - A radial Gaussian mask modulates dot opacity so edge dots are less visible.

    Parameters (pixels unless noted):
    - win: PsychoPy Window
    - n_dots: number of dots
    - dot_size: dot size in pixels (scalar)
    - speed: step size in pixels per frame (note: matches existing task usage)
    - dot_life: lifetime in frames for each dot; when life ends, dot is replanted
    - direction: motion direction for signal dots, in degrees (0=right, 90=up)
    - coherence: proportion in [0,1]
    - field_pos: (x, y) center position in window coords (pixels)
    - field_size: diameter of circular field (pixels)
    - gauss_sigma: sigma of Gaussian mask (pixels). If None, defaults to radius/2.
    - color: RGB color triplet in [-1,1] for dots (default white)
    - reassign_life: if True, re-draw signal/noise membership on life reset
    """

    def __init__(
        self,
        win,
        n_dots: int,
        dot_size: float,
        speed: float,
        dot_life: int,
        direction: float,
        coherence: float,
        field_pos: tuple,
        field_size: float,
        gauss_sigma: float | None = None,
        color=(1, 1, 1),
        reassign_life: bool = True,
    ) -> None:
        self.win = win
        self.n = int(max(1, n_dots))
        self.dot_size = float(dot_size)
        self.speed = float(speed)
        self.dot_life = int(max(1, dot_life))
        self.direction = float(direction)
        self.coherence = float(np.clip(coherence, 0.0, 1.0))
        self.field_pos = np.array(field_pos, dtype=float)
        self.field_diam = float(field_size)
        self.radius = self.field_diam / 2.0
        self.gauss_sigma = float(gauss_sigma) if gauss_sigma is not None else (self.radius / 2.0)
        self.color = color
        self.reassign_life = bool(reassign_life)

        # Internal state: positions relative to field center (Nx2), life counters, membership
        self.xys = self._rand_points_in_circle(self.n, self.radius)
        self.life = np.random.randint(1, self.dot_life + 1, size=self.n, dtype=int)
        self._assign_membership()

        # Estimate frame duration (s) for per-frame step scaling (speed is px/s)
        self._frame_dur = getattr(self.win, 'monitorFramePeriod', None) or (1.0 / 60.0)

        # Stim used for drawing
        self.stim = visual.ElementArrayStim(
            self.win,
            nElements=self.n,
            sizes=self.dot_size,
            xys=self._apply_offset(self.xys),
            colors=self.color,
            colorSpace='rgb',
            elementTex=None,
            elementMask='circle',
            opacities=self._compute_opacity(self.xys),
            interpolate=False,
            autoLog=False,
        )

        # Precompute unit vector for signal direction
        theta = np.deg2rad(self.direction)
        self.signal_vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)

    # ---------------------- public API ----------------------
    def draw(self):
        """Update positions one frame and draw the dots."""
        self._step()
        # Update stim arrays and draw
        self.stim.xys = self._apply_offset(self.xys)
        self.stim.opacities = self._compute_opacity(self.xys)
        self.stim.draw()

    def set_direction(self, direction_deg: float):
        self.direction = float(direction_deg)
        theta = np.deg2rad(self.direction)
        self.signal_vec = np.array([np.cos(theta), np.sin(theta)], dtype=float)

    def set_coherence(self, coherence: float):
        self.coherence = float(np.clip(coherence, 0.0, 1.0))
        self._assign_membership()

    def set_field_pos(self, pos: tuple):
        self.field_pos = np.array(pos, dtype=float)

    def set_field_size(self, diameter_px: float, gauss_sigma: float | None = None):
        self.field_diam = float(diameter_px)
        self.radius = self.field_diam / 2.0
        if gauss_sigma is not None:
            self.gauss_sigma = float(gauss_sigma)

    # ---------------------- internals ----------------------
    def _assign_membership(self):
        n_sig = int(round(self.n * self.coherence))
        idx = np.arange(self.n)
        np.random.shuffle(idx)
        self.is_signal = np.zeros(self.n, dtype=bool)
        if n_sig > 0:
            self.is_signal[idx[:n_sig]] = True

    @staticmethod
    def _rand_points_in_circle(n, radius):
        # Sample uniformly over area of a circle
        r = radius * np.sqrt(np.random.rand(n))
        t = 2 * np.pi * np.random.rand(n)
        x = r * np.cos(t)
        y = r * np.sin(t)
        return np.column_stack([x, y]).astype(float)

    def _apply_offset(self, xys):
        return xys + self.field_pos[None, :]

    def _compute_opacity(self, xys):
        # Gaussian radial mask centered at field center
        r2 = np.sum(xys**2, axis=1)
        sigma2 = (self.gauss_sigma ** 2)
        # Avoid division by zero
        if sigma2 <= 0:
            return np.ones(self.n, dtype=float)
        alpha = np.exp(-0.5 * r2 / sigma2)
        # Clamp to [0,1]
        return np.clip(alpha, 0.0, 1.0)

    def _step(self):
        # Decrement life and respawn where needed; optionally reassign membership
        self.life -= 1
        dead = self.life <= 0
        if np.any(dead):
            self.xys[dead] = self._rand_points_in_circle(int(np.sum(dead)), self.radius)
            self.life[dead] = self.dot_life
            if self.reassign_life:
                # re-assign membership while preserving total coherence proportion
                self._assign_membership()

        # Compute per-dot step vectors (speed interpreted as px/s)
        step_scale = self.speed * self._frame_dur
        steps = np.empty_like(self.xys)
        # Signal dots: constant direction
        steps[self.is_signal] = self.signal_vec[None, :] * step_scale
        # Noise dots: random direction every frame
        n_noise = np.count_nonzero(~self.is_signal)
        if n_noise > 0:
            thetas = 2 * np.pi * np.random.rand(n_noise)
            rand_vecs = np.column_stack([np.cos(thetas), np.sin(thetas)])
            steps[~self.is_signal] = rand_vecs * step_scale

        # Update positions
        self.xys += steps

        # Wrap/replant dots that leave the circular field (replant uniformly)
        outside = np.sum(self.xys**2, axis=1) > (self.radius * self.radius)
        if np.any(outside):
            self.xys[outside] = self._rand_points_in_circle(int(np.sum(outside)), self.radius)
