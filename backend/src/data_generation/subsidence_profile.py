"""
subsidence_profile.py — CMRI/NCB Influence-Function Subsidence Model
=====================================================================

Implements the standard Indian geotechnical method for predicting the
subsidence trough shape above an underground extraction panel.

Physics basis:
    The influence-function (profile-function) method models the subsidence
    trough as an error-function (erf) shaped curve. For a finite-width
    panel, the full profile is the superposition of two half-plane
    solutions — one for each panel edge.

Key references:
    [1] CMRI (Central Mining Research Institute) Technical Manual on
        Mine Subsidence, Dhanbad (1997).
    [2] IS 14562:1998 — Indian Standard for prediction of ground
        movement and subsidence due to underground mining.
    [3] Saxena & Singh, "Subsidence prediction in Indian coal mines",
        J. Mines Metals & Fuels (2008).
    [4] Chatterjee et al. (2015), "Subsidence monitoring of Jharia
        Coalfield using GPS", J. Earth System Science, 124(5):1071-1081.
        Reports: subsidence rates 14.8–85 cm/year, horizontal
        displacement ~20–25% of vertical.
    [5] Djamaluddin et al. (2011), "Time-dependent subsidence prediction
        by means of field data", Int. J. Mining Sci. & Tech.
    [6] Raniganj InSAR rates: ~21 mm/year (Dey et al. 2018).
"""

import numpy as np
from scipy.special import erf, erfinv


class PanelGeometry:
    """Holds the randomised physical geometry of a single mine panel.

    All parameters are within literature-grounded ranges for Indian coal
    measures (see config.yaml comments for citations).
    """

    def __init__(self, depth_m, width_m, extraction_thickness_m,
                 angle_of_draw_deg, subsidence_factor,
                 face_advance_rate_m_per_day, panel_id="panel_001"):
        self.panel_id = panel_id
        self.depth = depth_m                    # H — extraction depth (m)
        self.width = width_m                    # W — panel width (m)
        self.extraction_thickness = extraction_thickness_m  # seam height (m)
        self.angle_of_draw = angle_of_draw_deg  # degrees
        self.subsidence_factor = subsidence_factor  # a (dimensionless)
        self.face_advance_rate = face_advance_rate_m_per_day  # m/day

        # ---------- Derived quantities ----------
        # Maximum subsidence: S_max = a × extraction_thickness (meters)
        # This is the deepest point of the trough at the centre of a
        # supercritical panel, after full extraction.
        self.s_max_m = self.subsidence_factor * self.extraction_thickness
        self.s_max_mm = self.s_max_m * 1000.0

        # Angle-of-draw boundary distance from panel edge at the surface.
        # By definition, the trough edge is located at:
        #   x_boundary = H × tan(angle_of_draw)
        # beyond each panel edge. Outside this boundary, subsidence ≈ 0.
        self.angle_of_draw_rad = np.radians(self.angle_of_draw)
        self.x_boundary = self.depth * np.tan(self.angle_of_draw_rad)

        # ---------- sigma derivation for the influence function ----------
        # We model each panel edge as an error-function transition:
        #   S_edge(d) = S_max/2 × [1 − erf(d / (σ√2))]
        # where d is horizontal distance from the panel edge (positive = outward).
        #
        # At d = x_boundary (the angle-of-draw limit), we want the subsidence
        # to have decayed to ~5% of the half-plane value:
        #   0.05 = 0.5 × [1 − erf(x_boundary / (σ√2))]
        #   ⟹ erf(x_boundary / (σ√2)) = 0.90
        #   ⟹ x_boundary / (σ√2) = erfinv(0.90)
        #   ⟹ σ = x_boundary / (√2 × erfinv(0.90))
        #
        # erfinv(0.90) ≈ 1.1631 (exact from scipy).
        # This gives σ a direct physical link to the angle of draw — NOT a
        # magic number. A deeper panel (larger H) or wider draw angle produces
        # a broader transition, exactly as observed in the field.
        self.sigma = self.x_boundary / (np.sqrt(2) * erfinv(0.90))

    def to_dict(self):
        """Serialise for panel_meta.json."""
        return {
            "panel_id": self.panel_id,
            "depth_m": self.depth,
            "width_m": self.width,
            "extraction_thickness_m": self.extraction_thickness,
            "angle_of_draw_deg": self.angle_of_draw,
            "subsidence_factor": self.subsidence_factor,
            "face_advance_rate_m_per_day": self.face_advance_rate,
            "s_max_mm": round(self.s_max_mm, 2),
            "sigma_m": round(self.sigma, 2),
            "x_boundary_m": round(self.x_boundary, 2),
        }


class SubsidenceProfile:
    """Computes displacement, tilt, and strain at arbitrary (x, t) using
    the CMRI/NCB influence-function model for a given PanelGeometry.

    Coordinate convention:
        x = horizontal distance from panel CENTRE, perpendicular to the
            panel's long axis. x = 0 is directly above panel centre.
        y = horizontal position along the panel's long axis (used only
            for face-advance timing, not for trough shape).
        t = time in HOURS since simulation start.
    """

    def __init__(self, geometry: PanelGeometry,
                 active_phase_fraction=0.80,
                 active_phase_days=60,
                 residual_tau_days=270,
                 h_disp_ratio=0.22):
        self.geom = geometry
        self.active_phase_fraction = active_phase_fraction
        self.active_phase_days = active_phase_days
        self.residual_tau_days = residual_tau_days
        # Horizontal-to-vertical displacement ratio (Chatterjee 2015: 20–25%)
        self.h_disp_ratio = h_disp_ratio

        # Pre-compute active-phase exponential rate constant.
        # We want: f_active × (1 − exp(−λ_active × T_active)) = active_phase_fraction
        # where T_active is in hours.
        self.T_active_hours = active_phase_days * 24.0
        # Solve: 1 − exp(−λ × T) = fraction  ⟹  λ = −ln(1 − fraction) / T
        self._lambda_active = -np.log(1.0 - self.active_phase_fraction) / self.T_active_hours
        # Residual phase time constant (hours)
        self._tau_residual_hours = self.residual_tau_days * 24.0

    # ──────────────────────────────────────────────────────────
    #  SPATIAL PROFILE  S(x)
    # ──────────────────────────────────────────────────────────

    def spatial_profile(self, x):
        """Subsidence magnitude at horizontal position x (metres from centre).

        For a finite-width panel with edges at ±W/2, the full trough profile
        is the superposition of two complementary-erf half-plane solutions:

            S(x) = S_max/2 × [ erf((W/2 + x) / (σ√2)) + erf((W/2 − x) / (σ√2)) ]

        Properties:
            - At x = 0 (centre):  S ≈ S_max for supercritical panels (W >> σ).
            - At x = ±(W/2 + x_boundary):  S ≈ 0.
            - For subcritical panels (W < ~1.4H), centre subsidence is less
              than S_max because the two erf tails overlap incompletely.

        Returns: S(x) in millimetres (always ≥ 0).
        """
        x = np.asarray(x, dtype=float)
        half_w = self.geom.width / 2.0
        s2 = self.geom.sigma * np.sqrt(2)

        profile = 0.5 * self.geom.s_max_mm * (
            erf((half_w + x) / s2) + erf((half_w - x) / s2)
        )
        return np.maximum(profile, 0.0)

    # ──────────────────────────────────────────────────────────
    #  TIME PROFILE  f(t)
    # ──────────────────────────────────────────────────────────

    def time_profile(self, t_hours, t_onset_hours=0.0):
        """Fraction of maximum subsidence realised at time t.

        Uses a two-rate exponential model:
            Active phase (t ≤ T_active after onset):
                f(t) = 1 − exp(−λ_active × Δt)
            Residual phase (t > T_active after onset):
                f(t) = f_end_active + (1 − f_end_active) × [1 − exp(−Δt_res / τ_res)]

        This captures the observed behaviour [Ref 5] where 70–90% of total
        subsidence occurs rapidly during the active phase (weeks to months
        after the mining face passes), then tapers into slow residual
        settlement that may continue for years.

        Args:
            t_hours:       time since simulation start (hours).
            t_onset_hours: time at which subsidence begins at this point
                           (depends on face advance position).

        Returns: f ∈ [0, 1] — fraction of S_max realised.
        """
        t_hours = np.asarray(t_hours, dtype=float)
        dt = t_hours - t_onset_hours  # time since onset
        f = np.zeros_like(dt)

        # Before onset: no subsidence
        active_mask = dt > 0

        # Active phase: rapid exponential approach
        active_dt = np.minimum(dt[active_mask], self.T_active_hours)
        f_active = 1.0 - np.exp(-self._lambda_active * active_dt)

        # Residual phase: for windows past the active duration
        residual_mask = dt[active_mask] > self.T_active_hours
        f_at_end_active = 1.0 - np.exp(-self._lambda_active * self.T_active_hours)
        residual_dt = dt[active_mask][residual_mask] - self.T_active_hours
        f_residual = f_at_end_active + (1.0 - f_at_end_active) * (
            1.0 - np.exp(-residual_dt / self._tau_residual_hours)
        )

        f_result = f_active.copy()
        f_result[residual_mask] = f_residual
        f[active_mask] = f_result

        return f

    # ──────────────────────────────────────────────────────────
    #  VERTICAL DISPLACEMENT  displacement(x, t)
    # ──────────────────────────────────────────────────────────

    def vertical_displacement(self, x, t_hours, t_onset_hours=0.0):
        """Vertical surface displacement in mm at position x and time t.

        displacement(x, t) = S(x) × f(t)

        Positive = downward (subsidence convention).
        """
        s_x = self.spatial_profile(x)
        f_t = self.time_profile(t_hours, t_onset_hours)
        # Broadcasting: s_x has shape (n_positions,), f_t has shape (n_times,)
        # Use outer product if both are arrays
        if np.ndim(s_x) > 0 and np.ndim(f_t) > 0:
            return np.outer(f_t, s_x)   # shape: (n_times, n_positions)
        return s_x * f_t

    # ──────────────────────────────────────────────────────────
    #  HORIZONTAL DISPLACEMENT
    # ──────────────────────────────────────────────────────────

    def horizontal_displacement(self, x, t_hours, t_onset_hours=0.0):
        """Horizontal surface displacement in mm at position x and time t.

        Empirically observed at ~20–25% of vertical displacement magnitude,
        directed radially AWAY from the trough centre (Chatterjee et al.
        2015, Jharia GPS study).

        Sign convention: positive = away from centre (tensile zone outside
        trough), negative = toward centre (compressive zone inside trough).
        The sign is derived from the spatial derivative of vertical
        displacement: d(S)/dx > 0 on one side, < 0 on the other.
        """
        v_disp = self.vertical_displacement(x, t_hours, t_onset_hours)

        # Direction: proportional to the slope of the subsidence trough.
        # The derivative of S(x) gives the tilt direction; horizontal
        # displacement follows the same sign (toward the trough centre
        # where the ground is sinking most).
        x = np.asarray(x, dtype=float)
        dx = max(self.geom.sigma * 0.01, 0.5)  # small step for numerical derivative
        ds_dx = (self.spatial_profile(x + dx) - self.spatial_profile(x - dx)) / (2 * dx)

        # Normalise direction and scale by ratio
        sign = np.sign(ds_dx)
        h_disp = self.h_disp_ratio * np.abs(v_disp)

        # Apply direction: negative sign because horizontal displacement
        # is toward the centre where subsidence is increasing (ds_dx < 0
        # on the positive-x side means ground moves inward).
        if np.ndim(h_disp) == 2:
            return -sign[np.newaxis, :] * h_disp
        return -sign * h_disp

    # ──────────────────────────────────────────────────────────
    #  TILT (spatial derivative of displacement)
    # ──────────────────────────────────────────────────────────

    def tilt_at_positions(self, x_positions, t_hours, t_onset_hours=0.0,
                          node_spacing_m=50.0):
        """Surface tilt in degrees at each node position.

        Tilt = arctan(d(vertical_displacement) / dx)

        Computed as a finite difference between adjacent x-positions at the
        same time step. This is exactly what a tilt sensor measures: the
        angle of the ground surface relative to horizontal at that point.

        IMPORTANT PHYSICAL INSIGHT: Tilt is the DERIVATIVE of the
        subsidence profile, so it is LARGEST at the panel EDGE (the
        inflection point of the S-curve), NOT at the trough centre where
        subsidence itself is largest but the ground is approximately flat.
        This is why deploying tilt sensors near the trough edge provides
        the best early-warning sensitivity.
        """
        x_positions = np.asarray(x_positions, dtype=float)
        dx = node_spacing_m

        # Vertical displacement at x ± dx/2 (central difference)
        v_plus = self.vertical_displacement(x_positions + dx / 2,
                                            t_hours, t_onset_hours)
        v_minus = self.vertical_displacement(x_positions - dx / 2,
                                             t_hours, t_onset_hours)

        # dv/dx in mm/m — then convert to angle
        slope = (v_plus - v_minus) / dx  # mm per metre
        slope_ratio = slope / 1000.0     # dimensionless (m/m)

        # arctan gives tilt angle in radians → convert to degrees
        tilt_deg = np.degrees(np.arctan(slope_ratio))
        return tilt_deg

    # ──────────────────────────────────────────────────────────
    #  STRAIN between adjacent nodes
    # ──────────────────────────────────────────────────────────

    def strain_between_nodes(self, x1, x2, t_hours, t_onset_hours=0.0):
        """Horizontal strain between two adjacent nodes in mm.

        strain = (horizontal_displacement(x2, t) − horizontal_displacement(x1, t))
                 / node_spacing

        This is what an inter-node stretch/strain sensor physically
        measures — the differential horizontal movement between two
        surface points. Positive = tensile (points moving apart),
        negative = compressive (points moving together).

        In the subsidence trough, tensile strain occurs outside the panel
        edges and compressive strain occurs above the panel centre.
        """
        h1 = self.horizontal_displacement(x1, t_hours, t_onset_hours)
        h2 = self.horizontal_displacement(x2, t_hours, t_onset_hours)
        spacing = abs(x2 - x1)
        if spacing < 1e-6:
            return np.zeros_like(h1)
        # Return as mm (raw differential displacement)
        return h2 - h1

    # ──────────────────────────────────────────────────────────
    #  FACE ADVANCE: compute onset time per node y-position
    # ──────────────────────────────────────────────────────────

    def onset_time_for_position(self, y_m, y_start=0.0, sim_onset_day=30):
        """Time (in hours) at which subsidence begins at a point at y-position
        along the panel axis.

        The mining face advances at a constant rate. When the face passes
        beneath a surface point, subsidence begins at that point.

            t_onset = sim_onset_offset + (y − y_start) / face_advance_rate

        This creates the PROPAGATING FRONT of correlated subsidence across
        nodes positioned along the panel, which is a key discriminator
        from spatially-uncorrelated confounders.
        """
        y_m = np.asarray(y_m, dtype=float)
        dy = np.maximum(y_m - y_start, 0.0)
        days_after_start = dy / self.geom.face_advance_rate
        return (sim_onset_day + days_after_start) * 24.0  # convert to hours
