"""
Lorenz System Module
====================
Companion code for the blog post:
  "The Stochastic Lorenz System: Chaos Meets Randomness"
  https://tuanle618.github.io/blog/2026/stochastic-lorenz-generative-images/

A comprehensive module for simulating and visualizing the Lorenz attractor
with various stochastic differential equation (SDE) formulations.

Classes:
    LorenzSystem: Main class for solving deterministic and stochastic Lorenz systems
    LorenzPlotter: Visualization utilities for Lorenz trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from dataclasses import dataclass
from typing import Tuple, Optional, List, Union, Any
from enum import Enum


class NoiseType(Enum):
    """Enumeration of available noise types for stochastic Lorenz systems."""

    NONE = "none"
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    ORNSTEIN_UHLENBECK = "ornstein_uhlenbeck"


@dataclass
class LorenzConfig:
    """Configuration for Lorenz system parameters."""

    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    description: str = "Classic Lorenz"

    @classmethod
    def from_preset(cls, index: int) -> "LorenzConfig":
        """
        Create a LorenzConfig from predefined presets.

        Parameters:
            index: Preset index (0-5)

        Returns:
            LorenzConfig instance
        """
        presets = [
            (10.0, 28.0, 8.0 / 3.0, r"Classic ($\sigma$=10, $\rho$=28, $\beta$=8/3)"),
            (
                10.0,
                24.0,
                8.0 / 3.0,
                r"Compressed ($\sigma$=10, $\rho$=24, $\beta$=8/3)",
            ),
            (10.0, 35.0, 8.0 / 3.0, r"Extended ($\sigma$=10, $\rho$=35, $\beta$=8/3)"),
            (16.0, 45.6, 4.0, r"Alternative ($\sigma$=16, $\rho$=45.6, $\beta$=4)"),
            (10.0, 28.0, 2.0, r"Tight Spiral ($\sigma$=10, $\rho$=28, $\beta$=2)"),
            (8.0, 30.0, 3.0, r"Loose ($\sigma$=8, $\rho$=30, $\beta$=3)"),
        ]

        if 0 <= index < len(presets):
            sigma, rho, beta, desc = presets[index]
            return cls(sigma=sigma, rho=rho, beta=beta, description=desc)
        else:
            raise ValueError(f"Index must be between 0 and {len(presets)-1}")


@dataclass
class LorenzResult:
    """Container for Lorenz system solution results."""

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    config: LorenzConfig
    noise_type: NoiseType
    # Optional noise process arrays (for OU noise)
    eta_x: Optional[np.ndarray] = None
    eta_y: Optional[np.ndarray] = None
    eta_z: Optional[np.ndarray] = None

    def as_tuple(self) -> Tuple[np.ndarray, ...]:
        """Return data as tuple (t, x, y, z) or (t, x, y, z, eta_x, eta_y, eta_z)."""
        if self.eta_x is not None:
            return (self.t, self.x, self.y, self.z, self.eta_x, self.eta_y, self.eta_z)
        return (self.t, self.x, self.y, self.z)


class LorenzSystem:
    """
    A class for simulating the Lorenz attractor with various noise types.

    The Lorenz system is defined by:
        dx/dt = \sigma(y - x)
        dy/dt = x(\rho - z) - y
        dz/dt = xy - \beta z

    Supported noise types:
        - NONE: Deterministic system
        - ADDITIVE: Gaussian white noise added to each equation
        - MULTIPLICATIVE: State-dependent noise
        - ORNSTEIN_UHLENBECK: Colored noise with temporal correlation

    Example:
        >>> lorenz = LorenzSystem()
        >>> result = lorenz.solve(noise_type=NoiseType.ADDITIVE, noise_strength=2.0)
        >>> print(result.x.shape)
    """

    def __init__(self, config: Optional[LorenzConfig] = None):
        """
        Initialize the Lorenz system.

        Parameters:
            config: LorenzConfig instance (uses default if None)
        """
        self.config = config or LorenzConfig()

    @property
    def sigma(self) -> float:
        return self.config.sigma

    @property
    def rho(self) -> float:
        return self.config.rho

    @property
    def beta(self) -> float:
        return self.config.beta

    def solve(
        self,
        t_span: Tuple[float, float] = (0, 100),
        dt: float = 0.001,
        state0: Optional[List[float]] = None,
        noise_type: NoiseType = NoiseType.NONE,
        D: float = 1.0,
        correlation_time: float = 1.0,
        n_points: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> LorenzResult:
        """
        Solve the Lorenz system with the specified noise type.

        Parameters:
            t_span: Time span (start, end)
            dt: Time step for stochastic methods
            state0: Initial conditions [x0, y0, z0]
            noise_type: Type of noise to apply
            D: Noise intensity (diffusion coefficient)
            correlation_time: Correlation time for OU noise
            n_points: Number of points (for deterministic solver)
            seed: Random seed for reproducibility

        Returns:
            LorenzResult containing the solution
        """
        if seed is not None:
            np.random.seed(seed)

        state0 = state0 or [1.0, 1.0, 1.0]

        if noise_type == NoiseType.NONE:
            return self._solve_deterministic(t_span, state0, n_points or 10000)
        elif noise_type == NoiseType.ADDITIVE:
            return self._solve_additive_noise(t_span, dt, state0, D)
        elif noise_type == NoiseType.MULTIPLICATIVE:
            return self._solve_multiplicative_noise(t_span, dt, state0, D)
        elif noise_type == NoiseType.ORNSTEIN_UHLENBECK:
            return self._solve_ou_noise(t_span, dt, state0, D, correlation_time)
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")

    def _solve_deterministic(
        self, t_span: Tuple[float, float], state0: List[float], n_points: int
    ) -> LorenzResult:
        """
        Solve deterministic Lorenz system using Euler method.

        This is equivalent to Euler-Maruyama with noise_strength = 0.
        Using explicit Euler integration for consistency with stochastic solvers
        and to avoid external dependencies.

        Euler method:
            x_{n+1} = x_n + f(x_n) * dt

        For the Lorenz system:
            x_{n+1} = x_n + \sigma(y_n - x_n) * dt
            y_{n+1} = y_n + (x_n(\rho - z_n) - y_n) * dt
            z_{n+1} = z_n + (x_n * y_n - \beta * z_n) * dt
        """
        # Compute dt from n_points to match the desired resolution
        dt = (t_span[1] - t_span[0]) / (n_points - 1)
        t = np.linspace(t_span[0], t_span[1], n_points)

        # Initialize arrays
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        z = np.zeros(n_points)

        # Set initial conditions
        x[0], y[0], z[0] = state0

        # Euler integration (Euler-Maruyama without noise)
        for i in range(1, n_points):
            x_curr, y_curr, z_curr = x[i - 1], y[i - 1], z[i - 1]

            # Deterministic drift terms: f(x) * dt
            dx = self.sigma * (y_curr - x_curr) * dt
            dy = (x_curr * (self.rho - z_curr) - y_curr) * dt
            dz = (x_curr * y_curr - self.beta * z_curr) * dt

            # Euler update: x_{n+1} = x_n + dx
            x[i] = x_curr + dx
            y[i] = y_curr + dy
            z[i] = z_curr + dz

        return LorenzResult(
            t=t,
            x=x,
            y=y,
            z=z,
            config=self.config,
            noise_type=NoiseType.NONE,
        )

    def _solve_additive_noise(
        self,
        t_span: Tuple[float, float],
        dt: float,
        state0: List[float],
        D: float,
    ) -> LorenzResult:
        """
        Solve stochastic Lorenz system using Euler-Maruyama with additive noise.

        Stochastic equations:
            dx = \sigma(y - x)dt + \sqrt{2D} dW_t^{(x)}
            dy = (x(\rho - z) - y)dt + \sqrt{2D} dW_t^{(y)}
            dz = (xy - \beta z)dt + \sqrt{2D} dW_t^{(z)}
        """
        t = np.arange(t_span[0], t_span[1], dt)
        n_steps = len(t)

        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        z = np.zeros(n_steps)

        x[0], y[0], z[0] = state0
        noise_coeff = np.sqrt(2.0 * D * dt)

        for i in range(1, n_steps):
            x_curr, y_curr, z_curr = x[i - 1], y[i - 1], z[i - 1]

            # Deterministic terms
            dx_det = self.sigma * (y_curr - x_curr) * dt
            dy_det = (x_curr * (self.rho - z_curr) - y_curr) * dt
            dz_det = (x_curr * y_curr - self.beta * z_curr) * dt

            # Stochastic terms (Gaussian white noise)
            dx_stoch = noise_coeff * np.random.normal()
            dy_stoch = noise_coeff * np.random.normal()
            dz_stoch = noise_coeff * np.random.normal()

            x[i] = x_curr + dx_det + dx_stoch
            y[i] = y_curr + dy_det + dy_stoch
            z[i] = z_curr + dz_det + dz_stoch

        return LorenzResult(
            t=t, x=x, y=y, z=z, config=self.config, noise_type=NoiseType.ADDITIVE
        )

    def _solve_multiplicative_noise(
        self,
        t_span: Tuple[float, float],
        dt: float,
        state0: List[float],
        D: float,
    ) -> LorenzResult:
        """
        Solve stochastic Lorenz with state-dependent (multiplicative) noise.

        Stochastic equations:
            dx = \sigma(y - x)dt + x \cdot \sqrt{2D} dW_t^{(x)}
            dy = (x(\rho - z) - y)dt + y \cdot \sqrt{2D} dW_t^{(y)}
            dz = (xy - \beta z)dt + z \cdot \sqrt{2D} dW_t^{(z)}
        """
        t = np.arange(t_span[0], t_span[1], dt)
        n_steps = len(t)

        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        z = np.zeros(n_steps)

        x[0], y[0], z[0] = state0
        noise_coeff = np.sqrt(2.0 * D * dt)

        for i in range(1, n_steps):
            x_curr, y_curr, z_curr = x[i - 1], y[i - 1], z[i - 1]

            # Deterministic terms
            dx_det = self.sigma * (y_curr - x_curr) * dt
            dy_det = (x_curr * (self.rho - z_curr) - y_curr) * dt
            dz_det = (x_curr * y_curr - self.beta * z_curr) * dt

            # Multiplicative noise terms
            dx_stoch = noise_coeff * x_curr * np.random.normal()
            dy_stoch = noise_coeff * y_curr * np.random.normal()
            dz_stoch = noise_coeff * z_curr * np.random.normal()

            x[i] = x_curr + dx_det + dx_stoch
            y[i] = y_curr + dy_det + dy_stoch
            z[i] = z_curr + dz_det + dz_stoch

        return LorenzResult(
            t=t, x=x, y=y, z=z, config=self.config, noise_type=NoiseType.MULTIPLICATIVE
        )

    def _solve_ou_noise(
        self,
        t_span: Tuple[float, float],
        dt: float,
        state0: List[float],
        D: float,
        correlation_time: float,
    ) -> LorenzResult:
        """
        Solve Lorenz system with Ornstein-Uhlenbeck colored noise.

        The OU noise process:
            d\eta_i = -(1/\tau) \eta_i dt + \sqrt{2D/\tau} dW_t

        Autocorrelation: E[\eta(t) \eta(t')] = D \cdot exp(-|t-t'|/\tau)
        Stationary distribution: \eta_i ~ N(0, D)
        """
        t = np.arange(t_span[0], t_span[1], dt)
        n_steps = len(t)

        # System state
        x = np.zeros(n_steps)
        y = np.zeros(n_steps)
        z = np.zeros(n_steps)

        # OU noise processes
        eta_x = np.zeros(n_steps)
        eta_y = np.zeros(n_steps)
        eta_z = np.zeros(n_steps)

        x[0], y[0], z[0] = state0

        # OU parameters
        theta = 1.0 / correlation_time  # Damping coefficient

        for i in range(1, n_steps):
            x_curr, y_curr, z_curr = x[i - 1], y[i - 1], z[i - 1]
            eta_x_curr = eta_x[i - 1]
            eta_y_curr = eta_y[i - 1]
            eta_z_curr = eta_z[i - 1]

            # Update OU noise processes
            sqrt_dt = np.sqrt(dt)
            dW_x = np.random.normal() * sqrt_dt
            dW_y = np.random.normal() * sqrt_dt
            dW_z = np.random.normal() * sqrt_dt

            ou_coeff = np.sqrt(2.0 * D * theta)
            eta_x[i] = eta_x_curr - theta * eta_x_curr * dt + ou_coeff * dW_x
            eta_y[i] = eta_y_curr - theta * eta_y_curr * dt + ou_coeff * dW_y
            eta_z[i] = eta_z_curr - theta * eta_z_curr * dt + ou_coeff * dW_z

            # Lorenz dynamics with colored noise
            dx = (self.sigma * (y_curr - x_curr) + eta_x[i]) * dt
            dy = (x_curr * (self.rho - z_curr) - y_curr + eta_y[i]) * dt
            dz = (x_curr * y_curr - self.beta * z_curr + eta_z[i]) * dt

            x[i] = x_curr + dx
            y[i] = y_curr + dy
            z[i] = z_curr + dz

        return LorenzResult(
            t=t,
            x=x,
            y=y,
            z=z,
            config=self.config,
            noise_type=NoiseType.ORNSTEIN_UHLENBECK,
            eta_x=eta_x,
            eta_y=eta_y,
            eta_z=eta_z,
        )


class LorenzPlotter:
    """
    Visualization utilities for Lorenz system trajectories.

    Supports both static (matplotlib) and interactive (plotly) plots.

    Example:
        >>> plotter = LorenzPlotter(color_gradient=["#CC95C0", "#DBD4B4", "#7AA1D2"])
        >>> plotter.plot_3d(result)
        >>> plotter.plot_2d_projections(result)
    """

    # Default color gradients
    DEFAULT_GRADIENTS = {
        "teal_gold": ["#16A085", "#F4D03F"],
        "purple_pink": ["#5C258D", "#F8A3B8"],
        "pastel": ["#CC95C0", "#DBD4B4", "#7AA1D2"],
        "green_yellow": ["#3CA55C", "#B5AC49"],
        "blue_yellow": ["#3D7EAA", "#FFE47A"],
    }

    def __init__(
        self,
        color_gradient: Optional[List[str]] = None,
        gradient_name: str = "teal_gold",
    ):
        """
        Initialize the plotter.

        Parameters:
            color_gradient: List of hex color codes for gradient
            gradient_name: Name of predefined gradient if color_gradient is None
        """
        if color_gradient is None:
            color_gradient = self.DEFAULT_GRADIENTS.get(
                gradient_name, self.DEFAULT_GRADIENTS["teal_gold"]
            )
        self.color_gradient = color_gradient
        self.cmap = LinearSegmentedColormap.from_list("custom", color_gradient)

    @staticmethod
    def _extract_data(
        data: Union[LorenzResult, Tuple, Any],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract time and coordinates from various data formats."""
        if isinstance(data, LorenzResult):
            return data.t, data.x, data.y, data.z
        elif hasattr(data, "y") and hasattr(data, "t"):
            # scipy solution object
            return data.t, data.y[0], data.y[1], data.y[2]
        elif isinstance(data, (tuple, list)) and len(data) >= 4:
            return data[0], data[1], data[2], data[3]
        else:
            raise ValueError(
                "Data must be LorenzResult, scipy solution, or tuple (t, x, y, z)"
            )

    def plot_3d(
        self,
        data: Union[LorenzResult, Tuple],
        title: str = "3D Lorenz Trajectory",
        view_angle: Tuple[float, float] = (15, 60),
        figsize: Tuple[int, int] = (12, 10),
        linewidth: float = 1.2,
        alpha: float = 0.8,
        show_colorbar: bool = True,
        show_title: bool = True,
        show_axes: bool = False,
        save_path: Optional[str] = None,
        dpi: int = 300,
    ) -> plt.Figure:
        """
        Plot 3D trajectory of the Lorenz system.

        Parameters:
            data: LorenzResult or tuple (t, x, y, z)
            title: Plot title
            view_angle: (elevation, azimuth) for 3D view
            figsize: Figure size
            linewidth: Line width
            alpha: Transparency
            show_colorbar: Show time colorbar
            show_title: Show title
            show_axes: Show axis elements
            save_path: Path to save figure
            dpi: Resolution for saved figure

        Returns:
            matplotlib Figure object
        """
        t, x, y, z = self._extract_data(data)

        fig = plt.figure(figsize=figsize, facecolor="white")
        ax = fig.add_subplot(111, projection="3d")

        # Create line collection with time-based colors
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = Line3DCollection(
            segments, cmap=self.cmap, linewidths=linewidth, alpha=alpha
        )
        lc.set_array(t[:-1])

        line = ax.add_collection3d(lc)

        # Set axis limits
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        ax.set_zlim(z.min(), z.max())

        # Set box aspect
        ax.set_box_aspect([figsize[0] / figsize[1], 1, 1])

        if not show_axes:
            ax.set_axis_off()
        else:
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])

        ax.view_init(elev=view_angle[0], azim=view_angle[1])

        if show_colorbar:
            cbar = plt.colorbar(line, ax=ax, shrink=0.6, aspect=30, pad=0.1)
            cbar.set_label("Time", rotation=270, labelpad=20, fontsize=12)

        if show_title:
            plt.title(title, fontsize=16, pad=20)

        plt.tight_layout(pad=0)

        if save_path:
            plt.savefig(
                save_path, dpi=dpi, bbox_inches="tight", pad_inches=0, facecolor="white"
            )

        plt.show()
        return fig

    def plot_2d_projections(
        self,
        data: Union[LorenzResult, Tuple],
        figsize: Tuple[int, int] = (20, 8),
        linewidth: float = 1.0,
        alpha: float = 0.8,
        show_labels: bool = True,
        show_titles: bool = True,
        show_ticks: bool = False,
        show_colorbar: bool = True,
        show_main_title: bool = True,
        main_title: str = "2D Projections",
        save_path: Optional[str] = None,
        save_subplots_dir: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot 2D projections (X-Y, X-Z, Y-Z) of the Lorenz trajectory.

        Parameters:
            data: LorenzResult or tuple (t, x, y, z)
            figsize: Figure size
            linewidth: Line width
            alpha: Transparency
            show_labels: Show axis labels
            show_titles: Show subplot titles
            show_ticks: Show axis ticks
            show_colorbar: Show time colorbar
            show_main_title: Show main title
            main_title: Main title text
            save_path: Path to save combined figure
            save_subplots_dir: Directory to save individual subplots

        Returns:
            matplotlib Figure object
        """
        t, x, y, z = self._extract_data(data)

        if show_colorbar:
            fig, axes = plt.subplots(
                1,
                3,
                figsize=figsize,
                gridspec_kw={"right": 0.88},
            )
        else:
            fig, axes = plt.subplots(1, 3, figsize=figsize)

        projections = [
            (x, y, "X", "Y", "X-Y Projection"),
            (x, z, "X", "Z", "X-Z Projection"),
            (y, z, "Y", "Z", "Y-Z Projection"),
        ]

        for i, (x_data, y_data, x_label, y_label, title) in enumerate(projections):
            points = np.array([x_data, y_data]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(
                segments, cmap=self.cmap, linewidths=linewidth, alpha=alpha
            )
            lc.set_array(t[:-1])

            axes[i].add_collection(lc)
            axes[i].set_xlim(x_data.min(), x_data.max())
            axes[i].set_ylim(y_data.min(), y_data.max())

            for spine in axes[i].spines.values():
                spine.set_visible(False)

            if show_labels:
                axes[i].set_xlabel(x_label, fontsize=12)
                axes[i].set_ylabel(y_label, fontsize=12)
            if show_titles:
                axes[i].set_title(title, fontsize=14)
            if not show_ticks:
                axes[i].set_xticks([])
                axes[i].set_yticks([])

        if show_main_title:
            plt.suptitle(main_title, fontsize=16, y=1.02)

        fig.subplots_adjust(wspace=0.25)

        if show_colorbar:
            cbar_ax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
            cbar = fig.colorbar(lc, cax=cbar_ax)
            cbar.set_label("Time", rotation=270, labelpad=20, fontsize=12)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        if save_subplots_dir:
            import os

            os.makedirs(save_subplots_dir, exist_ok=True)
            fig.canvas.draw()
            for i, ax in enumerate(axes):
                bbox = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
                    fig.dpi_scale_trans.inverted()
                )
                safe_title = main_title.replace(" ", "_")
                fname = os.path.join(
                    save_subplots_dir, f"{safe_title}_subplot_{i+1}.svg"
                )
                fig.savefig(fname, bbox_inches=bbox, format="svg")

        if save_path is None and save_subplots_dir is None:
            plt.show()

        return fig

    def plot_2d_single(
        self,
        data: Union[LorenzResult, Tuple],
        projection: str = "xy",
        figsize: Tuple[int, int] = (8, 8),
        linewidth: float = 1.0,
        alpha: float = 0.8,
        show_labels: bool = False,
        show_title: bool = False,
        show_ticks: bool = False,
        show_colorbar: bool = False,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot a single 2D projection.

        Parameters:
            data: LorenzResult or tuple (t, x, y, z)
            projection: 'xy', 'xz', or 'yz'
            figsize: Figure size
            linewidth: Line width
            alpha: Transparency
            show_labels: Show axis labels
            show_title: Show title
            show_ticks: Show axis ticks
            show_colorbar: Show colorbar
            title: Custom title
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        t, x, y, z = self._extract_data(data)

        projection_map = {
            "xy": (x, y, "X", "Y", "X-Y Projection"),
            "xz": (x, z, "X", "Z", "X-Z Projection"),
            "yz": (y, z, "Y", "Z", "Y-Z Projection"),
        }

        if projection not in projection_map:
            raise ValueError(f"projection must be one of {list(projection_map.keys())}")

        x_data, y_data, x_label, y_label, default_title = projection_map[projection]
        plot_title = title or default_title

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        points = np.array([x_data, y_data]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=self.cmap, linewidths=linewidth, alpha=alpha)
        lc.set_array(t[:-1])

        ax.add_collection(lc)
        ax.set_xlim(x_data.min(), x_data.max())
        ax.set_ylim(y_data.min(), y_data.max())

        for spine in ax.spines.values():
            spine.set_visible(False)

        if show_labels:
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(y_label, fontsize=12)
        if show_title:
            ax.set_title(plot_title, fontsize=14)
        if not show_ticks:
            ax.set_xticks([])
            ax.set_yticks([])
        if show_colorbar:
            cbar = fig.colorbar(lc, ax=ax, shrink=0.6, aspect=30)
            cbar.set_label("Time", rotation=270, labelpad=20, fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

        return fig

    def plot_multiple_views(
        self,
        data: Union[LorenzResult, Tuple],
        figsize: Tuple[int, int] = (16, 12),
        linewidth: float = 1.0,
        alpha: float = 0.8,
        show_titles: bool = True,
        save_path: Optional[str] = None,
    ) -> plt.Figure:
        """
        Plot 3D trajectory from multiple viewing angles.

        Parameters:
            data: LorenzResult or tuple (t, x, y, z)
            figsize: Figure size
            linewidth: Line width
            alpha: Transparency
            show_titles: Show subplot titles
            save_path: Path to save figure

        Returns:
            matplotlib Figure object
        """
        t, x, y, z = self._extract_data(data)

        viewing_angles = [
            (15, 60, "Classic View"),
            (10, 0, "Side View"),
            (85, 0, "Top View"),
            (25, 120, "Angled View"),
            (5, 270, "Front View"),
            (30, 210, "Perspective View"),
        ]

        fig = plt.figure(figsize=figsize)

        for i, (elev, azim, subtitle) in enumerate(viewing_angles):
            ax = fig.add_subplot(2, 3, i + 1, projection="3d")

            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = Line3DCollection(
                segments, cmap=self.cmap, linewidths=linewidth, alpha=alpha
            )
            lc.set_array(t[:-1])

            ax.add_collection3d(lc)
            ax.set_axis_off()
            ax.view_init(elev=elev, azim=azim)

            if show_titles:
                ax.set_title(subtitle, fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()
        return fig

    def plot_interactive_3d(
        self,
        data: Union[LorenzResult, Tuple],
        title: str = "Interactive 3D Lorenz Attractor",
        color_scale: str = "Viridis",
        width: int = 1000,
        height: int = 800,
    ):
        """
        Create an interactive 3D plot using Plotly.

        Parameters:
            data: LorenzResult or tuple (t, x, y, z)
            title: Plot title
            color_scale: Plotly colorscale name
            width: Figure width
            height: Figure height

        Returns:
            plotly Figure object
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            raise ImportError(
                "plotly is required for interactive plots. "
                "Install with: pip install plotly"
            )

        t, x, y, z = self._extract_data(data)
        colors = np.linspace(0, 1, len(t))

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(
                        color=colors,
                        colorscale=color_scale,
                        width=2,
                        colorbar=dict(title="Time", thickness=20, len=0.7),
                    ),
                    hovertemplate=(
                        "<b>X</b>: %{x:.2f}<br>"
                        "<b>Y</b>: %{y:.2f}<br>"
                        "<b>Z</b>: %{z:.2f}<extra></extra>"
                    ),
                )
            ]
        )

        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor="center"),
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.3)),
                aspectmode="cube",
            ),
            width=width,
            height=height,
            showlegend=False,
        )

        return fig


# Convenience functions for quick usage
def quick_solve(
    noise_type: str = "none", config_index: int = 0, **kwargs
) -> LorenzResult:
    """
    Quick function to solve Lorenz system.

    Parameters:
        noise_type: "none", "additive", "multiplicative", or "ou"
        config_index: Preset configuration index (0-5)
        **kwargs: Additional arguments passed to LorenzSystem.solve()

    Returns:
        LorenzResult
    """
    noise_map = {
        "none": NoiseType.NONE,
        "additive": NoiseType.ADDITIVE,
        "multiplicative": NoiseType.MULTIPLICATIVE,
        "ou": NoiseType.ORNSTEIN_UHLENBECK,
    }

    config = LorenzConfig.from_preset(config_index)
    lorenz = LorenzSystem(config)
    return lorenz.solve(noise_type=noise_map[noise_type], **kwargs)


def quick_plot(result: LorenzResult, plot_type: str = "3d", **kwargs) -> plt.Figure:
    """
    Quick function to plot Lorenz results.

    Parameters:
        result: LorenzResult from solve
        plot_type: "3d", "2d", "multi", or "interactive"
        **kwargs: Additional arguments passed to plot function

    Returns:
        Figure object
    """
    plotter = LorenzPlotter()

    if plot_type == "3d":
        return plotter.plot_3d(result, **kwargs)
    elif plot_type == "2d":
        return plotter.plot_2d_projections(result, **kwargs)
    elif plot_type == "multi":
        return plotter.plot_multiple_views(result, **kwargs)
    elif plot_type == "interactive":
        return plotter.plot_interactive_3d(result, **kwargs)
    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")


if __name__ == "__main__":
    # Example usage — reproduces the blog post figures.
    # Parameters match the interactive Plotly plots in the post:
    #   dt = 0.005, 9000 steps  =>  t_span = (0, 45)
    print("Lorenz System Module Demo")
    print("=" * 50)

    # Create system with classic parameters (sigma=10, rho=28, beta=8/3)
    lorenz = LorenzSystem()
    plotter = LorenzPlotter(gradient_name="teal_gold")

    # ── 1. Deterministic ────────────────────────────────────────────────
    print("\n1. Solving deterministic Lorenz system...")
    result_det = lorenz.solve(t_span=(0, 45), n_points=9000)

    plotter.plot_3d(result_det, title="Deterministic Lorenz Attractor")
    plotter.plot_2d_projections(
        result_det, main_title="Deterministic Lorenz — 2D Projections"
    )

    # ── 2. Additive noise (D = 3.0) ────────────────────────────────────
    print("2. Solving with additive noise (D = 3.0)...")
    result_add = lorenz.solve(
        noise_type=NoiseType.ADDITIVE,
        D=3.0,
        t_span=(0, 45),
        dt=0.005,
    )

    plotter.plot_3d(result_add, title="Additive Noise (D = 3.0)")
    plotter.plot_2d_projections(
        result_add, main_title="Additive Noise — 2D Projections"
    )

    # ── 3. Multiplicative noise (D = 0.02) ─────────────────────────────
    print("3. Solving with multiplicative noise (D = 0.02)...")
    result_mult = lorenz.solve(
        noise_type=NoiseType.MULTIPLICATIVE,
        D=0.02,
        t_span=(0, 45),
        dt=0.005,
    )

    plotter.plot_3d(result_mult, title="Multiplicative Noise (D = 0.02)")
    plotter.plot_2d_projections(
        result_mult, main_title="Multiplicative Noise — 2D Projections"
    )

    # ── 4. Ornstein–Uhlenbeck noise (D = 5.0, tau = 5.0) ──────────────
    print("4. Solving with Ornstein-Uhlenbeck noise (D = 5.0, tau = 5.0)...")
    result_ou = lorenz.solve(
        noise_type=NoiseType.ORNSTEIN_UHLENBECK,
        D=5.0,
        correlation_time=5.0,
        t_span=(0, 45),
        dt=0.005,
    )

    plotter.plot_3d(result_ou, title="OU Colored Noise (D = 5.0, tau = 5.0)")
    plotter.plot_2d_projections(
        result_ou, main_title="OU Colored Noise — 2D Projections"
    )

    print("\nDone! Use LorenzSystem and LorenzPlotter classes for custom simulations.")
