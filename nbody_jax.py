import jax

_ENABLE_X64 = True
jax.config.update("jax_enable_x64", _ENABLE_X64)

import click
import diffrax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from typing import Optional
from pprint import pprint
from time import time
from jax.typing import ArrayLike
from animate import create_animation
from solution import Solution

_FLOAT_DTYPE = jnp.float64 if _ENABLE_X64 else jnp.float32
_DEFAULT_EPS = jnp.finfo(_FLOAT_DTYPE).eps**0.5


def pairwise_acceleration(position: jnp.ndarray, eps: float=_DEFAULT_EPS) -> jnp.ndarray:
    """Calculates the pairwise acceleration between particles.

    Args:
        position: The position of particles, must have shape (N,) where N is the number of particles.
        eps: The softening parameter to avoid singularities.

    Returns:
        The pairwise acceleration between particles.
    """
    distance_squared = -2 * position @ position.T
    diag = -0.5 * jnp.einsum("ii->i", distance_squared)  # points to the diagonal inplace
    distance_squared += diag + diag[:, None]

    return jnp.sum(
        (position[:, None, :] - position) * (distance_squared[..., None] + eps)**-1.5,
        axis=0
    )

@jax.jit
def vector_field(_t, y: tuple[jnp.ndarray], _args) -> tuple[jnp.ndarray]:
    """The vector field for the `diffrax.diffeqsolve`.

    Args:
        _t: The time (unused).
        y: The particle positions and velocities.
        _args: Additional arguments (unused).

    Returns:
        A tuple of the particle velocities and accelerations.
    """
    position, velocity = y
    acceleration = pairwise_acceleration(position)
    return (velocity, acceleration)

def simulate(
        start_time: float,
        end_time: float,
        position: ArrayLike,
        velocity: ArrayLike,
        max_steps: Optional[int]=None,
        times: Optional[ArrayLike]=None,
        show_progress: bool=False
    ) -> Solution:
    """Simulate the system of particles.

    Args:
        start_time: The start time of the simulation.
        end_time: The end time of the simulation.
        position: The initial position of particles.
        velocity: The initial velocity of particles.
        max_steps: The maximum number of steps.
        times: The times at which to save the solution.
        progress_meter: The progress meter.

    Returns:
        The solution of the simulation.
    """
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Dopri8()

    subs = {
        "end": diffrax.SubSaveAt(t1=True),
        # "steps": diffrax.SubSaveAt(steps=True),  # requires max steps
    }

    if times is not None:
        subs["times"] = diffrax.SubSaveAt(ts=times)

    saveat = diffrax.SaveAt(subs=subs)

    rtol = 1e-7
    atol = 1e-9
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    progress_meter = diffrax.NoProgressMeter()
    if show_progress:
        progress_meter = diffrax.TqdmProgressMeter(refresh_steps=100)

    solution = diffrax.diffeqsolve(
        term,
        solver,
        t0=start_time,
        t1=end_time,
        dt0=None,
        y0=(position, velocity),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
        progress_meter=progress_meter,
    )

    num_steps = solution.stats["num_steps"]
    if times is None:
        return Solution(
            solution.ys["end"][0],
            solution.ys["end"][1],
            num_steps,
        )
    return Solution(
        solution.ys["times"][0],
        solution.ys["times"][1],
        num_steps,
    )

def filename_formatter(filename: str, num_particles: int, num_dim: int, axes: tuple) -> str:
    """Format the filename with the given parameters.

    Args:
        filename: The filename.
        num_particles: The number of particles (replaces '%N').
        num_dim: The number of dimensions (replaces '%D').
        axes: The particle position axes to plot (replaces '%A').

    Returns:
        The formatted filename.
    """
    return filename.replace("%N", str(num_particles)).replace("%D", str(num_dim)).replace("%A", f"{axes[0]}-{axes[1]}")

@click.command()
@click.argument("num-particles", type=int)
@click.option("-d", "--num-dim", default=2, type=int, help="Number of dimensions.", show_default=True)
@click.option("--seed", default=0, type=int, help="Random seed.", show_default=True)
@click.option("--duration", default=1.0, type=float, help="Simulation duration.", show_default=True)
@click.option("--max-steps", type=int, help="Maximum number of steps.", show_default=True)
@click.option("--show-progress", is_flag=True, help="Show progress bar.", show_default=True)
@click.option("--show-animation", is_flag=True, help="Show animation of simulation.", show_default=True)
@click.option("--save-animation", is_flag=True, help="Save animation to file.", show_default=True)
@click.option("--animation-duration", default=5.0, type=float, help="Animation duration (seconds).", show_default=True)
@click.option("--animation-fps", default=30.0, type=float, help="Animation frame rate (frames per second).",
              show_default=True)
@click.option("-a", "--animation-axes", default=[(0, 1)], type=(int, int), help="The particle position axes to plot.",
              show_default=True, multiple=True)
@click.option("--animation-filename", default="nbody_%N-%D_%A.gif", type=str, help="Save animation to given filename.",
              show_default=True)
def cli(
    num_particles: int,
    num_dim: int,
    seed: int,
    duration: float,
    max_steps: Optional[int],
    show_progress: bool,
    show_animation: bool,
    save_animation: bool,
    animation_duration: float,
    animation_fps: float,
    animation_axes: tuple,
    animation_filename: str,
) -> None:
    """Run an N-body simulation using JAX.

    For example, to simulate 20 particles, run the following command:

    python nbody_jax.py 20
    \f

    Args:
        num_particles: The number of particles.
        num_dim: The number of dimensions.
        seed: The random seed.
        duration: The simulation duration.
        show_progress: Show progress bar.
        show_animation: AShow animation of simulation.
        save_animation: Save the animation to file.
        animation_duration: The animation duration.
        animation_fps: The animation frame rate.
        animation_axes: The particle position axes to plot.
        animation_filename: The animation filename.
    """
    animate = show_animation or save_animation
    # Validation
    if animate:
        max_axis = max(max(axes) for axes in animation_axes)
        if max_axis >= num_dim:
            raise ValueError(
                f"Animation axes contains axis value: {max_axis} which is greater or equal to the number of dimensions: {num_dim}."
            )

        min_axis = min(min(axes) for axes in animation_axes)
        if min_axis < 0:
            raise ValueError(
                f"Animation axes contains axis value: {min_axis} which is less than zero."
            )

    # Initialise positions and velocities
    rng = np.random.default_rng(seed=seed)
    init_position, init_velocity = rng.standard_normal((2, num_particles, num_dim))

    # Initialise times to save the solution
    times = None
    if animate:
        num_frames = int(animation_duration * animation_fps)
        times = np.linspace(0, duration, num_frames)

    # Run simulation
    start = time()
    solution = simulate(0.0, duration, init_position, init_velocity, times=times, show_progress=show_progress,
                        max_steps=max_steps)
    elapsed = time() - start

    # Print solution statistics
    print(f"Completed in {elapsed:.3f} seconds with {solution.num_steps} function evaluations.")

    # Show or save animation
    if animate:
        writer = "pillow"
        for axes in animation_axes:
            ani = create_animation(solution.position, frame_rate=animation_fps, axes=axes)

            if show_animation:
                plt.show()

            if save_animation:
                filename = filename_formatter(animation_filename, num_particles, num_dim, axes)
                ani.save(filename, writer=writer, fps=animation_fps)

if __name__ == "__main__":
    cli()
