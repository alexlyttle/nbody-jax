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
from matplotlib.collections import LineCollection
from jax.typing import PyTree, ArrayLike

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
def vector_field(_t, y: PyTree, _args) -> PyTree:
    """The vector field for the `diffrax.diffeqsolve`.
    
    Args:
        _t: The time (unused).
        y: A tuple or list of the particle positions and velocities.
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
        max_steps: int=1000000, 
        times: Optional[ArrayLike]=None, 
        show_progress: bool=False
    ) -> diffrax.Solution:
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
        "steps": diffrax.SubSaveAt(steps=True),
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

    return solution

def create_animation(
        position: jnp.ndarray,
        frame_rate: float=60.0,
        padding: float=0.05,
        memory: Optional[int]=None
    ) -> animation.FuncAnimation:
    """Create an animation of the particle positions.
    
    Args:
        position: The position of particles.
        frame_rate: The frame rate of the animation.
        padding: The axes padding as a fraction of the position range.
        memory: The number of frames to remember (defaults to all frames).
    
    Returns:
        The animation.
    """
    
    num_frames = position.shape[0]
    num_points = position.shape[1]

    if memory is None:
        memory = num_frames

    position_min = position.min(axis=(0, 1))
    position_max = position.max(axis=(0, 1))
    position_range = position_max - position_min
    position_padding = padding * position_range

    limits = np.stack(
        [position_min - position_padding, position_max + position_padding],
        axis=-1
    )

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    num_colors = len(colors)

    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("position")
    
    artists = []
    for k in range(num_points):
        artists.append(ax.plot([], [], c=colors[k%num_colors])[0])

    def init():
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        return artists

    def update(i):
        s = max(0, i-memory)
        for j in range(num_points):
            artists[j].set_data(position[s:i, j, 0], position[s:i, j, 1])
        return artists
    
    ani = animation.FuncAnimation(
        fig,
        update,
        num_frames,
        interval=1000//frame_rate,
        init_func=init,
        blit=True,
    )
    return ani

def plot(position, velocity):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for i in range(position.shape[1]):
        axes[0].plot(position[:, i, 0], position[:, i, 1])
        axes[1].plot(velocity[:, i, 0], velocity[:, i, 1])
    
    ax = axes[0]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("position")
    ax.set_aspect("equal")
    
    ax = axes[1]
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("velocity")
    ax.set_aspect("equal")
    return fig, axes

@click.command()
@click.argument("num-points", type=int)
@click.option("-d", "--num-dim", default=2, type=int, help="Number of dimensions.", show_default=True)
@click.option("--seed", default=0, type=int, help="Random seed.", show_default=True)
@click.option("--duration", default=1.0, type=float, help="Simulation duration.", show_default=True)
@click.option("--show-progress", is_flag=True, help="Show progress bar.", show_default=True)
@click.option("--animate", is_flag=True, help="Animate the simulation.", show_default=True)
@click.option("--animation-duration", default=5.0, type=float, help="Animation duration (seconds).", show_default=True)
@click.option("--animation-fps", default=30.0, type=float, help="Animation frame rate (frames per second).",
              show_default=True)
@click.option("--animation-filename", type=click.Path(dir_okay=False, writable=True), help="Save animation to given filename.",
              show_default=True)
def cli(
    num_points: int,
    num_dim: int,
    seed: int,
    duration: float,
    show_progress: bool,
    animate: bool,
    animation_duration: float,
    animation_fps: float,
    animation_filename: str,
) -> None:
    """CLI for the n-body simulation.
    
    Args:
        num_points: The number of particles.
        num_dim: The number of dimensions.
        seed: The random seed.
        duration: The simulation duration.
        show_progress: Show progress bar.
        animate: Animate the simulation.
        animation_duration: The animation duration.
        animation_fps: The animation frame rate.
        animation_filename: The animation filename.
    """

    rng = np.random.default_rng(seed=seed)
    init_position, init_velocity = rng.standard_normal((2, num_points, num_dim))

    times = None
    if animate:
        num_frames = int(animation_duration * animation_fps)
        times = jnp.linspace(0, duration, num_frames)        

    start = time()
    solution = simulate(0.0, duration, init_position, init_velocity, times=times, show_progress=show_progress)
    elapsed = time() - start

    print(f"Completed in {elapsed:.3f} seconds.")
    print("Solution statistics:")
    pprint(solution.stats)

    if animate:
        position = solution.ys["times"][0]
        ani = create_animation(position, frame_rate=animation_fps)
        plt.show()

        if animation_filename is not None:
            writer = animation.PillowWriter(
                fps=animation_fps,
            )
            ani.save(animation_filename, writer=writer)

if __name__ == "__main__":
    cli()
