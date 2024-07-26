from nbody_jax import simulate as simulate_jax
from nbody_numpy import simulate as simulate_numpy

import click
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from time import time
from animate import create_animation
from energy import kinetic_energy, potential_energy

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
@click.option("--xla", is_flag=True, help="Enable XLA.", show_default=True)
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
    xla: bool,
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
    """Run an N-body simulation.

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

    if xla:
        simulate = simulate_jax
    else:
        simulate = simulate_numpy

    init_T = kinetic_energy(init_velocity)
    init_V = potential_energy(init_position)
    init_energy = init_T + init_V
    print(f"Initial energy of the system: {init_energy:.3f}")

    # Run simulation
    start = time()
    solution = simulate(0.0, duration, init_position, init_velocity, times=times, show_progress=show_progress,
                        max_steps=max_steps)
    elapsed = time() - start

    # Print solution statistics
    print(f"Completed in {elapsed:.3f} seconds with {solution.num_steps} solver steps.")

    T = kinetic_energy(solution.velocity[-1])
    V = potential_energy(solution.position[-1])
    final_energy = T + V
    print(f"Final total energy of the system: {final_energy:.3f}")
    print(f"Change in energy: {final_energy - init_energy:.3e}")

    # Show or save animation
    if animate:
        for axes in animation_axes:
            ani = create_animation(solution.position, frame_rate=animation_fps, axes=axes)

            if show_animation:
                plt.show()

            if save_animation:
                filename = filename_formatter(animation_filename, num_particles, num_dim, axes)
                ani.save(filename, writer="pillow", fps=animation_fps)

if __name__ == "__main__":
    cli()
