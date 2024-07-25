import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from typing import Optional
from numpy.typing import ArrayLike

def create_animation(
        position: ArrayLike,
        frame_rate: float=60.0,
        padding: float=0.05,
        axes: Optional[tuple]=None,
        memory: Optional[int]=None
    ) -> animation.FuncAnimation:
    """Create an animation of the particle positions.

    Args:
        position: The position of particles.
        frame_rate: The frame rate of the animation.
        padding: The axes padding as a fraction of the position range.
        axes: The particle position axes to plot (defaults to (0, 1)).
        memory: The number of frames to remember (defaults to all frames).

    Returns:
        The animation.
    """
    position = np.asarray(position)

    if axes is None:
        axes = (0, 1)

    assert len(axes) == 2, "axes must have length 2."

    num_frames = position.shape[0]
    num_particles = position.shape[1]

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
    ax.set_xlabel(f"x{axes[0]}")
    ax.set_ylabel(f"x{axes[1]}")
    ax.set_title("position")

    artists = []
    for k in range(num_particles):
        artists.append(ax.plot([], [], c=colors[k%num_colors])[0])

    def init():
        ax.set_xlim(limits[axes[0]])
        ax.set_ylim(limits[axes[1]])
        return artists

    def update(i):
        s = max(0, i-memory)
        for j in range(num_particles):
            artists[j].set_data(position[s:i, j, axes[0]], position[s:i, j, axes[1]])
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
