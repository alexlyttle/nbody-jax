import jax

_ENABLE_X64 = True
jax.config.update("jax_enable_x64", _ENABLE_X64)

import diffrax
import jax.numpy as jnp

from typing import Optional
from jax.typing import ArrayLike
from solution import Solution

_FLOAT_DTYPE = jnp.float64 if _ENABLE_X64 else jnp.float32
# _DEFAULT_EPS = jnp.finfo(_FLOAT_DTYPE).eps**0.5
_DEFAULT_EPS = 1e-12

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
