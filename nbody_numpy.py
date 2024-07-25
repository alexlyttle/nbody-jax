import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import timeit
from numpy.typing import ArrayLike
from typing import Optional
from solution import Solution

_DEFAULT_EPS = np.finfo(np.float64).eps**0.5

def pairwise_acceleration(position, eps=_DEFAULT_EPS):
    distance_squared = -2 * position @ position.T
    diag = -0.5 * np.einsum("ii->i", distance_squared)  # points to the diagonal inplace
    distance_squared += diag + diag[:, None]

    acceleration = np.sum(
        (position[:, None, :] - position) * (distance_squared[..., None] + eps)**-1.5,
        axis=0
    )
    return acceleration

def vector_field(_t, u, num_dim):
    num_particles = u.shape[0] // (2 * num_dim)
    size = num_dim * num_particles
    
    position, velocity = u[:size], u[size:]

    acceleration = pairwise_acceleration(position.reshape(num_particles, num_dim))

    du = np.zeros_like(u)
    du[:size] = velocity
    du[size:] = acceleration.ravel()

    return du

def simulate(
        start_time: float,
        end_time: float,
        position: ArrayLike,
        velocity: ArrayLike,
        max_steps: Optional[int]=None,
        times: Optional[ArrayLike]=None,
        show_progress: bool=False
    ) -> Solution: 

    num_particles, num_dim = position.shape
    size = num_particles * num_dim
    u = np.concatenate([position.flatten(), velocity.flatten()])

    time_span = (start_time, end_time)
    atol = 1e-9
    rtol = 1e-7

    solution = solve_ivp(
        vector_field,
        time_span,
        u,
        method="DOP853",
        t_eval=times,
        atol=atol,
        rtol=rtol,
        args=(num_dim,),
    )

    position = solution.y[:size].T.reshape((-1, num_particles, num_dim))
    velocity = solution.y[size:].T.reshape((-1, num_particles, num_dim))

    return Solution(
        position,
        velocity,
        position.shape[0] if times is None else "N/A",
    )
