import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from timeit import timeit
from diffrax import diffeqsolve, Dopri8, ODETerm, SaveAt, PIDController

def pairwise_acceleration(position, mass, eps=1e-12):
    distance_squared = -2 * position @ position.T
    diag = -0.5 * jnp.einsum("ii->i", distance_squared)  # points to the diagonal inplace
    distance_squared += diag + diag[:, None]

    return jnp.sum(
        (position[:, None, :] - position) * (distance_squared[..., None] + eps)**-1.5,
        axis=0
    )

def vector_field(_t, y, args):
    position, velocity = y
    acceleration = pairwise_acceleration(position, args[0])
    return (velocity, acceleration)

@jax.jit
def simulate(time, position, velocity, mass):
    term = ODETerm(vector_field)
    solver = Dopri8()
    saveat = SaveAt(ts=time)
    # saveat = SaveAt(steps=True)

    rtol = atol = 1e-12
    stepsize_controller = PIDController(rtol=rtol, atol=atol)

    results = diffeqsolve(
        term,
        solver,
        t0=time.min(),
        t1=time.max(),
        dt0=None,
        y0=(position, velocity),
        args=(mass,),
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=None,
    )

    return results.ys

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

def main():
    num_steps = 1000
    num_points = 3
    num_dim = 2

    time = jnp.linspace(0, 1, num_steps)
    mass = jnp.ones(num_points)

    rng = np.random.default_rng(seed=0)
    init_position, init_velocity = rng.standard_normal((2, num_points, num_dim))

    position, velocity = simulate(time, init_position, init_velocity, mass)

    plot(position, velocity)
    plt.show()

if __name__ == "__main__":
    main()
