import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from timeit import timeit

def pairwise_acceleration(X, eps=1e-12):
    distance_squared = -2 * X @ X.T
    diag = -0.5 * np.einsum("ii->i", distance_squared)  # points to the diagonal inplace
    distance_squared += diag + diag[:, None]

    acceleration = np.sum(
        (X[:, None, :] - X) * (distance_squared[..., None] + eps)**-1.5,
        axis=0
    )
    return acceleration

def nbody(t, u, num_dim):

    num_points = u.shape[0] // (2 * num_dim)
    size = num_dim * num_points
    
    x, dx = u[:size], u[size:]

    ddx = pairwise_acceleration(x.reshape(num_points, num_dim))

    du = np.zeros_like(u)
    du[:size] = dx
    du[size:] = ddx.ravel()

    return du

def simulate(t, x, dx): 

    num_points, num_dim = x.shape
    size = num_points * num_dim
    u = np.concatenate([x.flatten(), dx.flatten()])

    time_span = (t.min(), t.max())
    t_eval = t
    atol = rtol = 1e-12

    result = solve_ivp(
        lambda t, u: nbody(t, u, num_dim),
        time_span,
        u,
        method="DOP853",
        t_eval=t_eval,
        atol=atol,
        rtol=rtol,
    )

    x = result.y[:size].reshape((num_points, num_dim, -1))
    dx = result.y[size:].reshape((num_points, num_dim, -1))

    return x, dx

def plot(x, dx):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for i in range(x.shape[0]):
        axes[0].plot(x[i, 0], x[i, 1])
        axes[1].plot(dx[i, 0], dx[i, 1])
    
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

    t = np.linspace(0, 1, num_steps)

    rng = np.random.default_rng(seed=0)
    init_position, init_velocity = rng.standard_normal((2, num_points, num_dim))

    position, velocity = simulate(t, init_position, init_velocity)
    
    plot(position, velocity)
    plt.show()

if __name__ == "__main__":
    main()
