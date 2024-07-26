import numpy as np

def kinetic_energy(velocity):
    """Calculate the kinetic energy of the system.

    Args:
        velocity: The velocity of the particles (N, D).
    
    """
    return 0.5 * np.sum(velocity**2, axis=(-1, -2))

def potential_energy(position):
    """Calculate the potential energy of the system.

    Args:
        position: The position of the particles (N, D).

    """
    num_particles = position.shape[0]
    # num_dim = position.shape[1]
    energy = 0.0
    for i in range(num_particles):
        for j in range(num_particles):
            if i == j:
                continue
            distance = np.linalg.norm(position[i] - position[j])
            energy += 1.0 / distance
    return -0.5 * energy