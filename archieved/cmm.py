import numpy as np
import matplotlib.pyplot as plt

# Simplified 2D centroidal dynamics model
class Quadruped2D:
    def __init__(self, mass, inertia):
        self.mass = mass
        self.inertia = inertia
        self.position = np.array([0, 0])  # Initial position
        self.velocity = np.array([0, 0])  # Initial velocity
        self.foot_positions = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])  # Simplified foot positions

    def update(self, forces):
        # Simplified dynamics update
        acceleration = np.sum(forces, axis=0) / self.mass
        self.velocity = self.velocity.astype(np.float64)
        self.velocity += acceleration * 0.01
        self.position += self.velocity * 0.01

    def visualize(self):
        plt.figure()
        plt.scatter(self.position[0], self.position[1], label='Robot CoM')
        for foot in self.foot_positions:
            plt.scatter(foot[0] + self.position[0], foot[1] + self.position[1], label='Foot')
        plt.legend()
        plt.show()

# Example usage
quad = Quadruped2D(mass=10, inertia=1)
forces = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])  # Example forces

for _ in range(100):
    quad.update(forces)
    quad.visualize()
