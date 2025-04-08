"""
This code is part of a Job Shop Scheduling Problem (JSP) solution using Particle Swarm Optimization (PSO).
It includes classe for representing a particle in the PSO algorithm.
"""
import numpy as np
from GLN_PSOc import JSP_PSO_Solver
class Particle:
    """
    A class representing a particle in a Particle Swarm Optimization (PSO) algorithm.
    Each particle has a position, velocity, and a best-known position.
    The purpose is to solve JSP (Job Shop Scheduling Problem) using PSO.
    The particle's position is represented as a random keys encoding of the solution.
    """
    def __init__(self, position, velocity):
        self.position = np.random.uniform()
        self.velocity = 0
        self.best_position = position.copy()
        self.best_fitness = self.fitness()
        self.schedule = JSP_PSO_Solver.decode_position(position)

    def fitness(self):
        # Calculate the fitness of the particle's position
        # The fitness function should be defined based on the specific problem instance
        # For example, it could be the makespan of the schedule generated from the position
        return self.schedule.calculate_makespan()

    def update_position(self, ):
        self.position += self.velocity
        # Ensure the particle stays within bounds
        self.position = np.clip(self.position, 0, 1)
        # Update the best position if the current position is better
        self.update_best_position()
    
    def update_best_position(self):
        if self.fitness() < self.fitness(self.best_position):
            self.best_position = self.position.copy()
    
    