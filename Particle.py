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
    def __init__(self, index=0,position=np.random.uniform(), velocity=0,solver=JSP_PSO_Solver()):
        self.index = index
        self.solver = solver
        self.position = position
        self.velocity = velocity
        self.current_fitness=self.fitness()
        self.personal_best = position.copy()
        self.best_fitness = self.current_fitness
        self.schedule = solver.decode_position(position)

    def fitness(self):
        # Calculate the fitness of the particle's position
        return self.solver.fitness(self.schedule)

    def update_position(self):
        self.position += self.velocity
        # Ensure the particle stays within bounds
        self.position = np.clip(self.position, 0, 1)
        # Update the current fitness
        self.current_fitness=self.fitness()
    
    def update_schedule(self):
        self.schedule=self.solver.decode_position(self.position)
    
    def update_personal_best(self):
        if self.current_fitness < self.best_fitness:
            self.best_position = self.position.copy()
            self.best_fitness = self.current_fitness
    
    def update_global_best(self):
        # Update the global best position if the current position is better
        if self.current_fitness < self.solver.global_best_fitness:
            self.solver.global_best_position = self.position.copy()
            self.solver.global_best_fitness = self.current_fitness

    def update_local_best(self):
        k=self.solver.neighberhood_size
        K=self.solver.population_size
        # Get the neighborhood of the particle
        neighborhood = []
        for i in range(self.index-k//2, self.index+k//2+1):
            if i>=0 and i<K:
                neighborhood.append(self.solver.particles[i])

    
    