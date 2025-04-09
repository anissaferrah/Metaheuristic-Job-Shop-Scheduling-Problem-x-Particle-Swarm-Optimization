"""
This code is part of a Job Shop Scheduling Problem (JSP) solution using Particle Swarm Optimization (PSO).
It includes classe for representing a particle in the PSO algorithm.
"""
import numpy as np
class Particle:
    """
    A class representing a particle in a Particle Swarm Optimization (PSO) algorithm.
    Each particle has a position, velocity, and a best-known position.
    The purpose is to solve JSP (Job Shop Scheduling Problem) using PSO.
    The particle's position is represented as a random keys encoding of the solution.
    """
    def __init__(self,solver,index=0):
        self.index = index
        self.solver = solver
        self.position = np.random.uniform(size=self.solver.instance.num_operations)
        self.personal_best = self.position.copy()
        self.velocity = np.zeros(self.solver.instance.num_operations)
        self.schedule = None
        self.current_fitness= float('inf')
        self.personal_best_fitness = float('inf')
        self.perform_crossover = False
        self.update_schedule()
        self.update_current_fitness()
    
    def update_schedule(self):
        self.schedule=self.solver.decode_position(self.position)
    
    def update_current_fitness(self):
        self.current_fitness=self.fitness()
    

    def fitness(self):
        # Calculate the fitness of the particle's position
        return self.solver.fitness(self.schedule)
    
    def update_personal_best(self):
        if self.current_fitness < self.personal_best_fitness:
            self.best_position = self.position.copy()
            self.personal_best_fitness = self.current_fitness
    
    def update_global_best(self):
        # Update the global best position if the current position is better
        if self.current_fitness < self.solver.global_best_fitness:
            self.solver.global_best = self.copy()

    def update_local_best(self):
        k=self.solver.neighberhood_size
        K=self.solver.population_size
        # Get the neighborhood of the particle
        neighborhood = []
        i=(self.index-(k-1)//2) % K
        while i%K <= (self.index+(k-1)//2) % K:
            neighborhood.append(self.solver.particles[i])
            i+=1
        self.local_best=min(neighborhood, key=lambda p: p.personal_best_fitness).position

    def update_near_neighbor_best(self):
        def FDR(j,d):
            # Calculate the Fitness Ddistance Ratio value between this particle and the i-th particle for a given dimension d
            if self.index!=j:
                return (self.current_fitness-self.solver.particles[j].personal_best_fitness)/abs(self.solver.particles[j].personal_best[d]-self.position[d])
            else:
                return -1
        # Find the near neighbor with the best fitness distance ratio
        self.near_neighbor_best = np.zeros(self.solver.instance.num_operations)
        for d in range(self.solver.instance.num_operations):
            self.near_neighbor_best[d] = sorted(self.solver.particles, key=lambda p: FDR(p.index,d),reverse=True).pop(0).personal_best[d]
        
    def update_velocity(self):
        # Determine whether or not to do crossover
        self.perform_crossover=np.random.uniform() < self.solver.crossover_probability
        if not self.perform_crossover:
            # Update the velocity based on personal best, global best, and local best
            rp = np.random.uniform()
            rg = np.random.uniform()
            rl = np.random.uniform()
            rn = np.random.uniform()

            self.velocity = (self.solver.weight * self.velocity +
                             self.solver.cpersonal * rp * (self.personal_best - self.position) +
                             self.solver.cglobal * rg * (self.solver.global_best.position - self.position) +
                             self.solver.clocal * rl * (self.local_best - self.position) +
                             self.solver.cneighbor * rn * (self.near_neighbor_best - self.position))
            self.velocity = np.clip(self.velocity, -self.solver.vmax, self.solver.vmax)
        
    def update_position(self):
        # Update the position of the particle based on its velocity
        # Determine whether to update the position according to the new velocity or to perform crossover
        if self.perform_crossover:
            # Determine whether to perform the crossover or to keep the current position
            if np.random.uniform() > self.solver.pu:
                self.position=self.solver.global_best_position
        else:
            self.position+=self.velocity    
        self.update_schedule()
        self.update_current_fitness()