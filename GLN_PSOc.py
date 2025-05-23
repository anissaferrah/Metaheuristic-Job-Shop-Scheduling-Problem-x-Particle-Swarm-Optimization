from Particle import Particle
import numpy as np
class JSP_PSO_Solver:
    """
    A class to solve the Job Shop Scheduling Problem (JSP) using Particle Swarm Optimization (PSO).
    The class includes methods for initializing the particles, updating their positions, and calculating fitness.
    """
    def __init__(self, instance, population_size=40,neighberhood_size=7, max_iteration=100,weight=0.9,slope=-5/9990,intercept=8996/9990,min_weight=0.1,cpersonal=0.5,cglobal=0.5,clocal=1.5,cneighbor=1.5,vmax=0.25,crossover_probability=0.3,pu=0.7,delta=0):
        self.instance = instance
        self.population_size = population_size
        self.neighberhood_size = neighberhood_size
        self.max_iterations = max_iteration
        self.initial_weight = weight
        self.weight = weight  # Inertia weight
        self.slope = slope # Slope value for the inertia weight
        self.intercept = intercept # Intercept value for the inertia weight
        self.min_weight = min_weight # Minimum weight for the inertia weight
        self.cpersonal = cpersonal  # Personal best weight
        self.cglobal = cglobal # Global best weight
        self.clocal = clocal # Local best weight
        self.cneighbor = cneighbor # Near neighborhood best weight
        self.vmax = vmax # Maximum velocity
        self.crossover_probability = crossover_probability # Crossover probability
        self.pu = pu # Probability of keeping the same position in the next iteration and not setting it to pgbest
        self.delta = delta # Delta value for the scheduling algorithm
        self.particles = [self.initialize_particle(i) for i in range(population_size)]
        self.global_best = self.particles[0]  # Initialize the global best particle

    def decode_position(self, position):
        # Decode the position into a schedule
        # Get the operation-based permutation from the position
        obpermutation = self.get_operation_based_permutation(position)
        # Convert the operation-based permutation into a priority order
        priority_order = self.obpermuation_to_priority_order(obpermutation)
        # Generate the schedule based on the priority order
        schedule = self.generate_schedule(priority_order)
        # Return the schedule
        return schedule
    
    def get_operation_based_permutation(self, position):
        # Sort the indices based on the position values
        sorted_indices = np.argsort(position)
        # Group the sorted indices into jobs
        groups = [sorted_indices[i*self.instance.num_machines:(i+1)*self.instance.num_machines] 
                  for i in range(self.instance.num_jobs)]
        # Create the operation-based permutation
        permutation = np.zeros(len(position), dtype=int)
        for job, group in enumerate(groups):
            for idx in group:
                permutation[idx] = job

        return permutation

    """
    Convert the operation-based permutation into a priority order
    The priority order is a dictionary where the keys are tuples (job, operation) and the values are the priority of that operation
    The priority order is used to determine the order in which operations are scheduled
    """
    def obpermuation_to_priority_order(self, obpermutation):
        priority_order = {}
        job_operations_counter= {j:0 for j in range(self.instance.num_jobs)}
        for priority , job in enumerate(obpermutation):
            op_index = job_operations_counter[job]
            priority_order[(int(job),op_index)] = priority
            job_operations_counter[job] += 1        
        return priority_order
    
    def generate_schedule(self, priority_order):
        schedule = [[(-1, -1, -1, -1) for _ in range(self.instance.num_machines)] 
                   for _ in range(self.instance.num_jobs)]
        S = set()
        machine_end_times = [0] * self.instance.num_machines
        
        for job in range(self.instance.num_jobs):
            S.add((job, 0))

        processing_times = [[op[1] for op in job_ops] for job_ops in self.instance.operations]
        
        while S:
            sigmas = []
            phis = []
            for (j, op_idx) in S:
                machine = self.instance.operations[j][op_idx][0]
                prev_end = schedule[j][op_idx-1][3] if op_idx > 0 else 0
                sigma = max(prev_end, machine_end_times[machine])
                sigmas.append(sigma)
                phis.append(sigma + processing_times[j][op_idx])
            
            sigma_star = min(sigmas)
            phi_star = min(phis)
            
            candidates = []
            for (j, op_idx) in S:
                machine = self.instance.operations[j][op_idx][0]
                prev_end = schedule[j][op_idx-1][3] if op_idx > 0 else 0
                sigma = max(prev_end, machine_end_times[machine])
                if sigma <= sigma_star + self.delta * (phi_star - sigma_star):
                    candidates.append((j, op_idx, machine))
            
            candidates.sort(key=lambda x: (x[2], priority_order[(x[0], x[1])]))
            j, op_idx, machine = candidates[0]
            
            prev_end = schedule[j][op_idx-1][3] if op_idx > 0 else 0
            start = max(prev_end, machine_end_times[machine])
            end = start + processing_times[j][op_idx]
            
            schedule[j][op_idx] = (j+1, machine+1, start, end)
            machine_end_times[machine] = end
            S.remove((j, op_idx))
            
            if op_idx + 1 < self.instance.num_machines:
                S.add((j, op_idx + 1))
                
        return schedule
        
    def initialize_particle(self, index):
        particle=Particle(index=index ,solver=self)
        return particle
    
   
    def update_weight(self, iteration):
    # Décroissance linéaire de 0.9 → 0.4 sur les itérations
        self.weight = self.initial_weight - (self.initial_weight - self.min_weight) * (iteration / self.max_iterations)

    
    def run_solver(self):
        for iteration in range(self.max_iterations):
            self.update_weight(iteration)
            
            for particle in self.particles:
                 particle.update_personal_best()
            
            current_best = min(self.particles, key=lambda p: p.personal_best_fitness)
            if current_best.personal_best_fitness < self.global_best.personal_best_fitness:
                self.global_best = current_best.copy()
            
            for particle in self.particles:
                particle.update_local_best()
                particle.update_near_neighbor_best()
                particle.update_velocity()
                particle.update_position()
        return self.global_best        