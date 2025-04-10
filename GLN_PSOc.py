from Particle import Particle
import numpy as np
class JSP_PSO_Solver:
    """
    A class to solve the Job Shop Scheduling Problem (JSP) using Particle Swarm Optimization (PSO).
    The class includes methods for initializing the particles, updating their positions, and calculating fitness.
    """
    def __init__(self, instance, population_size=40,neighberhood_size=7, max_iteration=1000,weight=0.9,slope=-5/9990,intercept=8996/9990,min_weight=0.1,cpersonal=0.5,cglobal=0.5,clocal=1.5,cneighbor=1.5,vmax=0.25,crossover_probability=0.3,pu=0.7,delta=0):
        self.instance = instance
        self.population_size = population_size
        self.neighberhood_size = neighberhood_size
        self.max_iterations = max_iteration
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
    
    def generate_schedule(self, priority_order ):
        # Initialize the schedule and needed variables
        schedule = [[(-1,-1,-1,-1) for _ in range(self.instance.num_machines)] for _ in range(self.instance.num_jobs)]
        S = set()
        for op in priority_order:
            if op[1] == 0:
                S.add(op)    
        #  The processing time of the operation Oij
        processing_times = np.array([[op[1] for op in job] for job in self.instance.operations])

        # The earliest time that operation Oij in St could be started
        sigma = []
        for jidx in range(self.instance.num_jobs):
            sigma.append([])
            for opidx in range(1,self.instance.num_machines+1):
                sigma[jidx].append(self.instance.operations[jidx][opidx-1][1])
        sigma= np.array(sigma)

        phi= np.add(processing_times, sigma)

        secheduled_machines_end_times = [0]*self.instance.num_machines
        while S:
            # Finding phi_star and sigma_star and M_star
            phi_star = float('inf')
            sigma_star = float('inf')
            for op in S:
                jidx = op[0]
                opidx = op[1]
                if phi[jidx][opidx] < phi_star:
                    phi_star = phi[jidx][opidx]
                if sigma[jidx][opidx] < sigma_star:
                    sigma_star = sigma[jidx][opidx]
            # Getting the candidates for M_star
            M_stars = []
            for op in S:
                jidx = op[0]
                opidx = op[1]
                if phi[jidx][opidx] == phi_star:
                    M_stars.append((jidx,opidx,self.instance.operations[jidx][opidx][0]))
            # Getting the M_star based on the machine index
            M_star = sorted(M_stars, key=lambda x: x[2])[0][2]  
            
            
            # Getting the operations in which they occur in M_star and satisfy the formula
            O_stars = []
            for op in S:
                jidx = op[0]
                opidx = op[1]
                if self.instance.operations[jidx][opidx][0] == M_star:
                    if self.delta == 0:
                        if sigma[jidx][opidx] == sigma_star:
                            O_stars.append((jidx, opidx))
                    else:
                        if sigma[jidx][opidx] <sigma_star+ self.delta*(phi_star-sigma_star):
                            O_stars.append((jidx, opidx))
            # Selecting O_star based on the priority order
            O_star=sorted(O_stars, key=lambda x: priority_order[x])[0]

            # Create the next stage schedule
            if O_star[1] == 0:
                start_time = secheduled_machines_end_times[M_star]
            elif secheduled_machines_end_times[M_star] > schedule[O_star[0]][O_star[1]-1][3]:
                start_time = secheduled_machines_end_times[M_star]
            else:
                start_time = schedule[O_star[0]][O_star[1]-1][3]
            end_time = start_time + self.instance.operations[O_star[0]][O_star[1]][1]
            schedule[O_star[0]][O_star[1]] = (O_star[0]+1, M_star+1, start_time, end_time)
            secheduled_machines_end_times[M_star] = end_time
            # Create the next stage set of operations to be schuduled
            S.remove(O_star)
            if O_star[1]+1 < self.instance.num_machines:
                S.add((O_star[0], O_star[1]+1))
        return schedule
        
    def initialize_particle(self, index):
        particle=Particle(index=index ,solver=self)
        return particle
    
    def update_weight(self):
        # Update the inertia weight based on the iteration number
        self.weight = self.weight * self.slope + self.intercept
        if self.weight < self.min_weight:
            self.weight = self.min_weight
    
    def run_solver(self):
        # Run the PSO algorithm
        for iteration in range(self.max_iterations):
            for particle in self.particles:
                particle.update_personal_best()
            for particle in self.particles:
                particle.update_global_best()
            for particle in self.particles:
                particle.update_local_best()
            for particle in self.particles:
                particle.update_near_neighbor_best()
            for particle in self.particles:
                particle.update_velocity()
            for particle in self.particles:
                particle.update_position()
        for particle in self.particles:
                particle.update_global_best()  

        return self.global_best