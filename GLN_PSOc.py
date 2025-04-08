from Particle import Particle
from Instance import Instance
import numpy as np
import matplotlib.pyplot as plt
class JSP_PSO_Solver:
    """
    A class to solve the Job Shop Scheduling Problem (JSP) using Particle Swarm Optimization (PSO).
    The class includes methods for initializing the particles, updating their positions, and calculating fitness.
    """
    def __init__(self, instance, num_particles, max_iterations):
        self.instance = instance
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.particles = [self.initialize_particle() for _ in range(num_particles)]
        self.global_best_position = None
        self.global_best_fitness = float('inf')

    def decode_position(self, particle, problem_instance=None):
        if problem_instance is None:
            # Default to a problem instance with 1 job and 1 machine
            num_jobs = 1
            num_machines = 1
        else:
            num_jobs= problem_instance.num_jobs
            num_machines= problem_instance.num_machines
        # Decode the position into a schedule
        # Get the operration-based permutation from the position
        obpermutation = self.get_operation_based_permutation(particle.position, num_jobs, num_machines)
        # Convert the operation-based permutation into a priority order
        priority_order = self.obpermuation_to_priority_order(obpermutation)
        # Generate the schedule based on the priority order
        schedule = self.generate_schedule(priority_order, problem_instance)
        return schedule
    
    def get_operation_based_permutation(self, position):
        # Sort the indices based on the position values
        sorted_indices = np.argsort(position)
        # Group the sorted indices into jobs
        groups = [sorted_indices[i*self.instance.num_machines:(i+1)*self.instance.num_machines] 
                  for i in range(self.instance.num_jobs)]
        # Create the operation-based permutation
        permutation = np.zeros(len(position), dtype=int)
        for priority, group in enumerate(groups, 1):
            for idx in group:
                permutation[idx] = priority
        return permutation

    """
    Convert the operation-based permutation into a priority order
    The priority order is a dictionary where the keys are tuples (job, operation) and the values are the priority of that operation
    The priority order is used to determine the order in which operations are scheduled
    """
    def obpermuation_to_priority_order(self, obpermutation):
        priority_order = {}
        job_operations_counters= {j:0 for j in range(self.instance.num_jobs)}
        for priority , job in enumerate(obpermutation):
            op_index = job_operations_counters[job]
            priority_order[(job,op_index)] = priority
            job_operations_counters[job] += 1        
        return priority_order
    
    def generate_schedule(self, priority_order , problem_instance=None):
        if problem_instance is None:
            # Default to a problem instance with 1 job and 1 machine
            num_jobs = 0
            num_machines = 0
            operations = []
        else:
            num_jobs= problem_instance.num_jobs
            num_machines= problem_instance.num_machines
            operations= problem_instance.operations
        # Initialize the schedule and needed variables
        stage = 1
        schedule = [[(-1,-1,-1,-1) for _ in range(num_machines)] for _ in range(num_jobs)]
        S = set(priority_order.keys())
        processing_times = np.array([[op[1] for op in job] for job in self.instance.operations])
        sigma = []
        for jidx in range(num_jobs):
            sigma.append([])
            for opidx in range(num_machines,1):
                sigma[jidx].append(operations[jidx][opidx-1][1])
        sigma= np.array(sigma)
        phi= np.add(processing_times, sigma)
        delta=0
        secheduled_machines_end_times = [0]*num_machines
        while S:
            # Finding phi_star and sigma_star and M_star
            phi_star = float('inf')
            sigma_star = float('inf')
            for jidx in range(num_jobs):
                for opidx in range(num_machines):
                    if phi[jidx][opidx] < phi_star:
                        phi_star = phi[jidx][opidx]
                        jidx_star = jidx
                        opidx_star = opidx
                    if sigma[jidx][opidx] < sigma_star:
                        sigma_star = sigma[jidx][opidx]
            # Getting the candidates for M_star
            M_stars = []
            for jidx in range(num_jobs):
                for opidx in range(num_machines):
                    if phi[jidx][opidx] == phi_star:
                        M_stars.append((jidx,opidx,self.instance.operations[jidx][opidx][0]))
            # Getting the M_star based on the machine index
            M_stars = sorted(M_stars, key=lambda x: x[2])
            M_star = M_stars[0][2]
            jidx_star = M_stars[0][0]
            opidx_star = M_stars[0][1]
            
            
            # Getting the operations in which they occur in M_star and satisfy the formula
            O_stars = []
            for jidx in range(num_jobs):
                for opidx in range(num_machines):
                    if self.instance.operations[jidx][opidx][0] == M_star:
                        if delta == 0:
                            if sigma[jidx_star][opidx_star] == sigma_star:
                                O_stars.append((jidx, opidx))
                        else:
                            if phi[jidx_star][opidx_star] <sigma_star+ delta*(phi_star-sigma_star):
                                O_stars.append((jidx, opidx))
            # Selecting O_star based on the priority order
            O_star=sorted(O_stars, key=lambda x: priority_order[x])[0]

            # Create the next stage schedule
            if O_star[1] == 0:
                start_time = secheduled_machines_end_times[M_star]
            elif secheduled_machines_end_times[M_star] > schedule[O_star[0]][O_star[1]-1][2] :
                start_time = secheduled_machines_end_times[M_star]
            else:
                start_time = schedule[O_star[0]][O_star[1]-1][2]
            end_time = start_time + operations[O_star[0]][O_star[1]][1]
            schedule[O_star[0]][O_star[1]] = (O_star[0], M_star, start_time, end_time)
            secheduled_machines_end_times[M_star] = end_time
            # Create the next stage set of operations to be schuduled
            S.remove(O_star)
            stage += 1
        return schedule
    
    def fitness(self, schedule=None):
        if schedule is None:
            schedule = self.decode_position(self.particles[0])
        # Calculate the makespan of the schedule
        makespan = max([op[3] for job in schedule for op in job])
        return makespan
    
    