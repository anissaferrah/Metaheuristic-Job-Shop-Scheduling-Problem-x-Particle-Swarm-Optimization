class Instance:
    """
    A class representing an instance of the Job Shop Scheduling Problem (JSP).
    Each instance has a number of jobs and machines, and a list of operations for each job.
    The operations are represented as a matrix of tuples of (machine, duration) per job.
    The purpose is to solve JSP using Particle Swarm Optimization (PSO).
    The instances of the problem to be solved are of the particularity that they have a fixed number of jobs and machines and that the operations of each job are fixed.
    Also the operations of each job are fixed and the order of the operations is fixed.
    Additionally, the number of operations per job is equal to the number of machines, and each operation is assigned one machine.
    """
    def __init__(self, num_jobs, num_machines, operations):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.operations = operations