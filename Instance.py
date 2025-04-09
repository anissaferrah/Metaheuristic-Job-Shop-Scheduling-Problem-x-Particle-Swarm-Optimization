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
    instances=[]
    @staticmethod
    def load_instances(file_path):
        """
        Load instances from a file.
        The file should contain the number of jobs, number of machines, and the operations for each job.
        """
        with open(file_path, 'r') as f:
            lines = f.readlines()
        index=3
        while index < len(lines):
            parameters = lines[index-2].strip().split()
            num_jobs = int(parameters[0])
            num_machines = int(parameters[1])
            times= []
            operations = []
            for i in range(index ,index+ num_jobs):
                line= lines[i].strip().split()
                times.append([int(line[j]) for j in range(0, len(line))])
            index+= num_jobs + 1
            for i in range(index,index+num_jobs):
                line= lines[i].strip().split()
                operations.append([(int(line[j]), times[i][j]) for j in range(0, len(line))])
            index+= num_jobs + 3
            instance = Instance(num_jobs, num_machines, operations)
            Instance.instances.append(instance)        

    def __init__(self, num_jobs=0, num_machines=0, operations=[]):
        self.num_jobs = num_jobs
        self.num_machines = num_machines
        self.num_operations = num_jobs * num_machines
        self.operations = operations

    @staticmethod
    def get_num_instances():
        """
        Get the number of instances loaded.
        """
        return len(Instance.instances)
    
    @staticmethod
    def get_instance(index=0):
        """
        Get an instance of the JSP by index.
        """
        if index < len(Instance.instances):
            return Instance.instances[index]
        else:
            raise IndexError("Instance index out of range.")