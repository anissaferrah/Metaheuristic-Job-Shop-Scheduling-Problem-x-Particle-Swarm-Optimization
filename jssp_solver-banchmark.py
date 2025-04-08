import numpy as np
import matplotlib.pyplot as plt
import random
import os

class JSSP_PSO:
    def __init__(self, instance_file):
        # Vérifier l'existence du fichier
        if not os.path.exists(instance_file):
            abs_path = os.path.abspath(instance_file)
            raise FileNotFoundError(f"""
            Fichier introuvable : {instance_file}
            Chemin absolu essayé : {abs_path}
            Vérifiez que :
            1. Le fichier existe dans le répertoire : {os.getcwd()}
            2. Le nom correspond exactement (extension .txt incluse)
            3. Vous avez les droits d'accès
            """)
        
        self.jobs = []
        self.num_machines = 0
        self.num_jobs = 0
        self.parse_taillard(instance_file)
        self.processing_info = self.get_processing_info()

        # Paramètres PSO
        self.num_particles = 30
        self.max_iter = 100
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        
    def parse_taillard(self, filename):
        with open(filename) as f:
            lines = [line.strip() for line in f if line.strip()]
            
        times_start = lines.index("Times") + 1
        machines_start = lines.index("Machines") + 1
        
        # Lire le header
        header = list(map(int, lines[1].split()))
        self.num_jobs, self.num_machines = header[0], header[1]
        
        # Lire les temps
        times = []
        for line in lines[times_start:times_start + self.num_jobs]:
            times.append(list(map(int, line.split())))
            
        # Lire les machines (convertir en 0-based)
        machines = []
        for line in lines[machines_start:machines_start + self.num_jobs]:
            machines.append([m-1 for m in map(int, line.split())])
            
        # Construire la structure des jobs
        for job_idx in range(self.num_jobs):
            operations = []
            for op_idx in range(self.num_machines):
                operations.append({
                    'machine': machines[job_idx][op_idx],
                    'time': times[job_idx][op_idx]
                })
            self.jobs.append(operations)
            
    def get_processing_info(self):
        return [[(op['machine'], op['time']) for op in job] for job in self.jobs]
    
    def initialize_particle(self):
        particle = []
        for job in self.jobs:
            keys = np.sort(np.random.uniform(0, 1, len(job))).tolist()
            particle.extend(keys)
        return np.array(particle)
    
    def decode(self, particle):
        operations = []
        ptr = 0
        for job_idx, job in enumerate(self.jobs):
            for op_idx in range(len(job)):
                operations.append((
                    job_idx,
                    op_idx,
                    particle[ptr],
                    self.processing_info[job_idx][op_idx][0],
                    self.processing_info[job_idx][op_idx][1]
                ))
                ptr += 1
                
        sorted_ops = sorted(operations, key=lambda x: x[2])
        
        machine_times = [0] * self.num_machines
        job_times = [0] * self.num_jobs
        schedule = {m: [] for m in range(self.num_machines)}
        
        for op in sorted_ops:
            job_idx, _, _, machine, duration = op
            start = max(job_times[job_idx], machine_times[machine])
            end = start + duration
            
            schedule[machine].append((job_idx, start, end))
            machine_times[machine] = end
            job_times[job_idx] = end
            
        return max(machine_times), schedule
    
    def optimize(self):
        particles = [self.initialize_particle() for _ in range(self.num_particles)]
        velocities = [np.zeros_like(p) for p in particles]
        
        personal_best = particles.copy()
        personal_best_scores = [self.decode(p)[0] for p in particles]
        
        global_best_idx = np.argmin(personal_best_scores)
        global_best = particles[global_best_idx]
        global_best_score = personal_best_scores[global_best_idx]
        
        convergence = []
        
        for iter in range(self.max_iter):
            for i in range(self.num_particles):
                r1, r2 = random.random(), random.random()
                velocities[i] = (self.w * velocities[i] +
                                 self.c1 * r1 * (personal_best[i] - particles[i]) +
                                 self.c2 * r2 * (global_best - particles[i]))
                
                new_position = particles[i] + velocities[i]
                
                # Réparation des clés
                ptr = 0
                for job in self.jobs:
                    job_len = len(job)
                    new_position[ptr:ptr+job_len] = np.sort(new_position[ptr:ptr+job_len])
                    ptr += job_len
                
                particles[i] = new_position
                current_score = self.decode(particles[i])[0]
                
                if current_score < personal_best_scores[i]:
                    personal_best[i] = particles[i]
                    personal_best_scores[i] = current_score
                    
                    if current_score < global_best_score:
                        global_best = particles[i]
                        global_best_score = current_score
                        
            convergence.append(global_best_score)
            print(f"Iteration {iter+1}: Best Makespan = {global_best_score}")
            
        return global_best_score, convergence, self.decode(global_best)[1]
    
    def plot_gantt(self, schedule):
        plt.figure(figsize=(15, 8))
        colors = plt.cm.get_cmap('tab20', self.num_jobs)
        
        for machine in schedule:
            for job in schedule[machine]:
                job_id, start, end = job
                plt.barh(machine, end - start, left=start, 
                        color=colors(job_id), edgecolor='black')
                plt.text((start + end)/2, machine, f'J{job_id}', 
                        ha='center', va='center', color='white')
        
        plt.xlabel('Time')
        plt.ylabel('Machines')
        plt.title(f'Job Shop Schedule - Makespan: {max([op[2] for ops in schedule.values() for op in ops])}')
        plt.grid(axis='x')
        plt.tight_layout()
        plt.show()
if __name__ == "__main__":
    try:
        # Utiliser le chemin absolu pour Windows
        solver = JSSP_PSO(r"C:\Users\HERO-INFO\Desktop\tai20_15.txt")  
        best_score, convergence, schedule = solver.optimize()
        solver.plot_gantt(schedule)
        
    except FileNotFoundError as e:
        print("ERREUR CRITIQUE :", e)
        print("\nListe des fichiers présents sur le Bureau :")
        desktop = r"C:\Users\HERO-INFO\Desktop"
        print(os.listdir(desktop))
