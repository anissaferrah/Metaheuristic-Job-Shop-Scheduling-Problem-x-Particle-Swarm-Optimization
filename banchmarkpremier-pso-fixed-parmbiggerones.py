# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')

# ==================================================================
# Modèle PSO amélioré
# ==================================================================
class ImprovedParticle:
    def __init__(self, num_operations, num_ops_per_job):
        self.num_ops_per_job = num_ops_per_job
        self.position = self._initialize_position(num_operations)
        self.velocity = np.zeros(num_operations)
        self.best_position = self.position.copy()
        self.best_score = float('inf')
        
    def _initialize_position(self, size):
        pos = np.random.rand(size) * 2 - 1
        for job in range(size // self.num_ops_per_job):
            start = job * self.num_ops_per_job
            end = start + self.num_ops_per_job
            pos[start:end] = np.cumsum(np.abs(pos[start:end]))
        return pos

def decode_particle(X, num_ops_per_job):
    indices = np.argsort(X)
    priority = np.zeros_like(indices)
    for i, idx in enumerate(indices):
        priority[idx] = i + 1
    return priority

# ==================================================================
# Métaheuristiques
# ==================================================================
def generate_schedule(priority, jobs, delta=0.5):
    num_jobs = len(jobs)
    operations_per_job = len(jobs[0])
    num_machines = max(op[0] for job in jobs for op in job) + 1

    schedule = []
    operation_order = []
    counters = {j: 0 for j in range(num_jobs)}
    
    for idx in np.argsort(priority):
        job = idx // operations_per_job
        operation = counters[job]
        operation_order.append((job, operation))
        counters[job] += 1

    next_op = [0] * num_jobs
    machine_times = [0] * num_machines
    job_times = [0] * num_jobs

    for _ in range(num_jobs * operations_per_job):
        candidates = []
        for job in range(num_jobs):
            if next_op[job] < operations_per_job:
                machine, duration = jobs[job][next_op[job]]
                start = max(job_times[job], machine_times[machine])
                end = start + duration
                candidates.append((job, machine, start, end))

        if not candidates: break

        end_times = [c[3] for c in candidates]
        min_end = min(end_times)
        min_start = min(c[2] for c in candidates if c[3] == min_end)
        crit_machine = [c[1] for c in candidates if c[3] == min_end][0]
        
        threshold = min_start + delta * (min_end - min_start)
        eligibles = [c for c in candidates if c[1] == crit_machine and c[2] <= threshold]
        
        if not eligibles:
            eligibles = [c for c in candidates if c[1] == crit_machine]

        min_prio = float('inf')
        selected = None
        for c in eligibles:
            prio = operation_order.index((c[0], next_op[c[0]]))
            if prio < min_prio: min_prio, selected = prio, c

        if selected is None:
            selected = min(candidates, key=lambda x: operation_order.index((x[0], next_op[x[0]])))

        job, machine, start, end = selected
        schedule.append((job, machine, start, end))
        next_op[job] += 1
        job_times[job] = end
        machine_times[machine] = end

    makespan = max(c[3] for c in schedule) if schedule else 0
    return makespan, schedule

def simulated_annealing(position, jobs, num_ops_per_job, temp=1000, cooling_rate=0.95):
    current = position.copy()
    best = current.copy()
    best_makespan, _ = generate_schedule(decode_particle(best, num_ops_per_job), jobs)
    
    while temp > 1:
        new = current.copy()
        idx1, idx2 = np.random.choice(len(new), 2, replace=False)
        new[idx1], new[idx2] = new[idx2], new[idx1]
        
        current_makespan, _ = generate_schedule(decode_particle(current, num_ops_per_job), jobs)
        new_makespan, _ = generate_schedule(decode_particle(new, num_ops_per_job), jobs)
        
        if (new_makespan < current_makespan or 
            np.random.rand() < np.exp(-(new_makespan - current_makespan)/temp)):
            current = new
            if new_makespan < best_makespan:
                best = new
                best_makespan = new_makespan
        
        temp *= cooling_rate
    
    return best

# ==================================================================
# Optimisation PSO
# ==================================================================
def pso_jssp(jobs, swarm_size=100, max_iter=500, inertia=0.9, cognitive=1.5, social=1.5, delta=0.25):
    operations_per_job = len(jobs[0])
    num_jobs = len(jobs)
    num_operations = num_jobs * operations_per_job
    
    swarm = [ImprovedParticle(num_operations, operations_per_job) for _ in range(swarm_size)]
    global_best_pos = None
    global_best_score = float('inf')
    
    for _ in tqdm(range(max_iter), desc=f'PSO {swarm_size} particles'):
        for particle in swarm:
            pi = decode_particle(particle.position, operations_per_job)
            makespan, _ = generate_schedule(pi, jobs, delta)
            
            if makespan < particle.best_score:
                particle.best_score = makespan
                particle.best_position = particle.position.copy()
                
            if makespan < global_best_score:
                global_best_score = makespan
                global_best_pos = particle.position.copy()

        for particle in swarm:
            r1, r2 = np.random.rand(num_operations), np.random.rand(num_operations)
            cognitive_component = cognitive * r1 * (particle.best_position - particle.position)
            social_component = social * r2 * (global_best_pos - particle.position)
            particle.velocity = inertia * particle.velocity + cognitive_component + social_component
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, -1, 1)
            
            # Maintien des contraintes de précédence
            for job in range(num_jobs):
                start = job * operations_per_job
                end = start + operations_per_job
                particle.position[start:end] = np.cumsum(np.abs(particle.position[start:end]))

    # Phase de local search
    global_best_pos = simulated_annealing(global_best_pos, jobs, operations_per_job)
    
    return generate_schedule(decode_particle(global_best_pos, operations_per_job), jobs, delta)

# ==================================================================
# Analyse et visualisation
# ==================================================================
def analyze_gantt(schedule, num_machines, filename):
    machine_timeline = defaultdict(list)
    
    for op in schedule:
        job, machine, start, end = op
        machine_timeline[machine].append((start, end, job))
    
    plt.figure(figsize=(15, 10))
    colors = plt.cm.tab20.colors
    
    for machine in sorted(machine_timeline.keys()):
        timeline = sorted(machine_timeline[machine], key=lambda x: x[0])
        last_end = 0
        for start, end, job in timeline:
            if start > last_end:
                plt.barh(machine, start - last_end, left=last_end, 
                        color='white', edgecolor='red', hatch='//', alpha=0.3)
            plt.barh(machine, end - start, left=start, 
                    color=colors[job % 20], edgecolor='black')
            plt.text((start + end)/2, machine, f'J{job}', 
                    ha='center', va='center', color='white', fontweight='bold')
            last_end = end
    
    plt.yticks(range(num_machines), [f'Machine {i}' for i in range(num_machines)])
    plt.xlabel('Time')
    plt.title('Optimized Schedule with Bottleneck Analysis')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

# ==================================================================
# Parsing des données et workflow
# ==================================================================
def parse_tai_file(filename):
    instances = []
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    i = 0
    while i < len(lines):
        if lines[i].startswith("Nb of jobs"):
            try:
                metadata = list(map(int, [x for x in lines[i+1].split() if x.isdigit()]))
                
                instance_info = {
                    "num_jobs": metadata[0],
                    "num_machines": metadata[1],
                    "lower_bound": metadata[-1],
                    "upper_bound": metadata[-2]
                }
                
                # Lecture des temps
                i += 2
                while not lines[i].lower().startswith("times"): i += 1
                i += 1
                times = []
                for _ in range(instance_info["num_jobs"]):
                    while not lines[i].replace(" ", "").isdigit(): i += 1
                    times.append(list(map(int, lines[i].split())))
                    i += 1
                
                # Lecture des machines
                while not lines[i].lower().startswith("machines"): i += 1
                i += 1
                machines = []
                for _ in range(instance_info["num_jobs"]):
                    while not lines[i].replace(" ", "").isdigit(): i += 1
                    machines.append([int(m)-1 for m in lines[i].split()])
                    i += 1
                
                # Construction des jobs
                jobs = []
                for job_idx in range(instance_info["num_jobs"]):
                    operations = []
                    for op_idx in range(instance_info["num_machines"]):
                        operations.append((machines[job_idx][op_idx], times[job_idx][op_idx]))
                    jobs.append(operations)
                
                instances.append({"metadata": instance_info, "jobs": jobs})
                
            except (IndexError, ValueError) as e:
                print(f"Erreur dans l'instance {len(instances)+1}: {str(e)}")
        i += 1
    
    return instances

# ==================================================================
# Workflow principal
# ==================================================================
if __name__ == "__main__":
    # Configuration
    benchmark_file = r'C:\Users\HERO-INFO\Desktop\M1-sii\meta project\banchmarks\tai20_15.txt'
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Paramètres fixes
    FIXED_PARAMS = {
        'swarm_size': 100,
        'max_iter': 300,
        'inertia': 0.9,
        'cognitive': 2.0,
        'social': 2.0,
        'delta': 0.3
    }
    
    # Chargement des instances
    instances = parse_tai_file(benchmark_file)
    
    results = []
    for idx, instance in enumerate(instances, 1):
        print(f"\n{'='*40}")
        print(f"Traitement de l'instance {idx}/{len(instances)}")
        
        # Optimisation avec paramètres fixes
        makespan, schedule = pso_jssp(instance['jobs'], **FIXED_PARAMS)
        
        # Analyse
        num_machines = instance['metadata']['num_machines']
        analyze_gantt(schedule, num_machines, f"{output_dir}/instance_{idx}_gantt.png")
        
        # Calcul du gap
        lb = instance['metadata']['lower_bound']
        gap = ((makespan - lb) / lb) * 100 if lb != 0 else 0
        
        results.append({
            'instance': idx,
            'makespan': makespan,
            'LB': lb,
            'UB': instance['metadata']['upper_bound'],
            'gap (%)': gap,
            'gantt': f"instance_{idx}_gantt.png"
        })
    
    # Rapport final
    print("\nRapport final:")
    print("{:<8} {:<10} {:<10} {:<10} {:<10} {:<20}".format(
        'Instance', 'Makespan', 'LB', 'UB', 'Gap (%)', 'Gantt Chart'))
    for res in results:
        print("{:<8} {:<10} {:<10} {:<10} {:<10.2f} {:<20}".format(
            res['instance'],
            res['makespan'],
            res['LB'],
            res['UB'],
            res['gap (%)'],
            res['gantt']))