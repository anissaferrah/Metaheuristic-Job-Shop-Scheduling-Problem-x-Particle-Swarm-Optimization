# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import matplotlib
matplotlib.use('Agg')  # Mode non interactif

class Particle:
    def __init__(self, num_operations, num_operations_per_job):
        self.position = np.random.rand(num_operations)
        num_jobs = num_operations // num_operations_per_job
        for job in range(num_jobs):
            start = job * num_operations_per_job
            end = start + num_operations_per_job
            self.position[start:end] = np.sort(self.position[start:end])
        self.velocity = np.zeros(num_operations)
        self.best_position = self.position.copy()
        self.best_score = float('inf')

def decode_particle(X, num_operations_per_job):
    sorted_indices = np.argsort(X)
    priority = np.zeros(len(X), dtype=int)
    current_priority = 1
    for idx in sorted_indices:
        priority[idx] = current_priority
        current_priority += 1
    return priority

def generate_schedule(priority, jobs, delta=0.5):
    num_jobs = len(jobs)
    num_machines = max(op[0] for job in jobs for op in job) + 1
    operations_per_job = len(jobs[0])
    
    schedule = []
    operation_order = []
    counters = {j: 0 for j in range(num_jobs)}
    
    for idx in np.argsort(priority):
        job = idx // operations_per_job
        operation = counters[job]
        operation_order.append((job, operation))
        counters[job] += 1

    next_operation = [0] * num_jobs
    machine_times = [0] * num_machines
    job_times = [0] * num_jobs

    for _ in range(num_jobs * operations_per_job):
        candidates = []
        for job in range(num_jobs):
            if next_operation[job] < operations_per_job:
                machine, duration = jobs[job][next_operation[job]]
                start = max(job_times[job], machine_times[machine])
                end = start + duration
                candidates.append((job, machine, start, end))

        if not candidates:
            break

        end_times = [c[3] for c in candidates]
        min_end = min(end_times)
        min_start = min(c[2] for c in candidates if c[3] == min_end)
        critical_machine = [c[1] for c in candidates if c[3] == min_end][0]
        
        threshold = min_start + delta * (min_end - min_start)
        eligible = [c for c in candidates 
                   if c[1] == critical_machine and c[2] <= threshold]
        
        if not eligible:
            eligible = [c for c in candidates if c[1] == critical_machine]

        min_priority = float('inf')
        selected = None
        for c in eligible:
            prio = operation_order.index((c[0], next_operation[c[0]]))
            if prio < min_priority:
                min_priority, selected = prio, c

        if selected is None:
            selected = min(candidates, key=lambda x: operation_order.index((x[0], next_operation[x[0]])))

        job, machine, start, end = selected
        schedule.append((job, machine, start, end))
        next_operation[job] += 1
        job_times[job] = end
        machine_times[machine] = end

    makespan = max(c[3] for c in schedule) if schedule else 0
    return makespan, schedule

def pso_jssp(jobs, max_iter=100, swarm_size=30, delta=0.5):
    operations_per_job = len(jobs[0])
    num_jobs = len(jobs)
    num_operations = num_jobs * operations_per_job
    
    swarm = [Particle(num_operations, operations_per_job) for _ in range(swarm_size)]
    global_best_pos = None
    global_best_score = float('inf')
    
    params = {'inertia': 0.9, 'cognitive': 1.5, 'social': 1.5}
    
    for _ in tqdm(range(max_iter), desc="Optimisation PSO"):
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
            cognitive = params['cognitive'] * r1 * (particle.best_position - particle.position)
            social = params['social'] * r2 * (global_best_pos - particle.position)
            particle.velocity = params['inertia'] * particle.velocity + cognitive + social
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, 0, 1)
            for job in range(num_jobs):
                start = job * operations_per_job
                end = start + operations_per_job
                particle.position[start:end] = np.sort(particle.position[start:end])

    pi = decode_particle(global_best_pos, operations_per_job)
    return generate_schedule(pi, jobs, delta)

def plot_gantt(schedule, num_machines, filename):
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = plt.cm.tab20.colors
    
    for entry in schedule:
        job, machine, start, end = entry
        duration = end - start
        
        ax.barh(machine, duration, left=start, height=0.7,
               color=colors[job % 20], edgecolor='black')
        
        ax.text(start + duration/2, machine, f'J{job}',
               va='center', ha='center', color='white', fontweight='bold')

    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'Machine {i}' for i in range(num_machines)])
    ax.set_xlabel('Time')
    ax.set_title('Gantt Chart - Job Shop Schedule')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()

def parse_all_instances(filename):
    """Parser corrigé pour les fichiers TAI"""
    instances = []
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    i = 0
    while i < len(lines):
        if lines[i].startswith('Nb of jobs'):
            try:
                # Lire la ligne suivante contenant les données numériques
                i += 1
                data_line = lines[i].split()
                
                # Extraire les 6 valeurs numériques attendues
                metadata = {
                    'num_jobs': int(data_line[0]),
                    'num_machines': int(data_line[1]),
                    'lower_bound': int(data_line[-1]),
                    'upper_bound': int(data_line[-2])
                }
                
                # Trouver la section Times
                i += 1
                while i < len(lines) and not lines[i].lower().startswith('times'):
                    i += 1
                i += 1
                
                # Lire les temps
                times = []
                for _ in range(metadata['num_jobs']):
                    times.append(list(map(int, lines[i].split())))
                    i += 1
                
                # Trouver la section Machines
                while i < len(lines) and not lines[i].lower().startswith('machines'):
                    i += 1
                i += 1
                
                # Lire les machines
                machines = []
                for _ in range(metadata['num_jobs']):
                    machines.append([m-1 for m in map(int, lines[i].split())])
                    i += 1
                
                # Construire la structure jobs
                jobs = []
                for job_idx in range(metadata['num_jobs']):
                    operations = []
                    for op_idx in range(metadata['num_machines']):
                        operations.append((
                            machines[job_idx][op_idx],
                            times[job_idx][op_idx]
                        ))
                    jobs.append(operations)
                
                instances.append({
                    'metadata': metadata,
                    'jobs': jobs
                })
                
            except (IndexError, ValueError) as e:
                print(f"Erreur de parsing ligne {i+1}: {str(e)}")
                continue
        else:
            i += 1
    return instances

def run_benchmark(filename):
    instances = parse_all_instances(filename)
    results = []
    
    output_dir = "benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, instance in enumerate(instances, 1):
        print(f"\n{'='*40}")
        print(f"Processing Instance {idx}/{len(instances)}")
        print(f"Jobs: {instance['metadata']['num_jobs']}")
        print(f"Machines: {instance['metadata']['num_machines']}")
        print(f"Bounds: LB={instance['metadata']['lower_bound']} | UB={instance['metadata']['upper_bound']}")
        
        # Run PSO
        makespan, schedule = pso_jssp(
            instance['jobs'],
            max_iter=100,
            swarm_size=40,
            delta=0.5
        )
        
        # Save Gantt
        gantt_path = f"{output_dir}/instance_{idx}_gantt.png"
        plot_gantt(schedule, instance['metadata']['num_machines'], gantt_path)
        
        # Calculate metrics
        gap = ((makespan - instance['metadata']['lower_bound']) / 
               instance['metadata']['lower_bound']) * 100
        
        results.append({
            'instance': idx,
            'makespan': makespan,
            'LB': instance['metadata']['lower_bound'],
            'UB': instance['metadata']['upper_bound'],
            'gap (%)': gap,
            'gantt_path': gantt_path
        })
    
    # Print report
    print("\n\nFinal Report:")
    print("{:<8} {:<10} {:<10} {:<10} {:<10} {:<20}".format(
        'Instance', 'Makespan', 'LB', 'UB', 'Gap (%)', 'Gantt Path'))
    
    for res in results:
        print("{:<8} {:<10} {:<10} {:<10} {:<10.2f} {:<20}".format(
            res['instance'],
            res['makespan'],
            res['LB'],
            res['UB'],
            res['gap (%)'],
            res['gantt_path']))
    
    return results

if __name__ == "__main__":
    benchmark_file = r'C:\Users\HERO-INFO\Desktop\M1-sii\meta project\banchmarks\tai20_15.txt'  # Mettre le bon chemin
    results = run_benchmark(benchmark_file)