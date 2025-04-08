# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Particule:
    def __init__(self, num_operations, num_operations_par_job):
        self.position = np.random.rand(num_operations)
        num_jobs = num_operations // num_operations_par_job
        # Ensure each job's operations are sorted
        for job in range(num_jobs):
            start = job * num_operations_par_job
            end = start + num_operations_par_job
            self.position[start:end] = np.sort(self.position[start:end])
        self.vitesse = np.zeros(num_operations)
        self.meilleur_position = self.position.copy()
        self.meilleur_score = float('inf')

def decoder_particule(X, num_operations_par_job):
    indices_tries = np.argsort(X)
    π = np.zeros(len(X), dtype=int)
    current_priority = 1
    for idx in indices_tries:
        π[idx] = current_priority
        current_priority += 1
    return π

def generer_planning(π, jobs, delta=0.5):
    num_jobs = len(jobs)
    num_machines = max([op[0] for job in jobs for op in job]) + 1
    operations_par_job = len(jobs[0])
    
    planning = []
    ordre_priorite = []
    compteurs = {j:0 for j in range(num_jobs)}
    
    for idx in np.argsort(π):
        job = idx // operations_par_job
        operation = compteurs[job]
        ordre_priorite.append((job, operation))
        compteurs[job] += 1

    next_op = [0] * num_jobs
    temps_machines = [0] * num_machines
    temps_jobs = [0] * num_jobs

    for _ in range(num_jobs * operations_par_job):
        candidates = []
        for job in range(num_jobs):
            if next_op[job] < operations_par_job:
                machine, duree = jobs[job][next_op[job]]
                debut = max(temps_jobs[job], temps_machines[machine])
                fin = debut + duree
                candidates.append((job, machine, debut, fin))

        if not candidates:
            break

        fins = [c[3] for c in candidates]
        fin_min = min(fins)
        debut_min = min(c[2] for c in candidates if c[3] == fin_min)
        machine_critique = [c[1] for c in candidates if c[3] == fin_min][0]
        
        seuil = debut_min + delta * (fin_min - debut_min)
        eligibles = [c for c in candidates 
                    if c[1] == machine_critique and c[2] <= seuil]
        
        # Fallback to all candidates on machine_critique if no eligibles
        if not eligibles:
            eligibles = [c for c in candidates if c[1] == machine_critique]

        priorite_min = float('inf')
        selection = None
        for c in eligibles:
            priorite = ordre_priorite.index((c[0], next_op[c[0]]))
            if priorite < priorite_min:
                priorite_min, selection = priorite, c

        if selection is None:
            # Handle case where no selection was made (fallback)
            selection = min(candidates, key=lambda x: ordre_priorite.index((x[0], next_op[x[0]])))

        job, machine, debut, fin = selection
        planning.append((job, machine, debut, fin))
        next_op[job] += 1
        temps_jobs[job] = fin
        temps_machines[machine] = fin

    makespan = max(c[3] for c in planning) if planning else 0
    return makespan, planning

def pso_jssp(jobs, max_iter=100, taille_essaim=30, delta=0.5):
    operations_par_job = len(jobs[0])
    num_jobs = len(jobs)
    num_operations = num_jobs * operations_par_job
    num_machines = max([op[0] for job in jobs for op in job]) + 1
    
    essaim = [Particule(num_operations, operations_par_job) for _ in range(taille_essaim)]
    gbest_pos = None
    gbest_score = float('inf')
    
    params = {'inertie': 0.9, 'cogni': 1.5, 'social': 1.5}
    
    for iter in tqdm(range(max_iter)):
        for particule in essaim:
            π = decoder_particule(particule.position, operations_par_job)
            makespan, _ = generer_planning(π, jobs, delta)
            
            if makespan < particule.meilleur_score:
                particule.meilleur_score = makespan
                particule.meilleur_position = particule.position.copy()
                
            if makespan < gbest_score:
                gbest_score = makespan
                gbest_pos = particule.position.copy()

        for particule in essaim:
            r1, r2 = np.random.rand(num_operations), np.random.rand(num_operations)
            cognitif = params['cogni'] * r1 * (particule.meilleur_position - particule.position)
            social = params['social'] * r2 * (gbest_pos - particule.position)
            particule.vitesse = params['inertie'] * particule.vitesse + cognitif + social
            particule.position += particule.vitesse
            particule.position = np.clip(particule.position, 0, 1)
            # Re-sort each job's operations to maintain order
            for job in range(num_jobs):
                start = job * operations_par_job
                end = start + operations_par_job
                particule.position[start:end] = np.sort(particule.position[start:end])

    π = decoder_particule(gbest_pos, operations_par_job)
    return generer_planning(π, jobs, delta)

def afficher_gantt(planning, num_machines):
    fig, ax = plt.subplots(figsize=(12, 6))
    couleurs = plt.cm.tab20.colors
    
    for entree in planning:
        job, machine, debut, fin = entree
        duree = fin - debut
        
        ax.barh(
            y=machine,
            width=duree,
            left=debut,
            height=0.7,
            color=couleurs[job % 20],
            edgecolor="black",
            align="center"
        )
        
        ax.text(
            x=debut + duree/2,
            y=machine,
            s=f"J{job}",
            va="center",
            ha="center",
            color="white",
            fontweight="bold"
        )
    
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {i}" for i in range(num_machines)])
    ax.set_xlabel("Temps")
    ax.set_title("Diagramme de Gantt - Planification Job Shop")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    jobs = [
        [(0, 2), (2, 4), (1, 1)],  # Job 0
        [(1, 3), (0, 1), (2, 5)],  # Job 1
        [(2, 2), (1, 2), (0, 3)],  # Job 2
        [(0, 4), (1, 1), (2, 2)]   # Job 3
    ]
    makespan, planning = pso_jssp(jobs, max_iter=100, delta=0.3)
    print(f"Makespan optimal: {makespan}")
    
    num_machines = max(op[0] for job in jobs for op in job) + 1
    afficher_gantt(planning, num_machines)