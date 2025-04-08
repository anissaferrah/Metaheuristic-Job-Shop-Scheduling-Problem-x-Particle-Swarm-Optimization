# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from collections import defaultdict

class Particule:
    def __init__(self, num_operations):
        self.position = np.random.rand(num_operations)
        self.vitesse = np.zeros(num_operations)
        self.meilleur_position = self.position.copy()
        self.meilleur_score = float('inf')

def decoder_particule(X, num_operations_par_job):
    indices_tries = np.argsort(X)
    groupes = [indices_tries[i*num_operations_par_job:(i+1)*num_operations_par_job] 
              for i in range(len(X)//num_operations_par_job)]
    
    π = np.zeros(len(X), dtype=int)
    for priorite, groupe in enumerate(groupes, 1):
        for idx in groupe:
            π[idx] = priorite
    return π
def generer_planning(π, jobs, delta=0.5):
    num_jobs = len(jobs)
    operations_par_job = len(jobs[0])
    num_machines = max([op[0] for job in jobs for op in job]) + 1
    
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
        
        # Gestion des cas sans éligibles
        if not eligibles:
            eligibles = [c for c in candidates if c[1] == machine_critique]
            if not eligibles:
                eligibles = candidates.copy()
        if not eligibles:
            break

        priorite_min = float('inf')
        selection = eligibles[0]  # Valeur par défaut
        for c in eligibles:
            priorite = ordre_priorite.index((c[0], next_op[c[0]]))
            if priorite < priorite_min:
                priorite_min, selection = priorite, c

        job, machine, debut, fin = selection
        planning.append((job, machine, debut, fin))
        next_op[job] += 1
        temps_jobs[job] = fin
        temps_machines[machine] = fin

    makespan = max(c[3] for c in planning) if planning else 0
    return makespan, planning


def pso_jssp(jobs, max_iter=100, taille_essaim=30, inertie=0.8, cogni=1.5, social=1.5, delta=0.3):
    operations_par_job = len(jobs[0])
    num_jobs = len(jobs)
    num_operations = num_jobs * operations_par_job
    
    essaim = [Particule(num_operations) for _ in range(taille_essaim)]
    gbest_pos = None
    gbest_score = float('inf')
    
    for iter in tqdm(range(max_iter), desc=f"PSO(t={taille_essaim},i={max_iter})", leave=False):
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
            cognitif = cogni * r1 * (particule.meilleur_position - particule.position)
            social_component = social * r2 * (gbest_pos - particule.position)
            particule.vitesse = inertie * particule.vitesse + cognitif + social_component
            particule.position += particule.vitesse
            particule.position = np.clip(particule.position, 0, 1)

    π = decoder_particule(gbest_pos, operations_par_job)
    return generer_planning(π, jobs, delta)

def grid_search(jobs, param_grid):
    best = {'params': None, 'makespan': float('inf')}
    keys, values = zip(*param_grid.items())
    total_combinaisons = np.prod([len(v) for v in values])
    
    for combo in tqdm(itertools.product(*values), total=total_combinaisons, desc="Grid Search"):
        params = dict(zip(keys, combo))
        makespan, _ = pso_jssp(jobs, **params)
        
        if makespan < best['makespan']:
            best = {'params': params, 'makespan': makespan}
    
    return best

def afficher_gantt(planning, num_machines):
    plt.figure(figsize=(15, num_machines))
    couleurs = plt.cm.get_cmap('tab20', 20)
    
    for entry in planning:
        job, machine, start, end = entry
        plt.barh(machine, end - start, left=start, 
                color=couleurs(job % 20), edgecolor='black')
        plt.text((start + end)/2, machine, f'J{job}', 
                ha='center', va='center', color='white', fontweight='bold')
    
    plt.yticks(range(num_machines), [f'Machine {i}' for i in range(num_machines)])
    plt.xlabel('Temps')
    plt.title('Diagramme de Gantt - Solution Optimisée')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Configuration benchmark
    jobs = [
        [(0, 2), (2, 4), (1, 1)],  # Job 0
        [(1, 3), (0, 1), (2, 5)],  # Job 1
        [(2, 2), (1, 2), (0, 3)],  # Job 2
        [(0, 4), (1, 1), (2, 2)]   # Job 3
    ]
    
    # Grille de paramètres avec 3 valeurs par paramètre
    param_grid = {
        'taille_essaim': [20, 40, 60],
        'max_iter': [50, 100, 150],
        'inertie': [0.7, 0.8, 0.9],
        'cogni': [1.5, 2.0, 2.5],
        'social': [1.5, 2.0, 2.5],
        'delta': [0.2, 0.3, 0.4]
    }
    
    # Recherche des meilleurs paramètres
    meilleur_config = grid_search(jobs, param_grid)
    
    # Ré-exécution avec la meilleure configuration
    num_machines = max(op[0] for job in jobs for op in job) + 1
    makespan, planning = pso_jssp(jobs, **meilleur_config['params'])
    
    # Affichage des résultats
    print(f"\nMeilleure configuration trouvée:")
    for k, v in meilleur_config['params'].items():
        print(f"- {k}: {v}")
    print(f"Makespan final: {makespan}")
    
    afficher_gantt(planning, num_machines)