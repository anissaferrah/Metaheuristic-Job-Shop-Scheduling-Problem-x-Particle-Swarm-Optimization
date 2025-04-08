import numpy as np
import matplotlib.pyplot as plt
import time

# 🌟 Classe représentant une particule pour l'algorithme PSO
class Particle:
    def __init__(self, num_ops):
        # Position aléatoire (priorités des opérations)
        self.position = np.random.rand(num_ops)
        # Vitesse initiale (zéro)
        self.velocity = np.zeros(num_ops)
        # Meilleure position personnelle trouvée
        self.pbest_position = self.position.copy()
        self.pbest_fitness = float('inf')  # fitness = makespan ici

#  Génère un planning (schedule) actif à partir des priorités (position)
def generate_schedule(permutation, jobs):
    num_jobs = len(jobs) #3
    num_machines = len(jobs[0])#3
    
    # Initialisation
    next_op = [0] * num_jobs  # Prochaine opération à exécuter pour chaque job [0,0,0]
    machine_available = [0] * num_machines  # Temps dispo des machines [0,0,0]
    job_available = [0] * num_jobs  # Temps dispo des jobs [0,0,0]
    schedule = []  # Liste finale du planning
    
    # Liste de priorités (ordre des opérations selon permutation)
    priority_list = [(i // num_machines, i % num_machines) for i in permutation.argsort()] 
   #Particle 1:[0.8, 0.1, 0.5, 0.3, 0.9, 0.2, 0.4, 0.7, 0.6]
   # [1, 5, 3, 6, 0, 2, 8, 7, 4]
   #[0, 0, 0, 1, 1, 1, 2, 2, 2]

   #[0, 1, 2, 3, 4, 5, 6, 7, 8]
   #[1, 0, 1, 0, 2, 0, 1, 2, 2]
   
   #[(0,1), (1,2), (1,0), (2,0), (0,0), (0,2), (2,2), (2,1), (1,1)]
    while True:
        schedulable = []
        for job in range(num_jobs):
            if next_op[job] < num_machines:
                machine, duration = jobs[job][next_op[job]]
                start = max(job_available[job], machine_available[machine])
                finish = start + duration
                schedulable.append((job, machine, start, finish))

        if not schedulable:
            break

        # Sélectionne l'opération avec le temps de fin le plus tôt
        min_finish = min(op[3] for op in schedulable)
        candidates = [op for op in schedulable if op[3] == min_finish]

        # Utilise les priorités pour départager en cas d'égalité
        selected = min(candidates, key=lambda x: priority_list.index((x[0], next_op[x[0]])))

        job, machine, start, finish = selected
        schedule.append((job, machine, start, finish))

        # Mise à jour des disponibilités
        next_op[job] += 1
        job_available[job] = finish
        machine_available[machine] = finish

    # Makespan = fin de la dernière opération
    makespan = max(op[3] for op in schedule)
    return makespan, schedule

#  Affiche un diagramme de Gantt
def plot_gantt(schedule, num_machines):
    fig, ax = plt.subplots()
    for entry in schedule:
        job, machine, start, finish = entry
        ax.broken_barh([(start, finish - start)], (machine - 0.4, 0.8),
                       facecolors=plt.cm.tab10(job))
        ax.text(start + (finish - start)/2, machine, f"J{job}", va='center', ha='center', color='white')
    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f'M{i}' for i in range(num_machines)])
    ax.set_xlabel('Time')
    ax.set_title('Job Shop Schedule (Gantt Chart)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

#  Algorithme PSO appliqué au problème de job shop
def pso_jssp(jobs, num_machines, max_iter=50, pop_size=20):
    num_ops = len(jobs) * num_machines #9
    print("num_ops",num_ops)
    swarm = [Particle(num_ops) for _ in range(pop_size)]  # Création de l’essaim
    print("Particle",swarm) # Particle 1:[0.8, 0.1, 0.5, 0.3, 0.9, 0.2, 0.4, 0.7, 0.6]
    gbest_position = None
    gbest_fitness = float('inf')
    
    # Paramètres PSO
    w_start, w_end = 0.9, 0.4
    cp, cg = 1.5, 1.5

    print("\n=== Début de l'optimisation PSO ===\n")

    for iter in range(max_iter):
        w = w_start - (w_start - w_end) * (iter / max_iter)  # inertie variable
        print("w",w)
        print(f"--- Itération {iter+1}/{max_iter} ---")

        for i, particle in enumerate(swarm):
            makespan, _ = generate_schedule(particle.position, jobs)
            print(f"  Particule {i+1} : Makespan = {makespan:.2f}")

            # Mise à jour du meilleur perso (pbest)
            if makespan < particle.pbest_fitness:
                particle.pbest_fitness = makespan
                particle.pbest_position = particle.position.copy()

            # Mise à jour du meilleur global (gbest)
            if makespan < gbest_fitness:
                gbest_fitness = makespan
                gbest_position = particle.position.copy()
                print(f"     Nouveau meilleur global trouvé : {gbest_fitness}")

        # Mise à jour des vitesses et positions
        for particle in swarm:
            r1, r2 = np.random.rand(num_ops), np.random.rand(num_ops)
            cognitive = cp * r1 * (particle.pbest_position - particle.position)
            social = cg * r2 * (gbest_position - particle.position)
            particle.velocity = w * particle.velocity + cognitive + social
            particle.position += particle.velocity
            particle.position = np.clip(particle.position, 0, 1)  # Garder entre 0 et 1

    print("\n=== Fin de l'optimisation ===\n")

    # Génère le planning final à partir du gbest
    best_makespan, best_schedule = generate_schedule(gbest_position, jobs)
    return best_makespan, best_schedule

#  Exemple simple (3 jobs × 3 machines)
jobs = [
    [(0, 3), (1, 2), (2, 2)],  # Job 0
    [(1, 2), (2, 1), (0, 4)],  # Job 1
    [(2, 4), (0, 3), (1, 1)]   # Job 2
]
  
# ⏱ Exécution
start_time = time.time()
makespan, schedule = pso_jssp(jobs, num_machines=3)
elapsed = time.time() - start_time

#  Résultat final
print(f" Temps d'exécution : {elapsed:.2f} secondes")
print(f" Meilleur makespan trouvé : {makespan}")
print("\n Planning final (schedule) :")
for entry in schedule:
    print(f"  Job {entry[0]} -> Machine {entry[1]} : start = {entry[2]}, finish = {entry[3]}")

# 📈 Gantt chart
plot_gantt(schedule, num_machines=3)
