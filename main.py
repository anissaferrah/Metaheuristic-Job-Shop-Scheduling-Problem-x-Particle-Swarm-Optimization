from Instance import Instance
from GLN_PSOc import JSP_PSO_Solver
import matplotlib.pyplot as plt
import numpy as np

def print_schedule(schedule):
    """Affiche le planning de manière lisible"""
    print("\nPlanning optimal:")
    print("Job\tMachine\tStart\tEnd")
    for job in schedule:
        for op in job:
            print(f"{op[0]}\t{op[1]}\t{op[2]}\t{op[3]}")

def plot_gantt(schedule, num_machines):
    """Crée un diagramme de Gantt à partir du planning"""
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab20.colors
    
    for job_schedule in schedule:
        for op in job_schedule:
            job, machine, start, end = op
            duration = end - start
            
            ax.barh(
                y=machine,
                width=duration,
                left=start,
                height=0.7,
                color=colors[job % 20],
                edgecolor="black",
                align="center"
            )
            
            ax.text(
                x=start + duration/2,
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
    ax.set_title("Diagramme de Gantt - Solution optimale")
    plt.grid(axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

def main():
    # Charger les instances (remplacer par votre fichier de données)
    # Instance.load_instances("chemin/vers/vos/données.txt")
    
    # Créer une instance manuellement pour l'exemple
    operations = [
        [(0, 2), (3, 4), (2, 1), (1, 1)],  # Job 0
        [(3, 3), (0, 1), (1, 1), (2, 5)],   # Job 1
        [(0, 4), (1, 1), (3, 1), (2, 2)]    # Job 2
    ]
    instance = Instance(num_jobs=3, num_machines=4, operations=operations)
    
    # Configurer le solveur PSO
    solver = JSP_PSO_Solver(
        instance=instance,
        population_size=30,
        max_iteration=100,
        weight=0.9,
        cpersonal=0.5,
        cglobal=0.5,
        clocal=1.5,
        cneighbor=1.5,
        vmax=0.25,
        crossover_probability=0.3,
        pu=0.7,
        delta=0
    )
    
    # Exécuter le solveur
    best_particle = solver.run_solver()
    
    # Afficher les résultats
    print(f"\nMakespan optimal: {best_particle.personal_best_fitness}")
    print_schedule(best_particle.schedule)
    plot_gantt(best_particle.schedule, instance.num_machines)

if __name__ == "__main__":
    main()