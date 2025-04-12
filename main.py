import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
from Instance import Instance
from GLN_PSOc import JSP_PSO_Solver
import matplotlib.pyplot as plt
import numpy as np
from os import getcwd

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
    plt.savefig('gantt_chart.png')
    print("Gantt chart saved as 'gantt_chart.png'")

def main():
    operations = [
    [(0, 2), (1, 1), (2, 3)],  # Job 0
    [(2, 2), (1, 1),(0, 1)]  # Job 1
]

    instance = Instance(num_jobs=2, num_machines=3, operations=operations)
    #print(getcwd())
   
    #instance = Instance.get_instance(0)
    solver = JSP_PSO_Solver(instance=instance)
     #    population_size=30,
     #    max_iteration=1,
      #   weight=0.9,
      #   cpersonal=0.5,
      #   cglobal=0.5,
      #   clocal=1.5,
      #   cneighbor=1.5,
      #   vmax=0.25,
      #   crossover_probability=0.3,
       #  pu=0.7,
       #  delta=0
    #)
    
    best_particle = solver.run_solver()
    
    print(f"\nMakespan optimal: {best_particle.personal_best_fitness}")
    print_schedule(best_particle.schedule)
    plot_gantt(best_particle.schedule, instance.num_machines)

if __name__ == "__main__":
    main()