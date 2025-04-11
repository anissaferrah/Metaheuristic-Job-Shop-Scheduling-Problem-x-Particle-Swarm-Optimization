import time
import csv
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, Process
import os
import matplotlib.pyplot as plt
from Instance import Instance
from GLN_PSOc import JSP_PSO_Solver

# Dossiers de sortie pour les CSV et les diagrammes de Gantt
GRID_CSV_DIR = "results/grid_search_results"
GANTT_DIR = "results/gantt_chart_results"
os.makedirs(GRID_CSV_DIR, exist_ok=True)
os.makedirs(GANTT_DIR, exist_ok=True)

BENCHMARK_FILES = [
    'benchmarks/tai20_15.txt',
    'benchmarks/tai30_15.txt',
    'benchmarks/tai50_15.txt',
    'benchmarks/tai100_20.txt'
]

def result_writer(queue, csv_filename):
    """Processus d'écriture CSV avec filtrage des champs techniques"""
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = [
            'instance_id', 'population_size', 'neighberhood_size', 'max_iteration',
            'weight', 'cpersonal', 'cglobal', 'clocal', 'cneighbor', 'vmax',
            'crossover_probability', 'pu', 'makespan', 'execution_time'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        while True:
            result = queue.get()
            if result is None:  # Signal de terminaison
                break
            # N'écrire que les champs pertinents pour le CSV
            filtered = {k: v for k, v in result.items() if k in fieldnames}
            writer.writerow(filtered)
            csvfile.flush()

def process_task(task_data):
    """Tâche de traitement avec séparation des données techniques"""
    instance_data, params, instance_idx = task_data
    try:
        instance = Instance(
            num_jobs=instance_data['num_jobs'],
            num_machines=instance_data['num_machines'],
            operations=instance_data['operations']
        )
        
        fixed_params = {
            'slope': -5/9990,
            'intercept': 8996/9990,
            'min_weight': 0.1,
            'delta': 0
        }
        
        start_time = time.time()
        solver = JSP_PSO_Solver(instance=instance, **{**params, **fixed_params})
        best_particle = solver.run_solver()
        execution_time = time.time() - start_time

        return {
            # Champs pour CSV
            'instance_id': instance_idx,
            'population_size': params['population_size'],
            'neighberhood_size': params['neighberhood_size'],
            'max_iteration': params['max_iteration'],
            'weight': params['weight'],
            'cpersonal': params['cpersonal'],
            'cglobal': params['cglobal'],
            'clocal': params['clocal'],
            'cneighbor': params['cneighbor'],
            'vmax': params['vmax'],
            'crossover_probability': params['crossover_probability'],
            'pu': params['pu'],
            'makespan': best_particle.personal_best_fitness,
            'execution_time': execution_time,
            
            # Champs techniques (préfixe _)
            '_schedule': best_particle.schedule,
            '_instance': instance
        }
    except Exception as e:
        print(f"Erreur sur l'instance {instance_idx}: {str(e)}")
        return None

def plot_gantt_chart(schedule, instance, title="Gantt Chart", filename=None):
    """Visualisation du planning avec gestion d'erreur améliorée"""
    from itertools import chain
    fig, ax = plt.subplots(figsize=(15, 8))
    colors = plt.cm.tab20.colors

    # Correction de la structure de schedule si nécessaire
    if any(isinstance(op, list) for op in schedule):
        schedule = list(chain.from_iterable(schedule))

    for op in schedule:
        try:
            job_id, machine_id = op[0], op[1]
            start, end = op[2], op[3]

            if isinstance(start, (tuple, list)):
                start = start[0]
            if isinstance(end, (tuple, list)):
                end = end[0]

            start = float(start)
            end = float(end)
            duration = end - start

            task_id = next(
                i for i, (m, _) in enumerate(instance.operations[job_id]) 
                if m == machine_id
            )

            ax.barh(
                machine_id, duration, left=start,
                color=colors[job_id % len(colors)],
                edgecolor='black', linewidth=0.5
            )

            ax.text(
                start + duration/2, machine_id,
                f'J{job_id+1}T{task_id+1}',
                va='center', ha='center',
                color='white', fontsize=8
            )

        except Exception as e:
            print(f"Erreur sur l'opération {op}: {str(e)}")
            continue

    ax.set_yticks(range(instance.num_machines))
    ax.set_yticklabels([f'Machine {i+1}' for i in range(instance.num_machines)])
    ax.set_xlabel("Temps")
    ax.set_ylabel("Machines")
    ax.set_title(title)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if filename:
        plt.savefig(filename)
        plt.close()
    else:
        plt.show()

def main():
    for benchmark_file in BENCHMARK_FILES:
        Instance.load_instances(benchmark_file)
        benchmark_name = os.path.splitext(os.path.basename(benchmark_file))[0]
        csv_filename = os.path.join(GRID_CSV_DIR, f"{benchmark_name}-gridsearch-results.csv")

        param_grid = {
            'population_size': [20, 30, 50],
            'neighberhood_size': [5, 7, 10],
            'max_iteration': [100, 500, 1000],
            'weight': [0.7, 0.9, 1.1],
            'cpersonal': [0.5, 0.75, 1.0],
            'cglobal': [0.5, 0.75, 1.0],
            'clocal': [1.0, 1.5, 2.0],
            'cneighbor': [1.0, 1.5, 2.0],
            'vmax': [0.25, 0.5, 1.0],
            'crossover_probability': [0.3, 0.5, 0.7],
            'pu': [0.5, 0.7, 1.0]
        }
            

        best_combination = None
        best_makespan = float('inf')

        with Manager() as manager:
            result_queue = manager.Queue()
            writer_process = Process(target=result_writer, args=(result_queue, csv_filename))
            writer_process.start()

            tasks = []
            if Instance.get_num_instances() > 0:
                instance_idx = 0
                instance = Instance.get_instance(instance_idx)
                instance_data = {
                    'num_jobs': instance.num_jobs,
                    'num_machines': instance.num_machines,
                    'operations': instance.operations
                }
                
                for values in product(*param_grid.values()):
                    param_dict = dict(zip(param_grid.keys(), values))
                    tasks.append((instance_data, param_dict, instance_idx))
            else:
                print(f"Aucune instance trouvée pour {benchmark_file}")
                continue

            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(process_task, task) for task in tasks]
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        # Écriture CSV
                        result_queue.put(result)
                        
                        # Mise à jour de la meilleure solution
                        if result['makespan'] < best_makespan:
                            best_makespan = result['makespan']
                            best_combination = {
                                'schedule': result['_schedule'],
                                'instance': result['_instance'],
                                'makespan': result['makespan'],
                                'params': {k: result[k] for k in param_grid.keys()}
                            }

            result_queue.put(None)
            writer_process.join()

        # Affichage des résultats dans le terminal
        print(f"\nRésultats pour {benchmark_name}:")
        if best_combination:
            print(f"Makespan optimal: {best_combination['makespan']}")
            print("Meilleurs paramètres utilisés :")
            for key, value in best_combination['params'].items():
                print(f"  {key}: {value}")
            plot_gantt_chart(
                best_combination['schedule'],
                best_combination['instance'],
                f"Meilleur planning - {benchmark_name} (Makespan: {best_combination['makespan']})",
                filename=os.path.join(GANTT_DIR, f"gantt_chart-{benchmark_name}.png")
            )
        else:
            print("Aucune solution valide trouvée")

if __name__ == "__main__":
    main()
