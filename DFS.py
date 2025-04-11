import copy
import matplotlib.pyplot as plt
from collections import defaultdict

class JobShopPureDFS:
    def __init__(self, jobs, machines):
        self.jobs = jobs
        self.machines = sorted(machines)
        self.best_makespan = float('inf')
        self.best_schedule = None
        self.initHelperStructures()

    def initHelperStructures(self):
        self.job_op_counts = [len(job) for job in self.jobs]
        self.machine_indices = {m: i for i, m in enumerate(self.machines)}
        self.all_ops = [(ji, oi, op)
                        for ji, job in enumerate(self.jobs)
                        for oi, op in enumerate(job)]

    def solve(self):
        initial_state = {
            'machine_schedules': {m: [] for m in self.machines},
            'scheduled_ops': set(),
            'op_end_times': {},
            'current_makespan': 0
        }

        stack = [initial_state]

        while stack:
            current = stack.pop()

            if self.isComplete(current):
                self.updateBest(current)
                continue

            candidates = self.getCandidates(current)

            for job_idx, op_idx, op in candidates:
                new_state = self.applyMove(current, job_idx, op_idx, op)
                if not self.shouldPrune(new_state):
                    stack.append(new_state)

        return self.best_schedule, self.best_makespan

    def isComplete(self, state):
        return len(state['scheduled_ops']) == len(self.all_ops)

    def getCandidates(self, state):
        candidates = []
        scheduled_ops = state['scheduled_ops']

        for job_idx, op_idx, op in self.all_ops:
            if (job_idx, op_idx) in scheduled_ops:
                continue
            if op_idx > 0 and (job_idx, op_idx - 1) not in scheduled_ops:
                continue
            candidates.append((job_idx, op_idx, op))
        return candidates

    def applyMove(self, state, job_idx, op_idx, op):
        new_state = {
            'machine_schedules': {m: list(ops) for m, ops in state['machine_schedules'].items()},
            'scheduled_ops': set(state['scheduled_ops']),
            'op_end_times': state['op_end_times'].copy(),
            'current_makespan': state['current_makespan']
        }

        machine = op['machine']
        processing_time = op['time']

        prev_op_end = state['op_end_times'].get((job_idx, op_idx - 1), 0) if op_idx > 0 else 0
        machine_available = new_state['machine_schedules'][machine][-1]['end'] if new_state['machine_schedules'][machine] else 0
        start_time = max(prev_op_end, machine_available)
        end_time = start_time + processing_time

        scheduled_op = {
            'job': job_idx + 1,  # Output job index as 1-based
            'op': op_idx + 1,    # Output operation index as 1-based
            'start': start_time,
            'end': end_time
        }
        new_state['machine_schedules'][machine].append(scheduled_op)
        new_state['scheduled_ops'].add((job_idx, op_idx))
        new_state['op_end_times'][(job_idx, op_idx)] = end_time
        new_state['current_makespan'] = max(new_state['current_makespan'], end_time)
        return new_state

    def estimateMakespan(self, state):
        machine_ends = {m: (state['machine_schedules'][m][-1]['end'] if state['machine_schedules'][m] else 0)
                        for m in self.machines}
        remaining_work = {m: 0 for m in self.machines}
        for job_idx, op_idx, op in self.all_ops:
            if (job_idx, op_idx) not in state['scheduled_ops']:
                machine = op['machine']
                remaining_work[machine] += op['time']
        return max(machine_ends[m] + remaining_work[m] for m in self.machines)

    def shouldPrune(self, state):
        return self.estimateMakespan(state) > self.best_makespan

    def updateBest(self, state):
        if state['current_makespan'] < self.best_makespan:
            self.best_makespan = state['current_makespan']
            self.best_schedule = copy.deepcopy(state['machine_schedules'])
            return True
        return False

def plot_gantt(schedule, n_machines):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['skyblue', 'lightgreen', 'salmon', 'plum', 'orange']
    for machine_index, ops in schedule.items():
        for op in ops:
            ax.barh(y=machine_index, width=op['end']-op['start'], left=op['start'],
                    height=0.5, align='center',
                    color=colors[(op['job'] - 1) % len(colors)], # Adjust job index for color
                    edgecolor='black')
            ax.text(op['start'] + (op['end']-op['start'])/2, machine_index,
                    f'J{op["job"]}O{op["op"]}',
                    va='center', ha='center', color='black', fontsize=8)
    ax.set_yticks(range(1, n_machines + 1))
    ax.set_yticklabels([f'M{m}' for m in range(1, n_machines + 1)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title("Gantt Chart for Best JSSP Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    jobs_data_1_based = [
         [{'machine': 3, 'time': 1}, {'machine': 1, 'time': 3}, {'machine': 2, 'time': 6}, 
     {'machine': 4, 'time': 7}, {'machine': 6, 'time': 3}, {'machine': 5, 'time': 6}],
    [{'machine': 2, 'time': 8}, {'machine': 3, 'time': 5}, {'machine': 5, 'time': 10}, 
     {'machine': 6, 'time': 10}, {'machine': 1, 'time': 10}, {'machine': 4, 'time': 4}],
    [{'machine': 3, 'time': 5}, {'machine': 1, 'time': 5}, {'machine': 2, 'time': 5}, 
     {'machine': 4, 'time': 3}, {'machine': 5, 'time': 8}, {'machine': 6, 'time': 9}],
    [{'machine': 2, 'time': 9}, {'machine': 1, 'time': 3}, {'machine': 3, 'time': 5}, 
     {'machine': 5, 'time': 4}, {'machine': 6, 'time': 3}, {'machine': 4, 'time': 1}],
    [{'machine': 3, 'time': 3}, {'machine': 2, 'time': 3}, {'machine': 5, 'time': 5}, 
     {'machine': 6, 'time': 4}, {'machine': 1, 'time': 9}, {'machine': 4, 'time': 10}],
    [{'machine': 2, 'time': 3}, {'machine': 4, 'time': 4}, {'machine': 6, 'time': 9}, 
     {'machine': 1, 'time': 10}, {'machine': 5, 'time': 4}, {'machine': 3, 'time': 1}]
    ]
    machines_data_1_based = [1, 2, 3, 4, 5, 6]

    print("\nJob Shop Scheduling Problem - Pure DFS Solver (1-based input)")
    solver = JobShopPureDFS(jobs_data_1_based, machines_data_1_based)
    schedule, makespan = solver.solve()

    print(f"\nOptimal Makespan: {makespan}")
    print("\nDetailed Schedule:")
    for machine_index in sorted(schedule.keys()):
        print(f"\nMachine {machine_index}:")
        for op in sorted(schedule[machine_index], key=lambda x: x['start']):
            print(f"  Job {op['job']} Op {op['op']}: {op['start']}-{op['end']} "
                  f"(Duration: {op['end']-op['start']})")

    plot_gantt(schedule, len(machines_data_1_based))