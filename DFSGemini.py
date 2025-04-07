import copy
import matplotlib.pyplot as plt
from collections import defaultdict

class JobShopPureDFS:
    def __init__(self, jobs, machines, log_file="pure_dfs_paths.log"):
        self.jobs = jobs
        self.machines = machines
        self.best_makespan = float('inf')
        self.best_schedule = None
        self._init_helper_structures()
        self.state_counter = 0
        self.log_file = log_file
        self.search_log = []

        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write("Job Shop Scheduling Search Path Log (Pure DFS)\n")
            f.write("===============================================\n\n")
            f.write(f"Problem Configuration:\n")
            f.write(f"- Jobs: {len(jobs)}\n")
            f.write(f"- Machines: {len(machines)}\n")
            f.write(f"- Total operations: {sum(len(job) for job in jobs)}\n\n")

    def _init_helper_structures(self):
        self.job_op_counts = [len(job) for job in self.jobs]
        self.machine_indices = {m: i for i, m in enumerate(self.machines)}
        self.all_ops = [(ji, oi, op)
                       for ji, job in enumerate(self.jobs)
                       for oi, op in enumerate(job)]

    def _log_state(self, state, action=None, pruned=False):
        """Log the current state of the search"""
        log_entry = {
            'state_id': len(self.search_log) + 1,
            'current_makespan': state['current_makespan'],
            'best_makespan': self.best_makespan,
            'scheduled_ops': len(state['scheduled_ops']),
            'total_ops': len(self.all_ops),
            'action': action,
            'pruned': pruned,
            'machine_schedules': copy.deepcopy(state['machine_schedules'])
        }
        self.search_log.append(log_entry)

        with open(self.log_file, 'a') as f:
            f.write(f"\n=== State #{log_entry['state_id']} ===\n")
            f.write(f"Current makespan: {state['current_makespan']}\n")
            f.write(f"Best makespan so far: {self.best_makespan}\n")
            f.write(f"Progress: {len(state['scheduled_ops'])}/{len(self.all_ops)} operations scheduled\n")

            if action:
                if pruned:
                    f.write("ACTION: Pruned this branch (would exceed best makespan)\n")
                elif action == "Initial state":
                    f.write("ACTION: Starting search\n")
                elif action == "Complete schedule found":
                    f.write("ACTION: Found complete schedule\n")
                else:
                    job_idx, op_idx, op = action
                    f.write(f"ACTION: Scheduled Job {job_idx} Operation {op_idx} on Machine {op['machine']} "
                           f"(Duration: {op['time']})\n")

            f.write("\nCurrent Schedule:\n")
            for machine in sorted(state['machine_schedules'].keys()):
                f.write(f"Machine {machine}:\n")
                for op_info in state['machine_schedules'][machine]:
                    f.write(f"  Job {op_info['job']} Op {op_info['op']}: {op_info['start']}-{op_info['end']} "
                           f"(Duration: {op_info['end']-op_info['start']})\n")

            if len(state['scheduled_ops']) == len(self.all_ops):
                f.write("\nCOMPLETE SCHEDULE EVALUATED\n")

    def solve(self):
        initial_state = {
            'machine_schedules': {m: [] for m in self.machines},
            'scheduled_ops': set(),
            'op_end_times': {},
            'current_makespan': 0
        }

        self._log_state(initial_state, action="Initial state")

        stack = [initial_state]

        while stack:
            current = stack.pop()

            if self._is_complete(current):
                if self._update_best(current):
                    self._log_state(current, action="Complete schedule found")
                continue

            candidates = self._get_candidates(current)

            for job_idx, op_idx, op in candidates:
                new_state = self._apply_move(current, job_idx, op_idx, op)

                if not self._should_prune(new_state):
                    stack.append(new_state)
                    self._log_state(new_state, action=(job_idx, op_idx, op))
                else:
                    self._log_state(new_state, action=(job_idx, op_idx, op), pruned=True)

        self._write_summary_report()
        return self.best_schedule, self.best_makespan

    def _write_summary_report(self):
        with open(self.log_file, 'a') as f:
            f.write("\n\n=== SEARCH SUMMARY ===\n")
            f.write(f"Total states explored: {len(self.search_log)}\n")
            f.write(f"Optimal makespan found: {self.best_makespan}\n")

            complete_schedules = [log for log in self.search_log if log['scheduled_ops'] == log['total_ops']]
            f.write(f"Number of complete schedules evaluated: {len(complete_schedules)}\n")

            if complete_schedules:
                f.write("\nComplete schedules found:\n")
                for i, log in enumerate(complete_schedules, 1):
                    f.write(f"{i}. State #{log['state_id']}: Makespan = {log['current_makespan']} "
                           f"{'(BEST)' if log['current_makespan'] == self.best_makespan else ''}\n")

    def _is_complete(self, state):
        return len(state['scheduled_ops']) == len(self.all_ops)

    def _get_candidates(self, state):
        candidates = []
        scheduled_ops = state['scheduled_ops']

        for job_idx, op_idx, op in self.all_ops:
            if (job_idx, op_idx) in scheduled_ops:
                continue

            if op_idx > 0 and (job_idx, op_idx - 1) not in scheduled_ops:
                continue

            candidates.append((job_idx, op_idx, op))

        return candidates

    def _apply_move(self, state, job_idx, op_idx, op):
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
            'job': job_idx,
            'op': op_idx,
            'start': start_time,
            'end': end_time
        }
        new_state['machine_schedules'][machine].append(scheduled_op)
        new_state['scheduled_ops'].add((job_idx, op_idx))
        new_state['op_end_times'][(job_idx, op_idx)] = end_time
        new_state['current_makespan'] = max(new_state['current_makespan'], end_time)

        return new_state

    def _estimate_makespan(self, state):
        machine_ends = {m: (state['machine_schedules'][m][-1]['end'] if state['machine_schedules'][m] else 0)
                       for m in self.machines}

        remaining_work = {m: 0 for m in self.machines}

        for job_idx, op_idx, op in self.all_ops:
            if (job_idx, op_idx) not in state['scheduled_ops']:
                machine = op['machine']
                remaining_work[machine] += op['time']

        return max(machine_ends[m] + remaining_work[m] for m in self.machines)

    def _should_prune(self, state):
        return self._estimate_makespan(state) >= self.best_makespan

    def _update_best(self, state):
        if state['current_makespan'] < self.best_makespan:
            self.best_makespan = state['current_makespan']
            self.best_schedule = copy.deepcopy(state['machine_schedules'])
            return True
        return False

def plot_gantt(schedule, n_machines):
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['skyblue', 'lightgreen', 'salmon', 'plum', 'orange']

    for machine in schedule:
        for op in schedule[machine]:
            ax.barh(y=machine, width=op['end']-op['start'], left=op['start'],
                    height=0.5, align='center',
                    color=colors[op['job'] % len(colors)],
                    edgecolor='black')
            ax.text(op['start'] + (op['end']-op['start'])/2, machine,
                    f'J{op["job"]}O{op["op"]}',
                    va='center', ha='center', color='black', fontsize=8)

    ax.set_yticks(range(n_machines))
    ax.set_yticklabels([f'M{m}' for m in range(n_machines)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machine")
    ax.set_title("Gantt Chart for Best JSSP Schedule")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    jobs = [
    # Job 0: Machine sequence 1-3-2 with times 2-4-1
    [{'machine': 0, 'time': 2}, {'machine': 2, 'time': 4}, {'machine': 1, 'time': 1}],
    
    # Job 1: Machine sequence 2-1-3 with times 3-1-5
    [{'machine': 1, 'time': 3}, {'machine': 0, 'time': 1}, {'machine': 2, 'time': 5}],
    
    # Job 2: Machine sequence 3-2-1 with times 2-2-3
    [{'machine': 2, 'time': 2}, {'machine': 1, 'time': 2}, {'machine': 0, 'time': 3}],
    
    # Job 3: Machine sequence 1-2-3 with times 4-1-2
    [{'machine': 0, 'time': 4}, {'machine': 1, 'time': 1}, {'machine': 2, 'time': 2}]
    ]
    machines = [0, 1, 2]

    print("\nJob Shop Scheduling Problem - Pure DFS Solver with Search Logging")
    solver = JobShopPureDFS(jobs, machines, log_file="pure_dfs_paths.log")
    schedule, makespan = solver.solve()

    print(f"\nOptimal Makespan: {makespan}")
    print("\nDetailed Schedule:")
    for machine in machines:
        print(f"\nMachine {machine}:")
        for op in sorted(schedule[machine], key=lambda x: x['start']):
            print(f"  Job {op['job']} Op {op['op']}: {op['start']}-{op['end']} "
                  f"(Duration: {op['end']-op['start']})")

    plot_gantt(schedule, len(machines))