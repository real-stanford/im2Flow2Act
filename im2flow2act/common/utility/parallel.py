def assign_task_bounds_to_gpus(n_tasks, n_gpus):
    """
    Assigns task ID bounds to GPUs as evenly as possible.

    Parameters:
    n_tasks (int): Number of tasks to be distributed.
    n_gpus (int): Number of GPUs available.

    Returns:
    list: A list of tuples where each tuple represents the lower (inclusive)
          and upper (exclusive) bounds of task IDs for that GPU.
    """
    # Calculate the base number of tasks per GPU and the remainder
    tasks_per_gpu = n_tasks // n_gpus
    remainder = n_tasks % n_gpus

    # Initialize the starting task ID
    start_id = 0

    # Distribute tasks to GPUs
    task_bounds = []
    for i in range(n_gpus):
        # Determine the number of tasks for this GPU
        num_tasks = tasks_per_gpu + (1 if i < remainder else 0)
        # Assign the bounds
        task_bounds.append((start_id, start_id + num_tasks))
        # Update the starting ID for the next GPU
        start_id += num_tasks

    return task_bounds


def assign_task_bounds_to_process(n_tasks, n_processes):
    """
    Assigns task ID bounds to GPUs as evenly as possible.

    Parameters:
    n_tasks (int): Number of tasks to be distributed.
    n_gpus (int): Number of GPUs available.

    Returns:
    list: A list of tuples where each tuple represents the lower (inclusive)
          and upper (exclusive) bounds of task IDs for that GPU.
    """
    # Calculate the base number of tasks per GPU and the remainder
    tasks_per_processes = n_tasks // n_processes
    remainder = n_tasks % n_processes

    # Initialize the starting task ID
    start_id = 0

    # Distribute tasks to GPUs
    task_bounds = []
    for i in range(n_processes):
        # Determine the number of tasks for this process
        num_tasks = tasks_per_processes + (1 if i < remainder else 0)
        # Assign the bounds
        task_bounds.append((start_id, start_id + num_tasks))
        # Update the starting ID for the next GPU
        start_id += num_tasks

    return task_bounds
