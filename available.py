from .nvsmpy import query

def get_free_gpu_ids(max_util=10, max_mem_util=50):

    free_gpus = []
    
    gpu_utils = query("utilization.gpu")
    mem_utils = query("utilization.memory")

    for i in range(len(gpu_utils)):
        if float(gpu_utils[i]) < max_util and float(mem_utils[i]) < max_mem_util:
            free_gpus.append(i)
    return free_gpus

def get_gpus_sorted(sort_key="memory.free", reverse=True, get_vals=False):
    free_gpus = []

    gpu_vals = [float(str) for str in query(sort_key)]
    free_gpus = [i for i, val in sorted(enumerate(gpu_vals), key=lambda x: x[1])]
    if get_vals:
        return (free_gpus, sorted(gpu_vals))
    else:
        return free_gpus