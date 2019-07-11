from .nvsmpy import query

def get_free_gpus(max_util=0.1, max_mem_util=0.5):

    free_gpus = []
    
    gpu_utils = query("utilization.gpu")
    mem_utils = query("utilization.memory")

    for i in range(len(gpu_utils)):
        if float(gpu_utils[i]) < max_util and float(mem_utils[i]) < max_mem_util:
            free_gpus.append(i)
    return free_gpus