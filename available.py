from .nvsmpy import query

def get_free_gpus():

    free_gpus = []
    
    gpu_utils = query("utilization.gpu")
    mems_used = query("memory.used")
    mems_free = query("memory.free")

    for i in range(len(gpu_utils)):
        if float(gpu_utils[i]) < 0.1 and float(mems_used[i]) / (float(mems_free[i])+float(mems_used[i])) < 0.5:
            free_gpus.append(i)
    return free_gpus