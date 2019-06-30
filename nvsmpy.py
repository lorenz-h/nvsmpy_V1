
import typing
import subprocess
import logging


def _ask_for_manual_gpu_override(gpus: typing.List[dict]) -> dict:
    print("System GPUS:")
    print(str(gpus))
    inp = input("Select one of the above GPUs by ID or enter n to exit: ")
    if inp == "n":
        print("Exiting as requested by user")
        exit()
    else:
        for gpu in gpus:
            try:
                if gpu["id"] == int(inp):
                    return gpu
            except ValueError:
                pass

        print("please select a valid id")
        ask_for_manual_gpu_override(gpus)


def _parse_nvsmi_output(subprocess_output):
    dec_output = subprocess_output.decode("ASCII").replace(",", "").split()
    gpus = []
    try:
        for gpu_id, i in enumerate(range(0, len(dec_output), 3)):
            new_gpu = {
                "id": gpu_id,
                "mem_used": float(dec_output[i]),
                "mem_free": float(dec_output[i+1]),
                "utilization": float(dec_output[i+2])
            }
            gpus.append(new_gpu)
    except:
        logging.exception("Could not parse subprocess output: "+str(output))
        raise

    return gpus


def _get_system_gpus():
    cmd = f"nvidia-smi --query-gpu=memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output, _ = proc.communicate()
    all_gpus = _parse_nvsmi_output(output)
    return all_gpus


def _filter_gpus(all_gpus: typing.List[dict], max_n_gpus: int, strict: bool) -> typing.List[dict]:
    if strict:
        # use all gpus that have than 10 mb occupied and have utilization <5%
        free_gpus = [gpu for gpu in all_gpus if (gpu["mem_used"] < 10.0 and gpu["utilization"] < 5)]
    else:
        # use all gpus that have more than 50% of their memory available and have utilization <50%
        free_gpus = [gpu for gpu in all_gpus if (gpu["mem_used"]/gpu["mem_free"] < 1.0 and gpu["utilization"] < 50)]
    
    if max_n_gpus is not None:
        while len(free_gpus) > max_n_gpus:
            free_gpus.pop(-1)

    return free_gpus


def get_free_gpu_ids(max_n_gpus: int = None, strict: bool = True, allow_manual_override: bool = False) -> typing.List[int]:

    all_gpus = _get_system_gpus()
    free_gpus = _filter_gpus(all_gpus, max_n_gpus, strict)

    if len(free_gpus) == 0:
        print("No free GPUs found.")
        if allow_manual_override:
            free_gpus = [_ask_for_manual_gpu_override(all_gpus)]

    return [gpu["id"] for gpu in free_gpus]


def get_all_gpu_ids() -> typing.List[int]:
    all_gpus = _get_system_gpus()
    return [gpu["id"] for gpu in all_gpus]

def print_gpu_info(use_logging=False) -> None:
    all_gpus = _get_system_gpus()
    for gpu in all_gpus:
        if use_logging:
            logging.info(gpu)
        else:
            print(gpu)

