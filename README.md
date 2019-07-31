# nvsmpy
This module makes it easy to find available gpus in a multi gpu system. It was tested on Ubuntu 16.04 and Windows 10. The most common use case is machine learning on a gpu server shared among multiple users. It calls nvidia-smi in a subprocess and parses it's output to find all gpus which are currently unused.

## Requirements
- Python >= 3.6 (Most likely backwards compatible to all versions of 3.x and if you remove f-strings and type hints)
- [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface) (Tested with versions 410.104 and 425.25)

## Usage
You can use nvsmpy find unoccupied GPUs in your system. What the criteria for this are defined in [available.py](available.py).
```python
from nvsmpy import get_free_gpu_ids

free_gpus = get_free_gpu_ids()
```

Using nvsmpy you can query nvidia-smi directly. To generate a list of queries supported by your system run [setup.py](setup.py), which will create a json file in the nvsmpy folder.
```python
from nvsmpy import query
utils = query("temperature.memory")
```
Furthermore you can display the nice system overview that nvidia-smi generates:
```python
from nvsmpy import print_gpu_stats
print_gpu_stats()
```


