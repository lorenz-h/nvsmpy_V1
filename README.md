# nvsmpy
This module makes it easy to find available gpus in a multi gpu system. It was tested on Ubuntu 16.04 and Windows 10. The most common use case is machine learning on a gpu server shared among multiple users. It calls nvidia-smi in a subprocess and parses it's output to find all gpus which are currently unused. On systems that use the GPU for display output the strict parameter should be set to false. You can also allow manual overriding, should no completely unused GPUs be found.

## Requirements
- Python >= 3.6 (Most likely backwards compatible to all versions of 3.x and if you remove f-strings and type hints)
- [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface) (Tested with versions 410.104 and 425.25)

## Usage
```python
from nvsmpy import get_free_gpu_ids

free_gpus = get_free_gpu_ids()
```



