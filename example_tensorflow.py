import os

import tensorflow as tf

from nvsmpy.available import get_free_gpus


if __name__ == "__main__":

    gpu_id = get_free_gpus()[0]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    #put your code here...