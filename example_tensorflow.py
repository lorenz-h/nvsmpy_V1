import os

import tensorflow as tf

from gpu_grabber import get_available_gpus


if __name__ == "__main__":

    gpu_id = get_available_gpus()[0]

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    #put your code here...