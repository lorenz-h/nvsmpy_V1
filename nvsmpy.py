import typing
import subprocess
import pathlib
import json
import logging


def _parse_valid_queries() -> typing.Dict:
    quote_chars = ["\'", "\""]

    cmd = f"nvidia-smi --help-query-gpu"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output, _ = proc.communicate()
    dec_output = output.decode("utf-8")
    sections = dec_output.split("\r\n\r\n")
    
    valid_queries = {}
    for section in sections:
        if len(section) != 0:
            if section[0][0] in quote_chars:
                subsections = section.split("\r\n")
                if subsections[0][-1] in quote_chars:
                    for qc in quote_chars:  # remove all quote chars from output
                        subsections[0] = subsections[0].replace(qc, "")
                    query = subsections[0].split(" or ")
                    description = subsections[1]

                    for synonym in query:
                        valid_queries[synonym] = description
    
    return valid_queries


def _get_valid_queries():
    valid_queries_file = pathlib.Path(__file__).parent / "valid_queries.json"
    if not valid_queries_file.is_file():
        logging.debug("nvsmpy parsing the valid queries directly from nvidia-smi")
        valid_queries = _parse_valid_queries()
        with open(valid_queries_file, "w+") as out_file:
            logging.debug("nvsmpy saving the valid queries to json file")
            json.dump(valid_queries, out_file)
    else:
        with open(valid_queries_file, "r") as conf_file:
            logging.debug("nvsmpy loading the valid queries from json file")
            valid_queries = json.load(conf_file)
    globals()["VALID_QUERIES"] = set(valid_queries) 


def get_n_system_gpus() -> int:
    cmd = f"nvidia-smi --query-gpu=count --format=csv,noheader,nounits"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, _ = proc.communicate()
    dec_output = output.decode("utf-8").replace("\n", "").replace("\r", "")
    gpu_count = int(dec_output)
    logging.debug(f"nvsmpy found {gpu_count} gpus in system.")
    globals()["N_SYSTEM_GPUS"] = gpu_count
    return gpu_count


def _init_globals():
    try:
        VALID_QUERIES
    except NameError:
        logging.debug("nvsmpy loading valid nvidia-smi queries...")
        _get_valid_queries()
    try:
        N_SYSTEM_GPUS
    except NameError:
        logging.debug("nvsmpy counting number of system GPUS...")
        get_n_system_gpus()


def query(*queries):
    _init_globals()

    for query in queries:
        assert query in VALID_QUERIES, f"Invalid query {query} requested."
        
    queries_str: str = ",".join(queries)
    
    cmd = f"nvidia-smi --query-gpu={queries_str} --format=csv,noheader,nounits"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output, _ = proc.communicate()
    dec_output = output.decode("utf-8").replace(" ", "").replace("\n", "").replace("\r", "").split(",")
    
    assert len(dec_output) == len(queries) * N_SYSTEM_GPUS, f"Failed to parse gpu information. Size mismatch between the number of queries and number of gpus"
    return dec_output


def print_gpu_stats():
    cmd = f"nvidia-smi"
    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, _ = proc.communicate()
    dec_output = output.decode("utf-8")
    print(dec_output)
