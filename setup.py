
import typing
import subprocess
import json
import pathlib
import logging

from nvsmpy import _get_valid_queries

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logging.warning("Searching for queries...")
    _get_valid_queries()
    logging.warning("Setup done.")