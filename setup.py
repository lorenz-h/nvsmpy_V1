
import typing
import subprocess
import json
import pathlib



if __name__ == "__main__":
    print("Setting up environment...")
    print("Loading valid queries...")
    valid_queries = get_all_queries()
    print("Saving valid queries to file...")
    pkg_dir = pathlib.Path(__file__).parent

    with open(pkg_dir / "valid_queries.json", "w+") as out_file:
        json.dump(valid_queries, out_file)
    print("Setup done.")