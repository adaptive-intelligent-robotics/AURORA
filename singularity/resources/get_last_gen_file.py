#!/usr/bin/env python3

import argparse
import os
import re

import glob


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-results", "-p", type=str, required=True)
    return parser.parse_args()


def get_last_gen_file(path_results: str) -> str:
    list_gen_files = glob.glob(os.path.join(path_results, "gen_*"))
    gen_numbers = [int(re.search(".*/gen_(?P<gen>[0-9]+)", gen_f).group("gen"))
                   for gen_f in list_gen_files]
    max_gen_number = max(gen_numbers)
    last_gen_file = f"gen_{max_gen_number}"
    return last_gen_file


def main():
    args = get_args()
    path_results = args.path_results

    try:
        last_gen_file = get_last_gen_file(path_results)
    except:
        last_gen_file = None

    if last_gen_file:
        path_last_gen_file = os.path.abspath(os.path.join(path_results, last_gen_file))
        print(path_last_gen_file)


if __name__ == '__main__':
    main()
