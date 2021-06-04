#!/usr/bin/env python3

import argparse

import bulk_operations

# TODO : Add Help Messages
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--save', type=str, required=True)
parser.add_argument('--formats', type=str, default=['png'], nargs="+")

parser.add_argument('--gen-step', type=int, default=100)
parser.add_argument('--gen-max', type=int, default=10000)

parser.add_argument('--show-titles', action='store_true')
parser.add_argument('--show-plots', action='store_true')

parser.add_argument('--exp', type=str, required=True)

args = parser.parse_args()

bulk_operations.generate_all_archive_graphs(folder_path=args.path,
                                            folder_save=args.save,
                                            generation_step=args.gen_step,
                                            max_generation=args.gen_max,
                                            do_put_title=args.show_titles,
                                            list_formats=args.formats,
                                            show_plots=args.show_plots,
                                            exp=args.exp)
