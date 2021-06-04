import os
from typing import List

import data_reader


def generate_all_archive_graphs(folder_path: str,
                                folder_save: str,
                                generation_step: int = 100,
                                max_generation: int = 10000,
                                do_put_title: bool = False,
                                list_formats: List[str] = None,
                                show_plots: bool = False,
                                exp: str = ''
                                ) -> None:
    if not list_formats:
        list_formats = ['png']

    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder {folder_path} does not exist.")

    if not os.path.exists(folder_save):
        os.makedirs(folder_save)

    for gen in range(0, max_generation + 1, generation_step):
        archive_file_path = os.path.join(folder_path, f"proj_{gen}.dat")
        try:
            data_reader.create_archive_figures(archive_file_path,
                                               folder_save,
                                               generation=gen,
                                               do_put_title=do_put_title,
                                               list_formats=list_formats,
                                               show_plots=show_plots,
                                               exp=exp)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    generate_all_archive_graphs("example/", 'images/', do_put_title=True)
