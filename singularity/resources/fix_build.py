import os
import argparse

def get_fixed_name(old_path):
    return old_path.replace('"', '').replace('=', '-').replace('.', '-')

def fix_names_executables(folder):
    to_fix = []
    for root, dirs, files in os.walk(folder):
        for name in files:
            if (name.find('"') != -1) or (name.find('=') != -1) or (name.find('.') != -1):
                to_fix.append(os.path.join(root, name))

    for path in to_fix:
        os.rename(path, get_fixed_name(path))


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path-folder', type=str, required=True)
    return parser.parse_args()


def main():
    arg = get_arguments()
    fix_names_executables(arg.path_folder)


if __name__ == '__main__':
    main()
