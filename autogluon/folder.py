import os
import shutil
import argparse

def move_files_to_root(root_dir):
    # Walk through the directory tree
    for dirpath, _, filenames in os.walk(root_dir):
        # Skip the root folder itself
        if os.path.abspath(dirpath) == os.path.abspath(root_dir):
            continue
        for filename in filenames:
            src = os.path.join(dirpath, filename)
            dst = os.path.join(root_dir, filename)
            # If a file with the same name exists in the root, rename the file
            if os.path.exists(dst):
                base, ext = os.path.splitext(filename)
                counter = 1
                new_filename = f"{base}_{counter}{ext}"
                dst = os.path.join(root_dir, new_filename)
                while os.path.exists(dst):
                    counter += 1
                    new_filename = f"{base}_{counter}{ext}"
                    dst = os.path.join(root_dir, new_filename)
            # print(f"Moving: {src} -> {dst}")
            shutil.move(src, dst)

def remove_empty_dirs(root_dir):
    # Walk through the directory tree from the bottom up and remove empty directories
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if dirpath == root_dir:
            continue
        if not dirnames and not filenames:
            # print(f"Removing empty directory: {dirpath}")
            os.rmdir(dirpath)

def main():
    parser = argparse.ArgumentParser(
        description="Recursively move all files from subdirectories into the outer (root) folder."
    )
    parser.add_argument('-f', '--folder', type=str, required=True)
    parser.add_argument("-r", "--remove-empty-dirs", action="store_true",)
    args = parser.parse_args()

    root_dir = os.path.abspath(args.folder)
    if not os.path.isdir(root_dir):
        print(f"Error: '{root_dir}' is not a valid directory.")
        return

    move_files_to_root(root_dir)
    if args.remove_empty_dirs:
        remove_empty_dirs(root_dir)

if __name__ == "__main__":
    main()