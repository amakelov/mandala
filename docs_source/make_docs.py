"""
convert .ipynb files in this directory to .md files and move them to the docs
directory for mkdocs to use
"""

import os
import argparse

# parse the command line arguments
parser = argparse.ArgumentParser(description='Convert .ipynb files to .md files')
parser.add_argument('--filenames', type=str, nargs='+', help='list of filenames to convert')
args = parser.parse_args()


if __name__ == '__main__':
    if args.filenames:
        ipynb_files = args.filenames
        # prepend "./" to the filenames
        ipynb_files = ['./' + f for f in ipynb_files if not f.startswith('./')]
    else:
        # find all .ipynb files recursively in the current directory and its subdirectories
        ipynb_files = []
        for root, dirs, files in os.walk('.'):
            for f in files:
                if f.endswith('.ipynb'):
                    ipynb_files.append(os.path.join(root, f))

    for f in ipynb_files:
        os.system('jupyter nbconvert --to notebook --execute --inplace ' + f)
        os.system(f"jupyter nbconvert --to markdown {f}")

    DOCS_REL_PATH = '../docs/docs/'
    # now, move the .md files to the docs directory
    for f in ipynb_files:
        # find the relative directory and the filename
        relative_dir = os.path.dirname(f)
        fname = os.path.basename(f)
        # if the target dir doesn't exist, create it
        if not os.path.isdir(DOCS_REL_PATH + relative_dir):
            os.system("mkdir -p " + DOCS_REL_PATH + relative_dir)
        # move to the DOCS_REL_PATH, under the same directory structure
        mv_cmd = "mv " + f.replace('.ipynb', '.md') + " " + DOCS_REL_PATH + relative_dir + '/' + fname.replace('.ipynb', '.md')
        print(mv_cmd)
        os.system(mv_cmd)

    # also, move any directories named "{fname}_files" to the docs directory
    for f in ipynb_files:
        files_folder = f.replace('.ipynb', '_files')
        if os.path.isdir(files_folder):
            # first, remove the directory if it already exists
            target_files_path = DOCS_REL_PATH + files_folder
            if os.path.isdir(target_files_path):
                os.system(f"rm -r {DOCS_REL_PATH}" + files_folder)
            # then, move the directory
            os.system("mv " + f.replace('.ipynb', '_files') + " " + DOCS_REL_PATH + f.replace('.ipynb', '_files'))