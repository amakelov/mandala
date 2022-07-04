import subprocess, argparse

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    cmd = ['ipython', '-i', 'console.py', '--']
    subprocess.call(cmd)