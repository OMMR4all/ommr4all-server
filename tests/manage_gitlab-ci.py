import argparse
import subprocess
import sys
import os
import tempfile
from typing import NamedTuple, List

this_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, '..'))
python = sys.executable
pip = os.path.join(os.path.dirname(python), 'pip')
CI_JOB_TOKEN = os.environ.get('CI_JOB_TOKEN')


class Repo(NamedTuple):
    name: str
    url: str
    hash: str


repos: List[Repo] = [
    Repo(
        'calamari',
        'https://github.com/Calamari-OCR/calamari.git',
        'eba8e9c35d2c301319cc9cb15d25124460aee2db',
    ),
    Repo(
        'ommr4all-line-detection',
        'https://github.com/OMMR4all/ommr4all-line-detection.git',
        '3d040e723610fc242c790ecc91ba6d3a52424894'
    ),
    Repo(
        'ommr4all-layout-analysis',
        'https://github.com/OMMR4all/ommr4all-layout-analysis.git',
        '73897fa31561ff670f95e512e6cb8f63f921ab20',
    ),
    Repo(
        'ommr4all-page-segmentation',
        'https://github.com/OMMR4all/ommr4all-page-segmentation.git',
        '6894d996e1b06be0c9f81e9fc91059acc286c184',
    ),
    Repo(
        'nautilus',
        'https://github.com/Gawajn/nautilus.git',
        'cff1f9cd53dda3d143dc14f07c4c3406cf0c726f',
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('--gpu', default=False, action='store_true')

    args = parser.parse_args()

    os.chdir(root_dir)

    if args.mode == 'setup':
        subprocess.check_call([pip, 'install', 'tensorflow~=2.4.0' if not args.gpu else 'tensorflow_gpu~=2.4.0'])
        subprocess.check_call([pip, 'install', '-r', 'requirements.txt'])

        with tempfile.TemporaryDirectory() as d:
            for repo, url, hash in repos:
                os.chdir(d)
                os.makedirs(repo, exist_ok=True)
                os.chdir(repo)
                subprocess.check_call(['git', 'init'])
                subprocess.check_call(['git', 'remote', 'add', 'origin', url])
                subprocess.check_call(['git', 'fetch', 'origin'])
                subprocess.check_call(['git', 'reset', '--hard', hash])
                subprocess.check_call([python, 'setup.py', 'install'])

        os.chdir(root_dir)
        subprocess.check_call([python, 'manage.py', 'migrate'])
    elif args.mode == 'run':
        subprocess.check_call([python, '-u', 'manage.py', 'test'])
    else:
        raise ValueError("Mode must be setup or run, got {}".format(args.mode))


if __name__ == '__main__':
    main()
