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
    fetch: bool = True


repos: List[Repo] = [
    Repo(
        'calamari',
        'https://github.com/Calamari-OCR/calamari.git',
        '3b9d17f2f42f98c752205c5fdcf546a129f2edd3',
    ),
    Repo(
        'ommr4all-line-detection',
        'https://gitlab-ci-token:{}@gitlab2.informatik.uni-wuerzburg.de/OMMR4all/ommr4all-line-detection.git'.format(CI_JOB_TOKEN),
        '762d18b890efee5ddd7d8038ae8981378acf6a6a'
    ),
    Repo(
        'ommr4all-layout-analysis',
        'https://gitlab-ci-token:{}@gitlab2.informatik.uni-wuerzburg.de/OMMR4all/ommr4all-layout-analysis.git'.format(CI_JOB_TOKEN),
        '0783057e13d1891aeb015c29c93247952c000e92',
    ),
    Repo(
        'page-segmentation',
        'https://gitlab-ci-token:{}@gitlab2.informatik.uni-wuerzburg.de/ls6/page-segmentation.git'.format(CI_JOB_TOKEN),
        '1289df8b536395368b044b07d019ce5faebe2ea1',
        fetch=False,
    ),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode')
    parser.add_argument('--gpu', default=False, action='store_true')

    args = parser.parse_args()

    os.chdir(root_dir)

    if args.mode == 'setup':
        subprocess.check_call([pip, 'install', 'tensorflow' if not args.gpu else 'tensorflow_gpu'])
        subprocess.check_call([pip, 'install', '-r', 'requirements.txt'])
        subprocess.check_call([python, 'manage.py', 'migrate'])

        with tempfile.TemporaryDirectory() as d:
            for repo, url, hash, fetch in repos:
                os.chdir(d)
                os.makedirs(repo, exist_ok=True)
                os.chdir(repo)
                subprocess.check_call(['git', 'init'])
                subprocess.check_call(['git', 'remote', 'add', 'origin', url])
                if fetch:
                    subprocess.check_call(['git', 'fetch', '--depth', '1', 'origin', hash])
                    subprocess.check_call(['git', 'reset', '--hard', 'FETCH_HEAD'])
                else:
                    subprocess.check_call(['git', 'fetch', 'origin'])
                    subprocess.check_call(['git', 'reset', '--hard', hash])
                subprocess.check_call([python, 'setup.py', 'install'])
    elif args.mode == 'run':
        subprocess.check_call([python, '-u', 'manage.py', 'test'])
    else:
        raise ValueError("Mode must be setup or run, got {}".format(args.mode))


if __name__ == '__main__':
    main()
