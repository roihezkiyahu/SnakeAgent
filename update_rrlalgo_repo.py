import os
import subprocess

REPO_URL = 'https://github.com/roihezkiyahu/RLAlgorithms.git'
REPO_DIR = 'RLAlgorithms'

def clone_or_update_repo(repo_url, repo_dir):
    if os.path.exists(repo_dir):
        print(f"Updating existing repository in {repo_dir}...")
        subprocess.run(['git', '-C', repo_dir, 'pull'], check=True)
    else:
        print(f"Cloning repository from {repo_url} to {repo_dir}...")
        subprocess.run(['git', 'clone', repo_url, repo_dir], check=True)

if __name__ == '__main__':
    clone_or_update_repo(REPO_URL, REPO_DIR)
