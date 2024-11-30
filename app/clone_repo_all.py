import csv
import os
from subprocess import call
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

def clone_repo(git_url, clone_path):
    result = call(['git', 'clone', git_url, clone_path])
    return result == 0


def clone_repository(repo_info, download_directory):
    repo_name = repo_info['name']
    user, repo = repo_name.split('/')
    git_url = f"https://github.com/{user}/{repo}.git"
    clone_path = os.path.join(download_directory, f"temp_{repo}")

    # Ensure the temporary directory exists
    if not os.path.exists(clone_path):
        os.makedirs(clone_path)

    cloned = clone_repo(git_url, clone_path)
    if not cloned:
        print(f"ffailed to clone {repo_name}")
        return False, repo_name
    
    print(f"successfully cloned {repo_name} to {clone_path}")
    return True, repo_name

def download_repositories(all_apps_path, download_directory, max_workers=5):
    with open(all_apps_path, mode='r') as infile:
        reader = csv.DictReader(infile)
        rows = list(reader)
        print(f"Initial number of rows: {len(rows)}")  

    failed_repos = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_repo = {executor.submit(clone_repository, row, download_directory): row for row in rows}
        for future in as_completed(future_to_repo):
            success, repo_name = future.result()
            if not success:
                failed_repos.append(repo_name)
                print(f"Error cloning {repo_name}")

    
    if failed_repos:
        print("\nfailed to clone:")
        for repo in failed_repos:
            print(repo)
    else:
        print("\nAll cloned successfully.")

if __name__ == "__main__":
    # paths
    all_apps_path = '/microDec/data/final_results/all_apps.csv'
    download_directory = '/microDec/all_repo'

   
    max_workers = 5

    download_repositories(all_apps_path, download_directory, max_workers)
