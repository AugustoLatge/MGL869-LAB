import subprocess
import os
import re
from collections import Counter
from datetime import datetime

from numpy.ma.extras import average


def is_valid_path(path):
    try:
        if not os.path.exists(path):
            print("Path does not exist.")
            return False
        if not os.access(path, os.R_OK):
            print("Path is not readable.")
            return False
        return True
    except Exception as e:
        print(f"Invalid path: {e}")
        return False

def get_commits(repo_path, issues, current_version_sha):
    found = {}
    try:
        print(f"Repo path: {repo_path}")
        if not is_valid_path(repo_path):
            return []

        result = subprocess.run(
            ['git', '-C', str(repo_path), 'checkout', current_version_sha],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        result.check_returncode()

        # Run the git log command with specified encoding
        result = subprocess.run(
            ['git', '-C', str(repo_path), 'log', '--pretty=format:%H %s'],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        result.check_returncode()  # Check if the command was successful

        # Split the output into lines
        commits = result.stdout.strip().split('\n')
        commit_list = []
        for line in commits:
            sha, message = line.split(' ', 1)

            for key in issues:
                if key in message:
                    # Get the list of modified files for each commit with specified encoding
                    files_result = subprocess.run(
                        ['git', '-C', str(repo_path), 'show', '--pretty=format:', '--name-only', sha],
                        capture_output=True,
                        text=True,
                        encoding='utf-8',
                        check=True
                    )
                    files_result.check_returncode()  # Check if the command was successful
                    files = files_result.stdout.strip().split('\n')
                    commit_list.append({'key': key,'sha': sha, 'message': message, 'priority': issues[key]["priority"], 'files': files})
                    found [key] = line
                    del issues[key]
                    break
        print("Commits extracted successfully")
        return commit_list
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while fetching commits:\n{e}")
        return []
    except Exception as e:
        print(f"An unexpected error occurred:\n{e}")
        return []

def attach_body_to_message(commits):
    attached = [commits.pop(0)]
    for commit in commits:
        if re.search(r'^\w{40}(?=\|\|\|)', commit):
            attached.append(commit)
        elif commit:
            attached[-1] += f" {commit}"
    return attached

def get_sha_date_author_message(commit):
    split_commit = commit.split("|||")
    return split_commit[0], split_commit[1], split_commit[2], split_commit[3]

def get_java_files_info_in_version(repo_path, sha):
    # Checkout commit
    result = subprocess.run(
        ['git', '-C', str(repo_path), 'checkout', sha],
        capture_output=True,
        text=True,
        encoding='utf-8',
        check=True
    )
    result.check_returncode()
    files_info = {}
    for file in repo_path.rglob("*.java"):
        with open(file, "rb") as f:
            files_info[file.name] = {
                "number_of_lines": sum(1 for _ in f),
                "number_of_lines_added": 0,
                "number_of_lines_removed": 0,
                "changed_by_n_commits_in_current_version": 0,
                "changed_by_n_commits_in_current_and_previous_versions": 0,
                "bugs_fixed_in_n_commits": 0,
                "authors_in_current_version": [],
                "authors_in_current_and_previous_versions": [],
                "authors_average_expertise": 0,
                "authors_minimal_expertise": 0,
                "commits_dates_for_current_version": [],
                "commits_dates_for_current_and_previous_versions": [],
                "average_commit_time_for_current_version": 0,
                "average_commit_time_for_current_and_previous_versions": 0,
                "changed_comments": 0,
                "unchanged_comments": 0,
            }
    return files_info

def calculate_number_of_lines(files_info, previous_version_files_info):
    for file_name, file_info in files_info.items():
        if file_name in previous_version_files_info:
            diff_lines = file_info["number_of_lines"] - previous_version_files_info[file_name]["number_of_lines"]
            file_info["number_of_lines_added"] = diff_lines if diff_lines >= 0 else 0
            file_info["number_of_lines_removed"] = -diff_lines if diff_lines < 0 else 0
        else:
            file_info["number_of_lines_added"] = file_info["number_of_lines"]
            file_info["number_of_lines_removed"] = 0

def calculate_authors_expertise(files_info, authors_tracking):
    authors_counts = Counter(authors_tracking)
    for file_name, file_info in files_info.items():
        expertises = []
        for author in set(file_info["authors_in_current_version"]):
            expertises.append(authors_counts[author])
        file_info["authors_average_expertise"] = int(round(average(expertises), 0)) if expertises else 0
        file_info["authors_minimal_expertise"] = int(round(min(expertises), 0)) if expertises else 0

def calculate_average_commit_time(files_info):
    for file_name, file_info in files_info.items():
        dates_diff_for_current_version = []
        dates_diff_for_current_and_previous_versions = []
        commits_dates_for_current_version = sorted(
            [datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z") for date in file_info["commits_dates_for_current_version"]], reverse=True)
        commits_dates_for_current_and_previous_versions = sorted(
            [datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z") for date in file_info["commits_dates_for_current_and_previous_versions"]], reverse=True)
        for i, date in enumerate(commits_dates_for_current_version):
            if i < len(commits_dates_for_current_version) - 1:
                date1 = date
                date2 = commits_dates_for_current_version[i + 1]
                dates_diff_for_current_version.append((date1 - date2).days)
        for i, date in enumerate(commits_dates_for_current_and_previous_versions):
            if i < len(commits_dates_for_current_and_previous_versions) - 1:
                date1 = date
                date2 = commits_dates_for_current_and_previous_versions[i + 1]
                dates_diff_for_current_and_previous_versions.append((date1 - date2).days)
        file_info["average_commit_time_for_current_version"] = int(round(average(dates_diff_for_current_version), 0)) if dates_diff_for_current_version else 0
        file_info["average_commit_time_for_current_and_previous_versions"] = int(round(average(dates_diff_for_current_and_previous_versions), 0)) if dates_diff_for_current_and_previous_versions else 0


def extract_comment_changes(repo_path, files_info, current_version_sha, previous_version_sha):
    #Run the git log -p command and grep comment lines
    # Get only the commits that are in the current version and not in the previous
    git_log_result = subprocess.run(
        ['git', '-C', str(repo_path), 'log', '-p', f'{previous_version_sha}..{current_version_sha}'],
        capture_output=True,
    )
    grep_result = subprocess.run(
        ['grep', '-IE', r'^commit\s[a-f0-9]{40}|^diff\s--git|^\+\s*\/\/|^\-\s*\/\/|^\+\s*\/\*\*|^\+\s*\*|^\+.*\*\/$|^\-\s*\/\*\*|^\-\s*\*|^\-.*\*\/$'],
        input=git_log_result.stdout,
        capture_output=True,
    )

    # The first item is an empty string, skip it
    commits_diffs = re.split(r"^commit\s[a-f0-9]{40}", grep_result.stdout.decode('utf-8'), flags=re.MULTILINE)[1:]

    for commit_diffs in commits_diffs:
        # The first item is an empty string, skip it
        diffs = commit_diffs.split("diff --git ")[1:]
        for diff in diffs:
            items = diff.split("\n")
            file_name = None
            unchanged_comments = True
            is_java_file = False
            for i, item in enumerate(items):
                if i == 0:
                    if not item.endswith(".java"):
                        break
                    is_java_file = True
                    file_name = item.split("/")[-1]
                elif item != "" and file_name in files_info:
                    files_info[file_name]["changed_comments"] += 1
                    unchanged_comments = False
            if is_java_file and unchanged_comments and file_name in files_info:
                files_info[file_name]["unchanged_comments"] += 1


def extract_previous_version_missing_files_info(repo_path, files_info, authors_tracking, previous_version_sha):
    # Run the git log command with specified encoding
    # Use second date as flag for author name extraction
    # Get only the commits that are in the previous version
    result = subprocess.run(
        ['git', '-C', str(repo_path), 'log', '--pretty=format:%H|||%aD|||%an|||%s %b', previous_version_sha],
        capture_output=True,
        text=True,
        encoding='utf-8',
        check=True
    )
    result.check_returncode()  # Check if the command was successful

    commits = attach_body_to_message(result.stdout.strip().split('\n'))
    for i, commit in enumerate(commits):
        print(f"Previous version - {i}")

        sha, date, author, message = get_sha_date_author_message(commit)

        if not re.search("HIVE|Hive|hive", message):
            # Focus on HIVE commits
            continue

        authors_tracking.append(author)

        # Get the list of modified files for each commit with specified encoding
        files_result = subprocess.run(
            ['git', '-C', str(repo_path), 'show', '--pretty=format:', '--name-only', sha],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        files_result.check_returncode()  # Check if the command was successful
        # Get .java files
        files = [f for f in files_result.stdout.strip().split('\n') if f.endswith('.java')]

        for file in files:
            file_name = file.split("/")[-1]
            if file_name in files_info:
                file_info = files_info[file_name]
                # Keep track of the number of commits that changed the file
                file_info["changed_by_n_commits_in_current_and_previous_versions"] += 1
                # Keep track of authors
                file_info["authors_in_current_and_previous_versions"].append(author)
                # Keep track of commits dates
                file_info["commits_dates_for_current_and_previous_versions"].append(date)


def extract_new_metrics(repo_path, current_version_sha, previous_version_sha):
    files_keys_found_in_version = []
    files_info = get_java_files_info_in_version(repo_path, current_version_sha)
    previous_version_files_info = get_java_files_info_in_version(repo_path, previous_version_sha)
    calculate_number_of_lines(files_info, previous_version_files_info)

    authors_tracking = []

    extract_comment_changes(repo_path, files_info, current_version_sha, previous_version_sha)

    extract_previous_version_missing_files_info(repo_path, files_info, authors_tracking, previous_version_sha)

    # Run the git log command with specified encoding
    # Get only the commits that are in the current version and not in the previous
    result = subprocess.run(
        ['git', '-C', str(repo_path), 'log', '--pretty=format:%H|||%aD|||%an|||%s %b', f'{previous_version_sha}..{current_version_sha}'],
        capture_output=True,
        text=True,
        encoding='utf-8',
        check=True
    )
    result.check_returncode()  # Check if the command was successful

    # Split the output into lines
    commits = attach_body_to_message(result.stdout.strip().split('\n'))
    for i, commit in enumerate(commits):
        print(f"Current version - {i}")

        sha, date, author, message = get_sha_date_author_message(commit)

        if not re.search("HIVE|Hive|hive", message):
            # Focus on HIVE commits
            continue

        authors_tracking.append(author)

        # Get the list of modified files for each commit with specified encoding
        files_result = subprocess.run(
            ['git', '-C', str(repo_path), 'show', '--pretty=format:', '--name-only', sha],
            capture_output=True,
            text=True,
            encoding='utf-8',
            check=True
        )
        files_result.check_returncode()  # Check if the command was successful
        # Get .java files
        files = [f for f in files_result.stdout.strip().split('\n') if f.endswith('.java')]

        for file in files:
            file_name = file.split("/")[-1]
            if file_name in files_info:
                files_keys_found_in_version.append(file_name)
                file_info = files_info[file_name]
                # Keep track of the number of commits that changed the file
                file_info["changed_by_n_commits_in_current_version"] += 1
                file_info["changed_by_n_commits_in_current_and_previous_versions"] += 1
                # Check if the commit is a bug fix
                if any(fix_indicator in message for fix_indicator in ("fix", "repair", "recover", "restore")):
                    file_info["bugs_fixed_in_n_commits"] += 1
                # Keep track of authors
                file_info["authors_in_current_version"].append(author)
                file_info["authors_in_current_and_previous_versions"].append(author)
                # Keep track of commits dates
                file_info["commits_dates_for_current_version"].append(date)
                file_info["commits_dates_for_current_and_previous_versions"].append(date)

    # Remove not found file_info keys
    files_keys_found_in_version = set(files_keys_found_in_version)
    files_info_keys = list(files_info.keys())
    for file_key in files_info_keys:
        if file_key not in files_keys_found_in_version:
            del files_info[file_key]

    calculate_authors_expertise(files_info, authors_tracking)

    calculate_average_commit_time(files_info)

    print("New file metrics extracted successfully")
    return files_info
