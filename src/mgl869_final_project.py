from pathlib import Path
import csv
import os
import json
from modules.jira_extractor import extract_issues
from modules.commits_info import get_commits, extract_new_metrics

file_path = Path('modules/ReleaseVersion_Commit.json')


def load_release_versions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def write_commits_to_csv(commit_list, output_file):
    # Check if the file exists
    if os.path.exists(output_file):
        # Delete the file if it exists
        os.remove(output_file)
        print(f"Deleted existing file: {output_file}")

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['key', 'priority', 'filename'])

        for commit in commit_list:
            key = commit['key']
            priority = commit['priority']
            for bug_file in commit['files']:
                writer.writerow([key, priority, bug_file])


def write_new_metrics_to_csv(new_metrics, output_file):
    # Check if the file exists
    if os.path.exists(output_file):
        # Delete the file if it exists
        os.remove(output_file)
        print(f"Deleted existing file: {output_file}")

    with open(output_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Name',
            'NumberOfLinesAdded',
            'NumberOfLinesRemoved',
            'ChangedByNCommitsInCurrentVersion',
            'ChangedByNCommitsInCurrentVersionAndPreviousVersions',
            'BugsFixedInNCommits',
            'AuthorsInCurrentVersion',
            'AuthorsInCurrentAndPreviousVersions',
            'AuthorsAvgExpertise',
            'AuthorsMinExpertise',
            'AvgCommitTimeForCurrentVersion',
            'AvgCommitTimeForCurrentVersionAndPreviousVersions',
            'ChangedComments',
            'UnchangedComments',
        ])
        
        for file_name, metrics in new_metrics.items():
            number_of_lines_added = metrics["number_of_lines_added"]
            number_of_lines_removed = metrics["number_of_lines_removed"]
            changed_by_n_commits_in_current_version = metrics["changed_by_n_commits_in_current_version"]
            changed_by_n_commits_in_current_and_previous_versions =  metrics["changed_by_n_commits_in_current_and_previous_versions"]
            bugs_fixed_in_n_commits = metrics["bugs_fixed_in_n_commits"]
            authors_in_current_version = len(metrics["authors_in_current_version"])
            authors_in_current_and_previous_versions = len(metrics["authors_in_current_and_previous_versions"])
            authors_average_expertise = metrics["authors_average_expertise"]
            authors_minimal_expertise = metrics["authors_minimal_expertise"]
            average_commit_time_for_current_version = metrics["average_commit_time_for_current_version"]
            average_commit_time_for_current_and_previous_versions = metrics["average_commit_time_for_current_and_previous_versions"]
            changed_comments = metrics["changed_comments"]
            unchanged_comments = metrics["unchanged_comments"]
            writer.writerow([
                file_name,
                number_of_lines_added,
                number_of_lines_removed,
                changed_by_n_commits_in_current_version,
                changed_by_n_commits_in_current_and_previous_versions,
                bugs_fixed_in_n_commits,
                authors_in_current_version,
                authors_in_current_and_previous_versions,
                authors_average_expertise,
                authors_minimal_expertise,
                average_commit_time_for_current_version,
                average_commit_time_for_current_and_previous_versions,
                changed_comments,
                unchanged_comments,
            ])


release_versions = load_release_versions(file_path)

for version in release_versions:
    if version == "1.2.0":
        continue
    JIRA_SEARCH_FILTER = "project = HIVE AND issuetype = Bug AND status in (Resolved, Closed) AND resolution = Fixed AND priority in (Blocker, Critical, Major, Minor, Trivial) AND affectedVersion = " + version
    NEW_METRICS_DIR = Path(os.path.realpath(__file__)).parent.parent / "data" / "new_metrics"
    PRIORITIES_FILE = NEW_METRICS_DIR / f"Priorities_{version}.csv"  # Extraire les commits dans ce fichier
    NEW_METRICS_FILE = NEW_METRICS_DIR / f"New_files_metrics_{version}.csv"

    print("Extracting jira issues...")
    issues = {}
    for issue in extract_issues(JIRA_SEARCH_FILTER, ["versions", "priority"]):
        issues[issue["key"]] = {
            "affectedVersions": [e["name"] for e in issue["fields"]["versions"]],
            "priority": issue["fields"]["priority"]["name"],
        }
    print(f"\tFound {len(issues)} issues.\n")

    # Note: Hive doit être sur la main branch pour que le code bonne les bons résultats
    print("Extracting github commits...")
    repo_path = Path('/home/augusto/Downloads/hive')
    print(repo_path)
    commits_list = get_commits(repo_path, issues, release_versions[version]["commit"])

    write_commits_to_csv(commits_list, PRIORITIES_FILE)

    # Note: Hive doit être sur la main branch pour que le code bonne les bons résultats
    print("Extracting new metrics from commits...")
    repo_path = Path('/home/augusto/Downloads/hive')
    print(repo_path)

    previous_version = release_versions[version]["previous_version"]
    new_metrics = extract_new_metrics(repo_path, release_versions[version]["commit"], release_versions[previous_version]["commit"])

    version_underscore = "_".join(version.split("."))

    write_new_metrics_to_csv(new_metrics, NEW_METRICS_FILE)
