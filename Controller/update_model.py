import argparse
import datetime
import os
import shutil

from clearml import Model, Task
from dotenv import load_dotenv
from git import GitCommandError, Repo

def configure_ssh_key(DEPLOY_KEY_PATH):
    """Configure Git to use the SSH deploy key for operations."""
    os.environ["GIT_SSH_COMMAND"] = f"ssh -i {DEPLOY_KEY_PATH} -o IdentitiesOnly=yes"

def clone_repo(REPO_URL, branch, DEPLOY_KEY_PATH) -> tuple[Repo, str]:
    """Clone the repository using the provided SSH key."""
    configure_ssh_key(DEPLOY_KEY_PATH)
    repo_path = REPO_URL.split("/")[-1].split(".git")[0]
    try:
        repo = Repo.clone_from(REPO_URL, repo_path, branch=branch, single_branch=True)
        return repo, repo_path
    except GitCommandError as e:
        print(f"Failed to clone repository: {e}")
        exit(1)

def ensure_archive_dir(repo: Repo):
    archive_path = os.path.join(repo.working_tree_dir, "weights", "archive")
    os.makedirs(archive_path, exist_ok=True)

def archive_existing_model(repo: Repo) -> str:
    """Archive existing model weights in the repository."""
    weights_path = os.path.join(repo.working_tree_dir, "weights")
    model_file = os.path.join(weights_path, "model.h5")
    if os.path.exists(model_file):
        today = datetime.date.today().strftime("%Y%m%d")
        archived_model_file = os.path.join(weights_path, "archive", f"model-{today}.h5")
        os.rename(model_file, archived_model_file)
        return archived_model_file

def update_weights(repo: Repo, model_path):
    # Update the model weights in the repository with the new model file.
    weights_path = os.path.join(repo.working_tree_dir, "weights")
    ensure_archive_dir(repo)
    archived_model_file = archive_existing_model(repo)
    target_model_path = os.path.join(weights_path, "model.h5")
    shutil.move(model_path, target_model_path)
    repo.index.add([archived_model_file, target_model_path])

def commit_and_push(repo: Repo, model_id, DEVELOPMENT_BRANCH):
    # Commit changes and push them to the remote repository.
    commit_message = f"Update model weights: {model_id}"
    tag_name = f"{model_id}-{datetime.datetime.now().strftime('%Y%m%d')}"
    try:
        repo.index.commit(commit_message)
        repo.create_tag(tag_name, message="Model update")
        repo.git.push("origin", DEVELOPMENT_BRANCH)
        repo.git.push("origin", "--tags")
    except GitCommandError as e:
        print(f"Failed to commit and push changes: {e}")
        exit(1)

def cleanup_repo(repo_path):
    # Clean up by removing the cloned repository directory.
    shutil.rmtree(repo_path, ignore_errors=True)

def update_model(model_id, env_path, REPO_URL, DEVELOPMENT_BRANCH, project_name):
    # Update the model weights in the repository using the provided model ID.
    task = Task.init(project_name=project_name, task_name="Update Model", task_type=Task.TaskTypes.custom)
    task.execute_remotely(queue_name="default", exit_process=True)
    load_dotenv(dotenv_path=env_path)
    DEPLOY_KEY_PATH = os.getenv("DEPLOY_KEY_PATH")
    repo, repo_path = clone_repo(REPO_URL, DEVELOPMENT_BRANCH, DEPLOY_KEY_PATH)
    try:
        model = Model(model_id=model_id)
        model_path = model.get_local_copy()
        update_weights(repo, model_path)
        commit_and_push(repo, model_id, DEVELOPMENT_BRANCH)
    finally:
        cleanup_repo(repo_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update model weights in the repository using a ClearML model ID.")
    parser.add_argument("--model_id", required=True, help="The ClearML model ID to fetch and update")
    parser.add_argument("--env_path", required=True, help="Path to the .env file containing environment variables")
    parser.add_argument("--repo_url", required=True, help="Repository URL")
    parser.add_argument("--development_branch", required=True, help="Development branch name")
    parser.add_argument("--project_name", required=True, help="ClearML Project name")
    args = parser.parse_args()
    update_model(args.model_id, args.env_path, args.repo_url, args.development_branch, args.project_name)
