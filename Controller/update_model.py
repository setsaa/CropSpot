def update_repository(repo_path, branch_name, commit_message, project_name, model_name):
    import os
    from git import Repo
    from clearml import Task

    task = Task.init(
        project_name=project_name,
        task_name="GitHub Repo Update",
        task_type=Task.TaskTypes.service,
    )

    repo = Repo(os.getcwd())

    # Check and switch to the specified branch, create if it doesn't exist
    if branch_name not in repo.heads:
        new_branch = repo.create_head(branch_name)
        new_branch.checkout()
        print(f"Created and switched to new branch: {branch_name}")
    else:
        repo.heads[branch_name].checkout()
        print(f"Switched to existing branch: {branch_name}")

    # Check if there are any changes
    changed_files = [item.a_path for item in repo.index.diff(None)] + repo.untracked_files
    if changed_files:
        print("Changes detected in the following files:", changed_files)

        # Stage all changes
        repo.git.add(all=True)

        # Commit changes
        repo.index.commit(commit_message)
        print("Committed changes.")

        # Push changes to the specified branch on remote
        origin = repo.remote(name="origin")
        origin.push(refspec=f"{branch_name}:{branch_name}")
        print("Pushed changes to remote repository on branch:", branch_name)
    else:
        print("No changes detected.")


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Automate the process of committing and pushing changes.")

    # Add arguments
    parser.add_argument("--repo_path", required=False, default="", help="Path to the local Git repository")
    parser.add_argument("--branch", required=False, default="Crop-33-Deploy-MLOPs-pipeline", help="The branch to commit and push changes to")
    parser.add_argument("--commit_message", required=False, default="Automated commit of model changes", help="Commit message")
    parser.add_argument("--project_name", required=False, default="CropSpot", help="Name of the ClearML project")
    parser.add_argument("--model_name", required=False, default="CropSpot_Model", help="ClearML trained model")

    args = parser.parse_args()

    update_repository(args.repo_path, args.branch)


# from git import GitCommandError, Repo


# # Configure Git to use the SSH deploy key for operations.
# def configure_ssh_key(DEPLOY_KEY_PATH):
#     import os

#     """Configure Git to use the SSH deploy key for operations."""
#     os.environ["GIT_SSH_COMMAND"] = f"ssh -i {DEPLOY_KEY_PATH} -o IdentitiesOnly=yes"


# # Clone the repository.
# def clone_repo(REPO_URL, branch, DEPLOY_KEY_PATH) -> tuple[Repo, str]:
#     from git import GitCommandError, Repo

#     """Clone the repository."""
#     configure_ssh_key(DEPLOY_KEY_PATH)
#     repo_path = REPO_URL.split("/")[-1].split(".git")[0]
#     try:
#         repo: Repo = Repo.clone_from(REPO_URL, repo_path, branch=branch, single_branch=True)
#         print(repo_path)
#         return repo, repo_path
#     except GitCommandError as e:
#         print(f"Failed to clone repository: {e}")
#         exit(1)


# # Ensure the archive directory exists within weights.
# def ensure_archive_dir(repo: Repo):
#     import os

#     """Ensures the archive directory exists within weights."""
#     archive_path = os.path.join(repo.working_tree_dir, "weights", "archive")
#     os.makedirs(archive_path, exist_ok=True)


# # Archive the existing model weights.
# def archive_existing_model(repo: Repo) -> str:
#     import datetime
#     import os

#     """Archives existing model weights."""
#     weights_path = os.path.join(repo.working_tree_dir, "weights")
#     model_file = os.path.join(weights_path, "model.h5")
#     if os.path.exists(model_file):
#         today = datetime.date.today().strftime("%Y%m%d")
#         archived_model_file = os.path.join(weights_path, "archive", f"model-{today}.h5")
#         os.rename(model_file, archived_model_file)
#         return archived_model_file  # Return the path of the archived file


# # Update the model weights in the repository.
# def update_weights(repo: Repo, model_path):
#     import os
#     import shutil

#     """Updates the model weights in the repository."""
#     weights_path = os.path.join(repo.working_tree_dir, "weights")
#     ensure_archive_dir(repo)
#     archived_model_file = archive_existing_model(repo)
#     target_model_path = os.path.join(weights_path, "model.h5")

#     # Use shutil.move for cross-device move
#     shutil.move(model_path, target_model_path)

#     # Add the newly archived model file to the Git index
#     repo.index.add([archived_model_file])

#     # Also add the new model file to the Git index
#     repo.index.add([target_model_path])


# # Commit and push changes to the remote repository.
# def commit_and_push(repo: Repo, model_id, DEVELOPMENT_BRANCH):
#     import datetime
#     from git import GitCommandError

#     """Commits and pushes changes to the remote repository."""
#     commit_message = f"Update model weights: {model_id}"
#     tag_name = f"{model_id}-{datetime.datetime.now().strftime('%Y%m%d')}"
#     try:
#         repo.index.commit(commit_message)
#         repo.create_tag(tag_name, message="Model update")
#         repo.git.push("origin", DEVELOPMENT_BRANCH)
#         repo.git.push("origin", "--tags")
#     except GitCommandError as e:
#         print(f"Failed to commit and push changes: {e}")
#         exit(1)


# # Safely remove the cloned repository directory.
# def cleanup_repo(repo_path):
#     import shutil

#     """Safely remove the cloned repository directory."""
#     shutil.rmtree(repo_path, ignore_errors=True)


# # Update the model weights in the repository.
# def update_model(model_id, env_path, REPO_URL, DEVELOPMENT_BRANCH, project_name):
#     import os
#     from clearml import Model, Task
#     from dotenv import load_dotenv

#     task = Task.init(
#         project_name=project_name,
#         task_name="GitHub Model Update",
#         task_type=Task.TaskTypes.service,
#     )

#     """Fetches the trained model using its ID and updates it in the repository."""

#     load_dotenv(dotenv_path=env_path)
#     DEPLOY_KEY_PATH = os.getenv("DEPLOY_KEY_PATH")

#     # Prepare repository and SSH key
#     repo, repo_path = clone_repo(REPO_URL, DEVELOPMENT_BRANCH, DEPLOY_KEY_PATH)
#     try:
#         # Fetch the trained model
#         model = Model(model_id=model_id)
#         model_path = model.get_local_copy()

#         # Update weights and push changes
#         update_weights(repo, model_path)
#         commit_and_push(repo, model_id, DEVELOPMENT_BRANCH)
#     finally:
#         # Ensure cleanup happens even if an error occurs
#         cleanup_repo(repo_path)


# if __name__ == "__main__":
#     import argparse

#     # Parse command-line arguments
#     parser = argparse.ArgumentParser(description="Update model weights in GitHub repo using a ClearML model ID")

#     # Add arguments
#     parser.add_argument(
#         "--model_id",
#         required=False,
#         help="The ClearML model ID to fetch and update",
#         default="d2af953123b34ce7916de255cd793f92",
#     )
#     parser.add_argument(
#         "--env_path",
#         required=False,
#         help="Path to the .env file",
#         default="/Users/apple/Desktop/AI_Studio/Introduction_to_MLOPS/First_Pipeline/.env",
#     )
#     parser.add_argument(
#         "--repo_url",
#         required=False,
#         help="Repository URL",
#         default="git@github.com:GitarthVaishnav/Cifar10_SimpleFlaskApp.git",
#     )
#     parser.add_argument(
#         "--development_branch",
#         required=False,
#         help="Development branch name",
#         default="development",
#     )
#     parser.add_argument(
#         "--project_name",
#         required=False,
#         help="ClearML Project name",
#         default="CIFAR-10 Project",
#     )

#     args = parser.parse_args()

#     update_model(
#         args.model_id,
#         args.env_path,
#         args.repo_url,
#         args.development_branch,
#         args.project_name,
#     )
