def update_repository(repo_path, branch_name, commit_message, project_name, queue_name, model_id, repo_url, deploy_key_path):
    from clearml import Task, Model

    task = Task.init(project_name=project_name, task_name="Update Model Weights in GitHub Repository")
    # task.execute_remotely(queue_name=queue_name, exit_process=True)

    import os
    from git import Repo, GitCommandError

    def get_model(model_id):
        from clearml import InputModel

        input_model = InputModel(model_id=model_id, project=project_name, only_published=True)
        input_model.connect(task=task)
        local_model = input_model.get_local_copy()

        local_model.save("model.h5")

    def configure_ssh_key(deploy_key_path):
        os.environ["GIT_SSH_COMMAND"] = f"ssh -i {deploy_key_path} -o IdentitiesOnly=yes"

    def clone_repo(repo_url, branch, deploy_key_path):
        configure_ssh_key(deploy_key_path)
        repo_path = repo_url.split("/")[-1].split(".git")[0]
        try:
            repo = Repo.clone_from(repo_url, repo_path, branch=branch, single_branch=True)
            print(repo_path)
            return repo, repo_path
        except GitCommandError as e:
            print(f"Failed to clone repository: {e}")
            exit(1)

    def ensure_archive_dir(repo):
        archive_path = os.path.join(repo.working_tree_dir, "weights", "archive")
        os.makedirs(archive_path, exist_ok=True)

    def archive_existing_model(repo):
        import datetime

        weights_path = os.path.join(repo.working_tree_dir, "weights")
        model_file = os.path.join(weights_path, "model.h5")
        if os.path.exists(model_file):
            today = datetime.date.today().strftime("%Y%m%d")
            archived_model_file = os.path.join(weights_path, "archive", f"model-{today}.h5")
            os.rename(model_file, archived_model_file)
            return archived_model_file

    def update_model_file(repo, model_path):
        import shutil

        weights_path = os.path.join(repo.working_tree_dir, "weights")
        ensure_archive_dir(repo)
        archived_model_file = archive_existing_model(repo)
        target_model_path = os.path.join(weights_path, "model.h5")
        shutil.move(model_path, target_model_path)
        repo.index.add([archived_model_file])
        repo.index.add([target_model_path])

    def commit_and_push(repo, branch):
        import datetime

        commit_message = f"Update model: {datetime.datetime.now().strftime('%Y%m%d')}"
        try:
            repo.index.commit(commit_message)
            repo.create_tag(commit_message, message="Model update")
            repo.git.push("origin", branch)
            repo.git.push("origin", "--tags")
        except GitCommandError as e:
            print(f"Failed to commit and push changes: {e}")
            exit(1)

    def cleanup_repo(repo_path):
        import shutil

        shutil.rmtree(repo_path, ignore_errors=True)

    repo, repo_path = clone_repo(repo_url, branch_name, deploy_key_path)

    try:
        # Fetch the trained model
        model = Model(model_id=model_id)
        model_path = model.get_local_copy()

        # Update model file and push changes
        update_model_file(repo, model_path)
        commit_and_push(repo, branch_name)
    finally:
        cleanup_repo(repo_path)

    print(f"Pushed changes to remote repository on branch: {branch_name}")


if __name__ == "__main__":
    import argparse

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Automate the process of committing and pushing changes.")

    # Add arguments
    parser.add_argument("--repo_url", required=True, help="Repository URL")
    parser.add_argument("--repo_path", required=False, default="", help="Path to the local Git repository")
    parser.add_argument("--branch", required=False, default="CROP-28-AUTOMATED", help="The branch to commit and push changes to")
    parser.add_argument("--commit_message", required=False, default="Automated commit of model changes", help="Commit message")
    parser.add_argument("--project_name", required=False, default="CropSpot", help="Name of the ClearML project")
    parser.add_argument("--queue_name", type=str, required=False, default="helldiver", help="ClearML queue name")
    parser.add_argument("--deploy_key_path", required=True, help="Path to the SSH deploy key")
    parser.add_argument("--model_id", type=str, required=True, help="The best model ID from ClearML")

    args = parser.parse_args()

    update_repository(
        repo_path=args.repo_path,
        branch_name=args.branch,
        commit_message=args.commit_message,
        project_name=args.project_name,
        model_name=args.model_name,
        queue_name=args.queue_name,
        model_id=args.model_id,
        repo_url=args.repo_url,
        deploy_key_path=args.deploy_key_path,
    )
