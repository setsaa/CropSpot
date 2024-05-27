def update_repository(repo_path, branch_name, commit_message, project_name, model_id, repo_url, deploy_key_path):
    import os
    from clearml import Task
    from git import Repo, GitCommandError
    from keras.models import load_model, save_model

    task = Task.init(project_name=project_name, task_name="Update Model Weights in GitHub Repository")

    def get_model(model_id):
        from clearml import InputModel

        input_model = InputModel(model_id=model_id)
        input_model.connect(task=task)
        local_model = input_model.get_local_copy()

        model = load_model(local_model)
        save_model(model, "model.h5")

        return "model.h5"

    def configure_ssh_key(deploy_key_path):
        os.environ["GIT_SSH_COMMAND"] = f"ssh -i {deploy_key_path} -o IdentitiesOnly=yes"

    def clone_repo(repo_url, branch, deploy_key_path):
        configure_ssh_key(deploy_key_path)
        repo_name = repo_url.split("/")[-1].split(".git")[0]
        repo_path = f"{repo_name}_clone"
        if os.path.exists(repo_path):
            import shutil

            shutil.rmtree(repo_path)
        try:
            repo = Repo.clone_from(repo_url, repo_path, branch=branch, single_branch=True)
            print(f"Cloned repository to path: {repo_path}")
            return repo, repo_path
        except GitCommandError as e:
            print(f"Failed to clone repository: {e}")
            exit(1)

    def archive_existing_model(repo):
        import datetime

        model_file = os.path.join(repo.working_tree_dir, "model.h5")
        if os.path.exists(model_file):
            archive_dir = os.path.join(repo.working_tree_dir, "archive")
            os.makedirs(archive_dir, exist_ok=True)
            today = datetime.date.today().strftime("%Y%m%d")
            current_time = datetime.datetime.now().strftime("%H%M%S")
            archived_model_file = os.path.join(archive_dir, f"model-{today}-{current_time}.h5")
            os.rename(model_file, archived_model_file)
            return archived_model_file
        return None

    def update_model_file(repo, model_path):
        import shutil

        archived_model_file = archive_existing_model(repo)
        target_model_path = os.path.join(repo.working_tree_dir, "model.h5")
        shutil.move(model_path, target_model_path)
        if archived_model_file:
            repo.index.add([archived_model_file])
        repo.index.add([target_model_path])

    def commit_and_push(repo, branch, commit_message):
        try:
            repo.index.commit(commit_message)
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
        model_path = get_model(model_id)
        print(f"Model path obtained: {model_path}")

        # Update model file and push changes
        update_model_file(repo, model_path)
        commit_and_push(repo, branch_name, commit_message)
    finally:
        cleanup_repo(repo_path)

    print(f"Pushed changes to remote repository on branch: {branch_name}")
