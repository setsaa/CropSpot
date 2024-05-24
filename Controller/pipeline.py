def create_CropSpot_pipeline(
    pipeline_name,
    project_name,
    dataset_name,
    queue_name,
    model_name_1,
    model_name_2,
    model_name_3,
    test_dataset,
    repo_path,
    branch,
    commit_message,
    model_name,
):
    """
    Create a ClearML pipeline for the CropSpot project.

    Parameters:
        pipeline_name (str): Name of the pipeline.
        project_name (str): Name of the ClearML project.
        dataset_name (str): Name of the dataset.
        queue_id (str): Name of the queue to execute the pipeline.

    Returns:
        None
    """

    from clearml import PipelineController, Task
    from upload_data import upload_dataset, download_dataset
    from preprocess_data import preprocess_dataset
    from resnet_train import resnet_train
    from densenet_train import densenet_train
    from vgg_train import vgg_train
    from model_evaluation import evaluate_model
    from compare_models import compare_models
    from update_model import update_repository

    packages = ["pandas", "numpy", "matplotlib", "seaborn", "tensorflow", "keras", "keras-tuner", "tqdm", "clearml", "scikit-learn"]

    # Initialize a new pipeline controller task
    pipeline = PipelineController(name=pipeline_name, project=project_name, add_pipeline_tags=True, target_project=project_name, auto_version_bump=True)

    # Add pipeline-level parameters with defaults from function arguments
    pipeline.add_parameter(name="project_name", default=project_name)
    pipeline.add_parameter(name="dataset_name", default=dataset_name)
    pipeline.add_parameter(name="queue_name", default=queue_name)
    pipeline.add_parameter(name="model_name_1", default=model_name_1)
    pipeline.add_parameter(name="model_name_2", default=model_name_2)
    pipeline.add_parameter(name="model_name_3", default=model_name_3)
    pipeline.add_parameter(name="test_dataset", default=test_dataset)
    pipeline.add_parameter(name="repo_path", default=repo_path)
    pipeline.add_parameter(name="branch", default=branch)
    pipeline.add_parameter(name="commit_message", default=commit_message)
    pipeline.add_parameter(name="model_name", default=model_name)

    # Set the default execution queue
    pipeline.set_default_execution_queue(queue_name)

    # Step 1: Upload Data
    pipeline.add_function_step(
        name="Data_Upload",
        task_name="Upload Raw Data",
        function=upload_dataset,
        function_kwargs=dict(
            dataset_name="${pipeline.dataset_name}",
            project_name="${pipeline.project_name}",
        ),
        task_type=Task.TaskTypes.data_processing,
        function_return=["raw_dataset_id", "raw_dataset_name"],
        helper_functions=[download_dataset],
        parents=None,
        project_name=project_name,
        cache_executed_step=False,
        packages=packages,
    )

    # Step 2: Preprocess Data
    pipeline.add_function_step(
        name="Data_Preprocessing",
        task_name="Preprocess Uploaded Data",
        function=preprocess_dataset,
        function_kwargs=dict(
            dataset_name="${Data_Upload.raw_dataset_name}",
            project_name="${pipeline.project_name}",
        ),
        task_type=Task.TaskTypes.data_processing,
        function_return=["processed_dataset_id", "processed_dataset_name"],
        parents=["Data_Upload"],
        project_name=project_name,
        cache_executed_step=False,
        packages=packages,
    )

    # Step 3(a): Train Model(s)
    pipeline.add_function_step(
        name="ResNet_Model_Training",
        task_name="ResNet Train Model",
        function=resnet_train,
        function_kwargs=dict(
            dataset_name="${Data_Preprocessing.processed_dataset_name}",
            project_name="${pipeline.project_name}",
        ),
        task_type=Task.TaskTypes.training,
        function_return=["resnet_model_id"],
        parents=["Data_Preprocessing"],
        project_name=project_name,
        cache_executed_step=False,
        packages=packages,
    )

    # Step 3(b): Train Model(s)
    pipeline.add_function_step(
        name="DenseNet_Model_Training",
        task_name="DenseNet Train Model",
        function=densenet_train,
        function_kwargs=dict(
            dataset_name="${Data_Preprocessing.processed_dataset_name}",
            project_name="${pipeline.project_name}",
        ),
        task_type=Task.TaskTypes.training,
        function_return=["densenet_model_id"],
        parents=["Data_Preprocessing"],
        project_name=project_name,
        cache_executed_step=False,
        packages=packages,
    )

    # Step 3(c): Train Model(s)
    pipeline.add_function_step(
        name="VGG_Model_Training",
        task_name="VGG Train Model",
        function=vgg_train,
        function_kwargs=dict(
            dataset_name="${Data_Preprocessing.processed_dataset_name}",
            project_name="${pipeline.project_name}",
        ),
        task_type=Task.TaskTypes.training,
        function_return=["VGG_model_id"],
        parents=["Data_Preprocessing"],
        project_name=project_name,
        cache_executed_step=False,
        packages=packages,
    )

    # Step 4(a): Evaluate Model(s)
    pipeline.add_function_step(
        name="ResNet_Model_Evaluation",
        task_name="ResNet Evaluate Model",
        function=evaluate_model,
        function_kwargs=dict(
            model_name="${pipeline.model_name_1}",
            test_dataset="${pipeline.test_dataset}",
            project_name="${pipeline.project_name}",
            task_name="ResNet Evaluate Model",
        ),
        task_type=Task.TaskTypes.testing,
        function_return=["test_accuracy"],
        parents=["ResNet_Model_Training"],
        project_name=project_name,
        cache_executed_step=False,
        packages=packages,
    )

    # Step 4(b): Evaluate Model(s)
    pipeline.add_function_step(
        name="DenseNet_Model_Evaluation",
        task_name="DenseNet Evaluate Model",
        function=evaluate_model,
        function_kwargs=dict(
            model_name="${pipeline.model_name_2}",
            test_dataset="${pipeline.test_dataset}",
            project_name="${pipeline.project_name}",
            task_name="DenseNet Evaluate Model",
        ),
        task_type=Task.TaskTypes.testing,
        function_return=["test_accuracy"],
        parents=["DenseNet_Model_Training"],
        project_name=project_name,
        cache_executed_step=False,
        packages=packages,
    )

    # Step 4(c): Evaluate Model(s)
    pipeline.add_function_step(
        name="VGG_Model_Evaluation",
        task_name="VGG Evaluate Model",
        function=evaluate_model,
        function_kwargs=dict(
            model_name="${pipeline.model_name_3}",
            test_dataset="${pipeline.test_dataset}",
            project_name="${pipeline.project_name}",
            task_name="VGG Evaluate Model",
        ),
        task_type=Task.TaskTypes.testing,
        function_return=["test_accuracy"],
        parents=["VGG_Model_Training"],
        project_name=project_name,
        cache_executed_step=False,
        packages=packages,
    )

    # Step 5: Compare Model(s)
    pipeline.add_function_step(
        name="Model_Comparison",
        task_name="Compare Models",
        function=compare_models,
        function_kwargs=dict(
            model_name_1="${pipeline.model_name_1}",
            model_score_1="${ResNet_Model_Evaluation.test_accuracy}",
            model_name_2="${pipeline.model_name_2}",
            model_score_2="${DenseNet_Model_Evaluation.test_accuracy}",
            model_name_3="${pipeline.model_name_3}",
            model_score_3="${VGG_Model_Evaluation.test_accuracy}",
            project_name="${pipeline.project_name}",
        ),
        task_type=Task.TaskTypes.service,
        function_return=["best_model_id"],
        parents=["ResNet_Model_Evaluation", "DenseNet_Model_Evaluation", "VGG_Model_Evaluation"],
        project_name=project_name,
        cache_executed_step=False,
        packages=packages,
    )

    # # Step 5: Update Model in GitHub Repository
    # pipeline.add_function_step(
    #     name="GitHub_Update",
    #     task_name="Update Model Weights in GitHub Repository",
    #     function=update_repository,
    #     function_kwargs={
    #         "repo_path": "${pipeline.repo_path}",
    #         "branch_name": "${pipeline.branch}",
    #         "commit_message": "${pipeline.commit_message}",
    #         "project_name": "${pipeline.project_name}",
    #         "model_name": "${pipeline.model_name}",
    #         "queue_name": "${pipeline.queue_name}",
    #     },
    #     task_type=Task.TaskTypes.service,
    #     parents=["Model_Training", "Model_Evaluation"],
    #     project_name=project_name,
    #     cache_executed_step=False,
    # )

    # Start the pipeline
    print("CropSpot Data Pipeline initiated. Check ClearML for progress.")
    pipeline.start(queue=queue_name)
    # pipeline.start_locally(run_pipeline_steps_locally=True)
