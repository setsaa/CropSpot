from pipeline import create_CropSpot_pipeline

def create_CropSpot_pipeline():
    from clearml import PipelineController

    queue_name = 'default'

    pipeline = PipelineController(
        name='simple_pipeline',
        project='CropSpot',
        version="1.0",
        add_pipeline_tags=True,
        target_project='CropSpot',
        auto_version_bump=True
    )

    pipeline.add_parameter(name="queue_name", default=queue_name)
    pipeline.add_parameter(name="model_path_1", default='Trained Models/cropspot_resnet_model.h5')
    pipeline.add_parameter(name="dataset_name", default='TomatoDiseaseDataset')

    pipeline.set_default_execution_queue(queue_name)

create_CropSpot_pipeline()