from pipelines.run_runner import pipeline
from pipelines.train import training_pipeline,valid_pipeline
from pipelines.statistic import get_params

def run_statistic_pipeline():
    config_path = "configs\\statisticConfig.yaml"
    get_params(config_path)

def run_runner_pipeline():
    config_path = "configs\\optimizerConfig.yaml"
    is_processed = False
    processed_timestamp = "20251230_212323"
    idxs = [760,1903,2207,479,596,634]
    pipeline(config_path, idxs, is_processed, processed_timestamp)
    
def run_train_pipeline():
    config_path = "configs\\trainConfig.yaml"
    training_pipeline(config_path)

def run_valid_pipeline():
    config_path = "configs\\trainConfig.yaml"
    model_checkpoint_path = "checkpoints\\attunet\\20260212_045750\\best_model.pth"
    # model_checkpoint_path = "checkpoints\\unet\\20260212_155606\\best_model.pth"
    # model_checkpoint_path = "checkpoints\\segformer\\20260213_062110\\best_model.pth"
    valid_pipeline(config_path, model_checkpoint_path)


if __name__ == "__main__":
    pass
    # run_train_pipeline()
    # run_valid_pipeline()
    # run_runner_pipeline()
