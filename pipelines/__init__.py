# 注：部分 pipeline 依赖完整环境（elapid, rasterio 等），导入失败时不阻塞其他模块加载
try:
    from .train import training_pipeline, valid_pipeline
except ImportError:
    pass

try:
    from .experiment import experiment_pipeline
except ImportError:
    pass

try:
    from .ablation import ablation_pipeline, build_experiment_matrix
except ImportError:
    pass

try:
    from .run_runner import pipeline as run_runner_pipeline
except ImportError:
    pass

try:
    from .results_analysis import (
        comparison_pipeline,
        model_comparison_pipeline,
        sample_visualization_pipeline,
        single_experiment_stat_pipeline,
    )
except ImportError:
    pass
