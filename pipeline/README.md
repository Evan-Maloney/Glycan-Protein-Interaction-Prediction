## Install Dependencies
```bash
pip install -r requirements.txt
```

## Running an Experiment
1. Configure the experiment in `configs/default_config.yaml` or create a new config file in `configs/experiments/`.
2. Run the experiment using the following command:
```bash
python python -m src.run_experiment [--config_path configs/experiments/my_experiment.yaml]
```

# Authenticating Hugging Face
```bash
export HF_TOKEN="your_token_here"
```

Note: Make sure you run pipeline commands from within the pipeline directory.