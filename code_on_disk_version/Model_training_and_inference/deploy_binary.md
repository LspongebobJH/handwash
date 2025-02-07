# Preparation
* Environment: conda env `new-ai-handwash`.
* Working directory: `~/handwash/code_on_disk_version/Model_training_and_inference/`, at the git branch `binary` (Please check and make sure you are at this branch). Note that the working directory of binary classification model is different from the original one, which deployed multi-class classification model.
* Pull latest updates from github by `git pull origin binary`.
* If you want to deploy the app at the same time, please be noted to revise backend and frontend port numbers. For instance, to avoid the conflict of backend ports, please revise the port number in  `deploy_backend.py` line 248 `web.run_app(app,port=9500)`.

# Deployment
* Execute `python deploy_backend.py`.
  
# Evaluation
* To acquire the evaluation results of step\_i, please replace the step\_i in `weights` in the following yaml script with the actual step.

```yaml
# ./config/st_gcn/handwash/test.yaml

weights: ./work_dir/recognition/handwash/ST_GCN/step_i/epoch1000_model.pt

# feeder
feeder: feeder.feeder.Feeder
test_feeder_args:
  data_path: ./data/handwash/test_data.npy
  label_path: ./data/handwash/test_label.pkl

# model
model: net.st_gcn.Model
model_args:
  in_channels: 3
  num_class: 1
  edge_importance_weighting: True
  graph_args:
    layout: "hand_wash"
    strategy: "spatial"

# test
phase: test
device: 0
test_batch_size: 64
```

After than, executing `python recognition -c config/st_gcn/handwash/test.yaml`, you will see evaluation results at the end of testing.