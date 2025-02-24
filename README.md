# SemiF-NonTargetWeedClassifier
The folder structure and codebase is structured according to hydra config usage.
The different parts of can be run using `python main.py` command and modifying the configs in `conf` folder.

> To run the codebase on a server, X11 forwarding will be required for interactive parts of `label_images` and `curate_labels` <br> Example command: `ssh -X <username>@<server>`

## Task description

* label_images: gets a sample of images to view and manually label from batches within a specific date range and location
* predict: auto-labels a sample of images while making sure to not try relabeling labeled images
* curate_labels: manually look at a sample of labeled images (generated by `predict.py`) to verify them/change them if needed
* move_files: copies files from longterm storage locations, and split it into train and validation datasets. Now the data is ready for training
* train: function to train the model (uses a pretrained YOLO model as the starting point)
* concat_labels: combines all the labeled image csvs (can have many since the interactive methods let you exit at any time) into one. This is used to catalog the data along with the model

### Config description
`conf/config.yaml` is the default config used. Logs are written to `hydra_logs` folder. 
`conf/train` folder can have multiple configurations for training. `conf/train/default.yaml` is the default configuration used.
