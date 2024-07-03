# Setup Environment
Required packages:
- pytorch 
- kappaconfig (`pip install kappaconfig`) to resolve yaml files
- wandb (`pip install wandb`)
- einops (`pip install einops`)
- sklearn (`pip install scikit-learn`)
- torchmetrics (`pip install torchmetrics`)
- pandas (`pip install pandas`)

NOTE: make sure that `torchmetrics` is up to date. sun397 train800 and train800val200 splits only have 396 classes
whereas the test split has 397 classes. Older versions of `torchmetrics` do not support this case.

# Setup configs

You'll need to specify some properties that are dependent on your server environment in a file called `static_config.yaml`.
A template is given as `static_config_example.yaml` which you can simply copy and adjust the values to your setup.

`cp static_config_example.yaml static_config.yaml`

In the `static_config.yaml` you need to specify the following properties:
- `vtab1k: ~/Documents/data/vtab-1k`: path to your vtab-1k dataset
- `local_root: ~/Documents/data_local`:  Optional: path to a fast storage (e.g. a local SSD on a compute node). The datasets will be copied to this path before training.
- `torch_home: ~/Documents/torchhub`: Optional: set the TORCH_HOME environment variable to download checkpoints into this path
