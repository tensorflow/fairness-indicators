The demos listed here are designed to be used with [Google Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb), a free cloud-based environment for Jupyter notebooks. They can also be run in a local Jupyter environment.

## Google Colaboratory

To run these demos on the cloud, go to `File` -> `Open notebook` in the Colaboratory toolbar, then click on `Github` and paste in the demo's URL. Alternatively, you can use the [Open in Colab](https://chrome.google.com/webstore/detail/open-in-colab/iogfkhleblhcpcekbiedikdehleodpjo?hl=en) Chrome extension to open a notebook directly from GitHub.

## Local Jupyter Environment

To run these demos on your local machine, you will need to install [Jupyter](https://jupyter.org/install). Then, run the following commands.

    jupyter nbextension enable --py widgetsnbextension --sys-prefix
    jupyter nbextension install --py --symlink tensorflow_model_analysis --sys-prefix
    jupyter nbextension enable --py tensorflow_model_analysis --sys-prefix

Afterwards, you can download any of the `.ipynb` files in this directory and run them via `jupyter notebook`.
