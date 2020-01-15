To use these demos, you will need to install [Jupyter](https://jupyter.org/install) and download `requirements.txt`. Then, run the following commands.

    jupyter nbextension enable --py widgetsnbextension --sys-prefix
    jupyter nbextension install --py --symlink tensorflow_model_analysis --sys-prefix
    jupyter nbextension enable --py tensorflow_model_analysis --sys-prefix
    pip install requirements.txt

Afterwards, you can download any of the `.ipynb` files in this directory and run them via `jupyter notebook`.
