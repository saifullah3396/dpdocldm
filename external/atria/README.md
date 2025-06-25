## Environment Setup

### Create a virtual environment
```
pip install --user virtualenv virtualenvwrapper
echo export WORKON_HOME=$HOME/.virtualenvs >> ~/.bashrc
echo source ~/.local/bin/virtualenvwrapper.sh >> ~/.bashrc
source ~/.bashrc
```

```
mkvirtualenv atria
workon atria
```

### Install from git
Install the build dependencies:
```
pip install setuptools>=60 setuptools-scm>=8.0 wheel pip==24.3.1 torch==2.3.0 torchinfo==1.8.0 torchtext==0.18.0
pip install git+https://git.opendfki.de/saifullah/atria.git@1.0.0
```
### Install from source
Install the dependencies:
```
pip install -r requirements.txt
```

Build atria hydra configurations:
```
python -m atria._hydra.build_configurations
```

Setup environment variables:
```
export PYTHONPATH=<path/to/atria>/src
```
