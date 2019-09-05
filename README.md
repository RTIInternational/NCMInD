# NCMInD

The HAI project aims to create an accurate representation of patient movement within the UNC Health Care system. On top of this, disease specific state modules can be appended to this location module, in order to model HAI transmissions and potential interventions. 


## Virtual Environment Setup

There are several ways to work with project code locally. The easiest is to use a virtual environment. 

If you already have python and pip installed on your computer, you can run the following from the repo directory:

```
pip install virtualenv
```

Create the environment:

```
virtualenv hai_env
```

Activate Environment:

```
source hai_env/bin/activate
```

Install Packages

```
pip install -r docker/requirements.txt
```

## Docker Setup

Eventually, we will likely move to running code (for large simulations) in the cloud. We have setup a docker image for this. 

From the root directory of the project:

`docker-compose build`

### Open Docker Container
`docker-compose run --rm hai bash`

### Open Jupyter notebook using Docker
`docker-compose up -d notebook`

Container is running here: `http://localhost:1111/tree?`


## Disclaimer

This work was supported by grant number U01CK000527. The conclusions, findings, and opinions expressed do not necessarily reflect the official position of the U.S. Centers for Disease Control and Prevention.













