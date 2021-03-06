# NCMInD

The HAI project aims to create an accurate representation of patient movement within the UNC Health Care system. On top of this, disease specific state modules can be appended to this location module, in order to model HAI transmissions and potential interventions. 

### Two Notes:
1. We used RTIs syntehtic population as a base population for this work. 

	Rineer, J. I., Bobashev, G. V., Jones, K. R., Hadley, E. C., Kery, C., et al. (2019) RTI 2017 Synthetic Population Extract. RTI International. Accessed 09/2019.

2. Documentation for the location model (544 location nodes) can be found [here](Stewardship-paper-ODD/S1 - Stewardship ODD.pdf).


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

## Next Steps
View the README.md in `NCMInD/`.


## Disclaimer

This work was supported by grant number U01CK000527. The conclusions, findings, and opinions expressed do not necessarily reflect the official position of the U.S. Centers for Disease Control and Prevention.