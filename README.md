# gRPC Packaged Basic Image Classification

## About this Repository

### gRPC and Modzy Container Specification

This is a gRPC + HTTP/2 implementation of the [Open Source Basic Image Classification model](https://github.com/modzy/packaged-basic-image-classification) and is derived from Modzy's [gRPC Python Model Template](https://github.com/modzy/grpc-model-template).

## Installation

Clone the repository:

```git clone https://github.com/modzy/grpc-packaged-basic-image-classification.git```

## Usage

All different methods of testing the gRPC template specification can be found in the [Usage section](https://github.com/modzy/grpc-model-template#Usage) of the gRPC Python Model Template.  

The following usage instructions demonstrate how to build the container image, run the container, open a shell inside the container, and run a test using the `grpc_model.src.model_client` module.

#### Build and run the container

From the parent directory of this repository, build the container image.

```docker build -t grpc-packaged-basic-image-classification .```

Run the container interactively.

```docker run -it grpc-packaged-basic-image-classification:latest```

#### Run a test inside the container

Open a different terminal, use `docker ps` to extract your running container ID, and open up a shell inside the container.

```docker exec -it <container-id> bash```

In the shell, submit a test using poetry (installed inside the container) and the grpc_model.src.model_client module

```poetry run python -m grpc_model.src.model_server``` 

## Additional Resources

### Managing Dependencies Via a Virtual Environment

This project template uses [Poetry](https://python-poetry.org/) in order to manage the Python dependencies that you 
use within the project. If this is your first time using this tool, you can follow the instructions provided
[here](https://python-poetry.org/docs/#installation) to install it.

There are two types of dependencies: core dependencies and development dependencies. Core dependencies are those that
are required to be installed for the main, production release of your project or package. Development dependencies are
auxiliary packages that are useful in aiding in providing functionality such as formatting, documentation or
type-checking, but are non-essential for the production release.

For each dependency you come across, make a determination on whether it is a core or development dependency, and add it
to the pyproject.toml file from the command line using the following command, where the `-D` flag is to be used only for
development dependencies.
```
poetry add [-D] <name-of-dependency>
```

When you are ready to run your code and have added all your dependencies, you can perform a `poetry lock` in order to
reproducibly fix your dependency versions. This will use the pyproject.toml file to crease a poetry.lock file. Then, in
order to run your code, you can use the following commands to set up a virtual environment and then run your code
within the virtual envrionment. The optional `--no-dev` flag indicates that you only wish to install core dependencies.
```
poetry install [--no-dev]
poetry run <your-command>
```

### Initializing Pre-Commit Hooks

This repository uses pre-commit hooks in order to assist you in maintaining a uniform and idiomatic code style.
If this is your first time using pre-commit hooks you can install the framework [here](https://pre-commit.com/#installation).
Once pre-commit is installed, all you need to do is execute the following command from the repository root:
```
pre-commit install
```

If you want to execute the pre-commit hooks at a time other than during the actual git commit, you can run:
```
pre-commit run --all-files
```


### Exporting current dependencies when ready to release

If you are developing within a virtual environment for convenience and reproducibility but would like to run directly
on top of pip inside of your docker container to have a very lightweight image, you can use the following instructions
in order to extract a `requirements.txt` from your virtual environment.

```
poetry export -f requirements.txt --output requirements.txt
```
OR
```
poetry export -f requirements.txt --output requirements.txt --without-hashes

```


### Compiling the protocol buffers (WARNING: only intended for template authors)

```
./scripts/compile_protocol_buffers.sh
```

