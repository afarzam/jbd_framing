# Makefile for jailbreak detection project

# Configuration
VENV_NAME?=venv
PYTHON=${VENV_NAME}/bin/python3
PIP=${VENV_NAME}/bin/pip3

# Default target executed when no arguments are given to make.
default: setup

# Set up python interpreter environment
.PHONY: setup
setup: 
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV_NAME)
	@echo "Installing dependencies..."
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

# Run the main jailbreak experiment
.PHONY: run_jailbreak
run_jailbreak: 
ifdef config_path
	@echo "Running jailbreak experiment with custom configuration..."
	$(PYTHON) main_jailbreak.py $(config_path)
else
	@echo "Please provide a config path. Usage: make run_jailbreak config_path=<path_to_config>"
endif

# Run the detector training
.PHONY: run_detector
run_detector: 
ifdef config_path
	@echo "Running detector training with custom configuration..."
	$(PYTHON) main_detector.py $(config_path)
else
	@echo "Please provide a config path. Usage: make run_detector config_path=<path_to_config>"
endif

# Run the model alignment
.PHONY: run_align
run_align: 
ifdef config_path
	@echo "Running model alignment with custom configuration..."
	$(PYTHON) main_align.py $(config_path)
else
	@echo "Please provide a config path. Usage: make run_align config_path=<path_to_config>"
endif

# Run the evaluation and plotting
.PHONY: run_eval
run_eval: 
ifdef config_path
	@echo "Running evaluation and plotting with custom configuration..."
	$(PYTHON) main_plot.py $(config_path)
else
	@echo "Please provide a config path. Usage: make run_eval config_path=<path_to_config>"
endif

# Run the processing and analysis
.PHONY: run_process
run_process: 
ifdef config_path
	@echo "Running processing and analysis with custom configuration..."
	$(PYTHON) main_processing.py $(config_path)
else
	@echo "Please provide a config path. Usage: make run_process config_path=<path_to_config>"
endif

# Run multiple experiments with different configs
.PHONY: multirun
multirun: 
ifdef config_paths
	$(foreach config,$(config_paths), \
		@echo "Running experiment with configuration $(config)..."; \
		$(PYTHON) main_jailbreak.py $(config); \
	)
else
	@echo "Please provide config paths. Usage: make multirun config_paths=\"<path1> <path2> ...\""
endif

# Clean up the project
.PHONY: clean
clean:
	@echo "Cleaning up..."
	rm -rf $(VENV_NAME)
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	find . -type f -name '*~' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf output/*
	rm -rf logs/*

# Help
.PHONY: help
help:
	@echo "Makefile commands:"
	@echo "setup         - Set up the virtual environment and install dependencies"
	@echo "run_jailbreak - Run the main jailbreak experiment"
	@echo "run_detector  - Run the detector training"
	@echo "run_align     - Run the model alignment"
	@echo "run_eval      - Run evaluation and plotting"
	@echo "run_process   - Run processing and analysis"
	@echo "multirun      - Run multiple experiments with different configurations"
	@echo "clean         - Clean up the project (remove virtual environment and temporary files)"
	@echo "help          - Display this help message" 