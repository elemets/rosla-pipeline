# Define the default target
SHELL := /bin/bash

.PHONY: all
all: run_pipeline

# Virtual environment directory
VENV_DIR := pipeline_venv

# Target to create virtual environment
$(VENV_DIR)/bin/activate:
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip

# Target to install requirements
.PHONY: requirements
requirements: $(VENV_DIR)/bin/activate
	$(VENV_DIR)/bin/pip install -r requirements.txt


# Target to run the pipeline script
.PHONY: run_pipeline
run_pipeline:
	python pipeline.py

# Target to build the all-deaths dataset (same collation as the pipeline, but
# keeping every death rather than only the overdoses). Pass extra flags with
# ARGS, e.g. `make all_deaths ARGS="--force"`.
.PHONY: all_deaths
all_deaths:
	python all_deaths.py $(ARGS)

# Optional: Clean up generated files
.PHONY: clean
clean:
	rm -rf ./pipeline_steps/input_files/converted/*
	rm -rf ./pipeline_steps/input_files/classified/*
	rm -rf ./pipeline_steps/input_files/classified_all/*
	rm -f ./pipeline_steps/input_files/combined_classified_data.csv
	rm -f ./pipeline_steps/input_files/combined_classified_data_geocoded.csv
	rm -f ./pipeline_steps/input_files/final.csv
	rm -rf $(VENV_DIR)
