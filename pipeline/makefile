# Makefile

PYTHON = python3
PDF2CSV_SCRIPT = Pdf2csv.py
GEOCODE_SCRIPT = geocode.py
CLASSIFY_SCRIPT = Classify.py
INPUT_FILE ?= input.pdf  # Default value if not provided

# Default target
all: process

# Check if the input is a PDF
process: check_pdf geocode classify

# If the input is a PDF, convert it to CSV
check_pdf:
	@if [ "${INPUT_FILE##*.}" = "pdf" ]; then \
		echo "Input is a PDF, converting to CSV..."; \
		$(PYTHON) $(PDF2CSV_SCRIPT) $(INPUT_FILE); \
		export INPUT_FILE=$$(echo $(INPUT_FILE) | sed 's/.pdf/.csv/'); \
	else \
		echo "Input is not a PDF, skipping PDF to CSV conversion..."; \
	fi

# Run the geocode script
geocode:
	@if [ "${INPUT_FILE##*.}" = "csv" ]; then \
		echo "Running geocode on CSV..."; \
		$(PYTHON) $(GEOCODE_SCRIPT) $(INPUT_FILE); \
	else \
		echo "Input is not a CSV, cannot geocode."; \
		exit 1; \
	fi

# Run the classify script
classify:
	@if [ "${INPUT_FILE##*.}" = "csv" ]; then \
		echo "Running classification..."; \
		$(PYTHON) $(CLASSIFY_SCRIPT) $(INPUT_FILE); \
	else \
		echo "Input is not a CSV, cannot classify."; \
		exit 1; \
	fi

# Clean up (optional)
clean:
	rm -f *_geocoded.csv