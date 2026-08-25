# Classification only: BERT + NFLIS regex layer.
#
#   make classify TEXT="ACUTE FENTANYL TOXICITY"
#   make classify INPUT=records.csv OUTPUT=classified.csv
#
# MODEL is a checkpoint directory name under models/ (or a full path).

PYTHON ?= python
MODEL  ?= bert_models/bioclinicalbert
INPUT  ?=
OUTPUT ?=
TEXT   ?=

.PHONY: classify install

install:
	$(PYTHON) -m pip install -r requirements.txt

classify:
ifneq ($(strip $(TEXT)),)
	$(PYTHON) classify_only.py --model $(MODEL) $(if $(strip $(OUTPUT)),--output $(OUTPUT),) "$(TEXT)"
else ifneq ($(strip $(INPUT)),)
	$(PYTHON) classify_only.py --model $(MODEL) --input $(INPUT) $(if $(strip $(OUTPUT)),--output $(OUTPUT),)
else
	@echo 'Usage: make classify TEXT="..."  |  make classify INPUT=in.csv OUTPUT=out.csv'
	@exit 1
endif
