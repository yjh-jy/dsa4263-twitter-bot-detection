# ------------------------------------------------------------
# Cross-platform settings (Bash, PowerShell, cmd)
# ------------------------------------------------------------
# On Unix-like systems, use bash; on Windows, let make use cmd.exe.
# This keeps command bodies simple and compatible across shells.

ifeq ($(OS),Windows_NT)
  SHELL := cmd.exe
  .SHELLFLAGS := /C
  PYTHON := python
else
  SHELL := /bin/bash
  .SHELLFLAGS := -c
  PYTHON := python
endif

DOCKER := docker compose

# ------------------------------------------------------------
# Build images
# ------------------------------------------------------------
build:
	$(DOCKER) build

build-nocache:
	$(DOCKER) build --no-cache

# ------------------------------------------------------------
# Run Jupyter Notebook (JupyterLab) for EDA purposes
# ------------------------------------------------------------
notebook:
	$(DOCKER) up jupyter

# ============================================================
# Preprocessing commands
# ============================================================

# Preprocessing step for traditional, random forest, xgboost
prep:
	$(DOCKER) run --rm runtime \
		$(PYTHON) -m src.preprocessing.preprocess

# Optional (preprocessing step for multimodal model)
prep-multimodal:
	$(DOCKER) run --rm runtime \
		$(PYTHON) -m src.preprocessing.preprocess_multimodal \
		--input data/raw/twitter_human_bots_dataset.csv \
		--output-dir data/cleaned/multimodal

# ============================================================
# Individual model training + evaluation commands
# ============================================================

# ---------- XGBoost ----------
xgb: # evaluate and produce plots
	$(DOCKER) run --rm runtime \
		$(PYTHON) -m src.models.xgboost_model \
		--force-retrain False

xgb-retrain: # only if you want to retrain, takes about 30mins to complete
	$(DOCKER) run --rm runtime \
			$(PYTHON) -m src.models.xgboost_model \
			--force-retrain True

# ---------- Random Forest ----------
rf:
	$(DOCKER) run --rm runtime \
		$(PYTHON) -m src.models.randomforest_model

# ---------- Traditional ML Models ----------
trad:
	$(DOCKER) run --rm runtime \
		$(PYTHON) -m src.models.traditional_ml_model

# Run all the main models sequentially
all-models: xgb rf classic

# ============================================================
# (Optional, training takes very long) Multimodal
# ============================================================
multimodal:
	$(DOCKER) run --rm runtime \
		$(PYTHON) -m src.models.multimodal_model \
		--csv data/cleaned/multimodal/all_splits.csv \
		--epochs 30 \
		--batch-size 32 \
		--out-dir outputs/multimodal

# ------------------------------------------------------------
# Get a bash shell inside the runtime container
# ------------------------------------------------------------
bash:
	$(DOCKER) run --rm -it runtime bash

# ------------------------------------------------------------
# Tear down containers (does not remove volumes)
# ------------------------------------------------------------
down:
	$(DOCKER) down

# ------------------------------------------------------------
# Hard cleanup (remove containers + volumes)
# ------------------------------------------------------------
clean:
	$(DOCKER) down -v

# ------------------------------------------------------------
# Make hygiene
# ------------------------------------------------------------
.PHONY: \
	build build-nocache notebook \
	prep prep-multimodal \
	xgb rf classic all-models \
	multimodal bash down clean