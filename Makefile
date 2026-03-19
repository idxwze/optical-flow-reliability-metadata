PYTHON ?= python3
DATASET_ROOT ?= /Users/seifeddinereguige/Documents/tfds_Dataset
OUTPUT_DIR ?= outputs

.PHONY: build_epe build_raft_epe train_epe train_raft_epe plots

build_epe:
	$(PYTHON) -m src.build_all_tables_epe --dataset_root "$(DATASET_ROOT)" --splits train --output_dir "$(OUTPUT_DIR)"

build_raft_epe:
	$(PYTHON) -m src.build_all_tables_raft_epe --dataset_root "$(DATASET_ROOT)" --splits train --raft_model small --output_dir "$(OUTPUT_DIR)"

train_epe:
	$(PYTHON) -m src.train_regressor --csv "$(OUTPUT_DIR)/table_all_scenarios_epe.csv" --target epe_mean --output_dir "$(OUTPUT_DIR)"

train_raft_epe:
	$(PYTHON) -m src.train_regressor --csv "$(OUTPUT_DIR)/table_all_scenarios_raft_epe.csv" --target epe_mean_raft --output_dir "$(OUTPUT_DIR)"

plots:
	$(PYTHON) -m src.plots --table "$(OUTPUT_DIR)/table_all_scenarios.csv" --preds "$(OUTPUT_DIR)/preds_gradient_boosting_reliability_score.csv" --target reliability_score --out_dir reports/figures
