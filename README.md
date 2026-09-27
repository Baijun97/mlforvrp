# PCVRPTW XGBoost Routing Pipeline

This is **Program 2** in your project.

## Relationship to Program 1

Program 1:

```text
.vrp + .sol
    ↓
reference-route replay
    ↓
32 features + 0/1 label
    ↓
train.csv / validation.csv / test.csv
    ↓
teammate trains XGBoost
```

Program 2 (this package):

```text
NEW .vrp + trained XGBoost model
            ↓
     initialise vehicle
            ↓
   generate feasible candidates
            ↓
 calculate the SAME 32 features
            ↓
      XGBoost scores
            ↓
 choose highest-score candidate
            ↓
 update location / time / capacity
            ↓
          repeat
            ↓
 complete routes + metrics
```

**The inference pipeline does not read `.sol`.**

---

## Files

```text
pcvrptw_xgboost_pipeline/
├── pcvrptw_core.py
├── model_adapter.py
├── routing_pipeline.py
├── solve_with_xgboost.py
├── calibrate_threshold.py
├── feature_columns.json
├── pipeline_config.json
├── requirements.txt
└── README.md
```

### `pcvrptw_core.py`

Shared problem logic:

- parse `.vrp`
- `floor(Euclidean)` distance
- feasibility filtering
- exactly the same 32 feature definitions used by Program 1
- vehicle-state update
- route verification

### `model_adapter.py`

Loads:

- `XGBClassifier.save_model(...json/.ubj)`
- raw XGBoost Booster models
- `.pkl/.pickle/.joblib` sklearn-style models

Recommended teammate export:

```python
model.save_model("xgboost_model.json")
```

### `routing_pipeline.py`

The full route-construction loop.

### `calibrate_threshold.py`

Recommended because PCVRPTW can intentionally end a route even when feasible
customers remain. Program 1 therefore emitted terminal replay states with all
candidate labels equal to 0.

This utility uses `validation.csv` to learn a score threshold:

- max score >= threshold -> continue to a customer
- max score < threshold -> return to depot / stop opening routes

---

# 1. Install

```bash
pip install -r requirements.txt
```

# 2. Put the trained model here

Example:

```text
models/
└── xgboost_model.json
```

The model MUST have been trained with the exact 32 columns in
`feature_columns.json`, in that order.

# 3. Calibrate stop threshold (recommended)

Use the `validation.csv` created by Program 1:

```bash
python calibrate_threshold.py \
  --validation generated_training_data/validation.csv \
  --model models/xgboost_model.json \
  --feature-columns feature_columns.json \
  --output pipeline_config.json
```

If your teammate did not include terminal-state rows, this calibration mechanism
is not valid. In that case you need a separate route-end/STOP rule.

# 4. Solve a NEW instance

Important: use only the `.vrp` as pipeline input. Do not give the pipeline the
test `.sol`.

```bash
python solve_with_xgboost.py \
  --instance PCVRPTW/C1_10_10.vrp \
  --model models/xgboost_model.json \
  --feature-columns feature_columns.json \
  --config pipeline_config.json \
  --output-dir outputs/C1_10_10
```

For a quick test without calibration:

```bash
python solve_with_xgboost.py \
  --instance PCVRPTW/C1_10_10.vrp \
  --model models/xgboost_model.json \
  --feature-columns feature_columns.json \
  --stop-threshold 0.5 \
  --output-dir outputs/C1_10_10
```

# 5. Pipeline output

```text
outputs/C1_10_10/
├── solution.json
├── routes.txt
└── decision_trace.csv
```

`routes.txt` is human-readable:

```text
Route #1: 473 543 679 ...
Route #2: ...
...
Total distance: ...
Vehicles used: ...
Collected prize: ...
Feasible: True
```

`solution.json` contains:

- every route
- VRP node IDs
- customer IDs
- route load
- route distance
- return time
- selected XGBoost scores
- total distance
- vehicles used
- visited/unvisited clients
- collected/uncollected prize
- runtime
- capacity feasibility
- time-window feasibility
- depot-return feasibility

`decision_trace.csv` contains every action selected by the model, useful for
debugging and explaining the model to the supervisor.

---

# Critical consistency rules

## 1. Same features in training and inference

The model was trained on:

```text
32 features
```

This pipeline calculates those same 32 features in the same order.

If `feature_columns.json` does not exactly match, the program stops with an
error rather than silently giving incorrect predictions.

## 2. Same distance rule

For the uploaded PCVRPTW archive:

```text
distance = floor(Euclidean distance)
travel_time = distance
```

Program 1 and Program 2 use the same rule.

## 3. Hard constraints are NOT delegated to XGBoost

Before XGBoost scores anything, the Python program filters out candidates that:

- exceed remaining capacity
- miss the customer due time
- cannot return to the depot before depot closing time

XGBoost ranks only feasible candidates.

## 4. Model output is not a route

For one state:

```text
candidate A -> score 0.18
candidate B -> score 0.81
candidate C -> score 0.42
```

The pipeline selects B, updates the state, recalculates all candidates, and calls
the model again.

The complete route is produced by repeated model calls.

## 5. Prize-collecting route end

This dataset is PCVRPTW, not ordinary CVRPTW. Some customers may deliberately
remain unvisited. Therefore the pipeline needs a route-end/global-stop rule.

The current design uses:

```text
max candidate score < stop_threshold
```

to end the current route. If this happens before a fresh route visits any
customer, the pipeline stops opening more vehicles.

Tune the threshold on validation data; do not assume 0.5 is optimal.
