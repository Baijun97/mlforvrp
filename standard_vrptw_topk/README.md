# Standard VRPTW: XGBRanker Demo and Top-K Experiments

This directory contains the 41-feature standard VRPTW implementation,
a trained XGBRanker, and 24 benchmark instances with reference solutions.

It is separate from the 32-feature PCVRPTW implementation at the repository root.
The bundled model must be used with the 41-feature implementation here.

## Included files

- models/xgboost_ranker.json: trained 41-feature XGBRanker
- demo_data/instances/: 12 validation and 12 test instances, each with a reference solution
- input_metadata/: feature order, instance splits and model SHA-256
- evaluate_xgb_vrptw.py: original greedy evaluation
- evaluate_topk_sampling.py: greedy and Top-K sampling evaluation
- generate_training_dataset.py: shared parsing and feature-generation implementation
- results/: recorded validation and test experiments
- EXPERIMENT_RESULTS.md: methodology, results and limitations

The full training dataset and original model-training pipeline are not
included in this directory. The bundled files support inference and route
evaluation, not complete model retraining.

## Installation

Run these commands from the repository root.
Python 3.12 was used for the demonstrated environment.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r standard_vrptw_topk/requirements-demo.txt
```

The requirements file records the demonstrated environment. Package
availability may differ across operating systems.

## Run a one-instance greedy demo

```bash
python standard_vrptw_topk/evaluate_topk_sampling.py \
  --model standard_vrptw_topk/models/xgboost_ranker.json \
  --pipeline-script standard_vrptw_topk/generate_training_dataset.py \
  --prepared-dir standard_vrptw_topk/input_metadata \
  --instances-dir standard_vrptw_topk/demo_data/instances \
  --output-dir standard_vrptw_topk/results/demo_greedy \
  --split valid \
  --max-customers 200 \
  --max-instances 1 \
  --candidate-mode topk \
  --top-k 25 \
  --rounding dimacs \
  --selection-strategy greedy \
  --selection-k 1
```

Use a new output directory for each run. The evaluator rejects non-empty
output directories.

Expected C101 result: feasible, 11 routes, approximately 7.627% distance
gap relative to the reference. Runtime varies by machine.

The evaluator uses reference solutions for comparison. The candidate
selection policy does not use reference routes to choose actions.

## Run the complete evaluation sets

In the command above:

- Use --max-instances 0 to evaluate every eligible instance.
- Use --split valid for the 12 validation instances.
- Use --split test for the 12 test instances.
- Choose a new --output-dir for each run.

The split manifest also lists training and larger instances that are not
bundled. The supplied demo data supports valid/test with max-customers 200.

## Run Top-K sampling

Replace the selection arguments in the demo command with:

```bash
--selection-strategy topk_softmax \
--selection-k 3 \
--temperature 1.0 \
--random-seed 0
```

Use a different output directory. For the recorded best-of-10 procedure,
run seeds 0 through 9 and select the feasible solution with minimum
generated distance separately for each instance. Break distance ties by
fewer routes, then smaller seed. Count the time for all ten attempts.

The greedy solution is not included in the recorded best-of-10 pool.

--top-k 25 controls candidate preselection before model scoring.
--selection-k 3 controls sampling among the highest-scoring feasible
actions after scoring. These are separate parameters.

Each seed initialises a random stream for the whole evaluation batch.
Preserve the instance set and evaluation order to reproduce trajectories.

## Outputs and interpretation

Outputs include instance_results.csv, summary.json, routes.csv,
generated solution files, selection_config.json and selection_trace.csv.

On the recorded test set, greedy achieved a mean distance gap of 28.292885%.
K=3 best-of-10 achieved 32.611059%, with approximately 10.15 times the
construction budget. All recorded test solves were feasible.

Top-K sampling improved some instances but did not improve average
test distance quality. See EXPERIMENT_RESULTS.md for the full analysis.

## Data provenance

See demo_data/SOURCE.md and any accompanying licence files.
Benchmark instances and reference solutions are third-party data.
Generated solutions under results/ are experimental outputs.
