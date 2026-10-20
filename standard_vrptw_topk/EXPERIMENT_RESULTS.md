# Top-K Sampling Experiment: Standard VRPTW

## Scope

This experiment evaluates score-based Top-K sampling in the original
41-feature standard VRPTW route constructor using the saved XGBRanker model.

It is separate from the repository's 32-feature PCVRPTW pipeline.
The trained model was not modified or retrained. These results concern
downstream route construction, not offline ranking accuracy.

## Method

- Model: xgb_ranker_200/xgboost_ranker.json
- Model objective: rank:pairwise
- Features: 41
- Instances: standard VRPTW benchmarks with at most 200 customers
- Existing instance-level train/validation/test split retained
- Validation instances: 12
- Test instances: 12
- Candidate preselection: original topk mode, K=25
- Distance convention: dimacs
- Hard feasibility filtering retained
- Baseline: greedy highest-score action
- Sampling: softmax over the highest-scoring K feasible actions
- Temperature: 1.0
- Random seeds: 0–9
- Validation selection K values: 3, 5, 10
- Selected test configuration: K=3, temperature=1.0, 10 runs
- No route repair or local search applied

Candidate preselection K=25 and score-based selection K are separate
parameters. Depot-return actions participate in sampling when present
in the feasible candidate set.

Each seed initialises one random-number stream for an entire evaluation
batch. An instance's sampled trajectory therefore also depends on the
preceding instances and their decisions.

Best-of-10 selects the feasible solution with the lowest generated
distance for each instance. Ties are resolved by fewer routes, then
smaller seed. Reference solutions are used for evaluation, not selection.
The greedy solution is not included in the best-of-10 pool.

Reported gaps are arithmetic means of per-instance percentage distance
gaps relative to the reference solutions. They are not proven optimality
gaps. Vehicle count is reported separately; selection is distance-first.

## Validation results

All validation runs were feasible.

| Method | Mean distance gap (%) | Mean vehicles | Mean construction budget per instance (s) |
|---|---:|---:|---:|
| Greedy | 34.1023 | 15.8333 | 0.2295 |
| K=3 single-run mean | 43.2458 | 16.3250 | 0.2314 |
| K=3 best-of-10 | 31.3725 | 14.5833 | 2.3135 |
| K=5 single-run mean | 49.6988 | 16.6250 | 0.2359 |
| K=5 best-of-10 | 35.6789 | 15.0000 | 2.3585 |
| K=10 single-run mean | 54.6016 | 16.9667 | 0.2362 |
| K=10 best-of-10 | 42.3372 | 15.7500 | 2.3616 |

K=3 was selected from the evaluated configurations using validation
results before running the test experiment. Temperature was fixed,
not tuned.

## Test results

All 132 test solves were feasible: 12 greedy solves and 120 sampled solves.

| Method | Feasible rate (%) | Mean distance gap (%) | Mean vehicles | Mean construction budget per instance (s) |
|---|---:|---:|---:|---:|
| Greedy | 100 | 28.292885 | 15.000000 | 0.225653 |
| K=3 single-run mean | 100 | 43.404238 | 15.758333 | 0.229079 |
| K=3 best-of-10 | 100 | 32.611059 | 14.333333 | 2.290786 |

Best-of-10 produced shorter routes by total distance on 4 of 12 test
instances and longer total distances on the other 8.

Its mean distance gap increased by 4.318174 percentage points relative
to greedy. Its mean construction budget was approximately 10.15 times
the greedy budget. Although mean vehicle count decreased, this did not
constitute an improvement in the preselected distance criterion.

Construction time excludes process startup and model loading.
Best-of-10 time includes all ten construction attempts.
These measurements are not complete end-to-end application timings.

## Conclusion

Under the evaluated configuration and sampling budget, Top-K softmax
sampling did not improve average test-set distance quality relative to
greedy construction. Multiple attempts improved some instances, but the
average validation benefit did not reproduce on the test set.

Greedy therefore remains the preferred baseline for distance quality
and construction speed in this experiment. This conclusion is limited
to the evaluated model, instances, temperature, K values and run budget;
it does not establish that all Top-K search methods are ineffective.

Offline XGBRanker ranking accuracy was not changed or evaluated by this
experiment. Feasibility on the evaluated instances does not guarantee
feasibility on every future instance.

## Evidence

Completed validation batch:
results/valid_batch_Cjq8DL/

Completed test batch:
results/test_batch_FioHQR/

Key files within the batches:

- validation_comparison.csv or test_comparison.csv
- best10_per_instance.csv
- all_sampling_runs.csv
- Per-run instance_results.csv, summary.json and selection_config.json
- Per-run routes.csv and selection_trace.csv, where generated
- logs/
- environment.txt
- Saved copies of the evaluation and data-generation scripts

The prepared feature list, instance split manifest and model SHA-256 are
recorded in input_metadata/. The trained model and benchmark instance files
are external inputs and are not bundled. Reproduction requires matching
versions of these inputs.
