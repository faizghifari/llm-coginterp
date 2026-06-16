# TODO

## Data cleanup
- [ ] Deduplicate models with multiple model_id/model_name entries (same model, different aliases)
- [ ] Remove experimental/inproper model entries (e.g., names describing training steps rather than model identity)
- [ ] Run verify_data.py after any data changes to ensure FK integrity

## Data expansion
- [ ] Process pending benchmark list from collaborators — deduplicate against existing benchmarks + skipped list, then add as extraction tasks
- [ ] HELM sweep data (312 benchmarks, 1155 models, 11208 results) extracted locally but not yet committed — needs review and commit

## Scripts (needs refactor before committing to repo)
- [ ] Refactor scripts/ directory — consolidate duplicate checkers (check_dupes.py, check_dupes2.py), clean up one-off scripts
- [ ] Add docstrings and CLI help to all utility scripts
- [ ] benchmark_analysis.md — refactor analysis output into proper report format

## Documentation
- [ ] Expand README with usage examples and data schema reference
- [ ] Add CONTRIBUTING.md if opening to collaborators
