Run the cleanup checks from the repository root after installing the runtime
requirements and `pytest`, `mypy`, and `ruff`:

```text
python -m pytest tests -q
python -m mypy
python -m ruff check main_code/hog_analysis_script.py main_code/utils_other.py tests --select E,F,W,I --ignore E501,E741
python -m ruff format main_code/hog_analysis_script.py main_code/utils_other.py tests --line-length 100 --check
```

`mypy.ini` uses gradual typing: existing untyped functions are allowed and their
bodies are deferred. Coverage currently includes the two cleaned modules; imported
modules are followed silently.

`analysis_expected.json` records statistics and histograms from commit `ed2a148`
for the seeded image and deterministic HOG descriptor in the regression tests.
Scharr and Sobel use their real filters. The HOG pipeline tests supply a descriptor
so they run with stock scikit-image while installation and unnormalized HOG
support are handled separately. These tests cover the pipeline, not HOG extraction.
