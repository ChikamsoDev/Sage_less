# SAGE for hosted Google Colab

A runnable research agent implementing the supplied blueprint's acting, learning and governed improvement loops. This project does **not** establish AGI, ASI, broad transfer, or superiority to a strong LLM agent.

1. Open **Start_SAGE.ipynb** in [Google Colab](https://colab.research.google.com/), using **File → Upload notebook**.
2. Select **Runtime → Change runtime type → an available GPU**.
3. Choose **Runtime → Run all**, and authorise Google Drive when Colab requests it.
4. Wait for automatic installation, checks, training, downloads and diagnostic tasks. Enter a goal in the final notebook interface and press **Run goal**. Set an output filename when you want a mechanically checked artifact.

No repository clone, separate source upload, API key, paid API, purchased compute, or local large-model download is required. The notebook embeds the source, tests, licence records and documentation with a checksum. Its only external downloads are Python packages and public model files inside hosted Colab. Google controls free GPU availability and runtime lifetimes; these cannot be guaranteed.

Example goal: `Read example_stock.csv. Calculate total quantity and total inventory value. Write a short report to stock_report.txt, preserving the input.` Set **Output file** to `stock_report.txt`. Input files can be added through the notebook widget. Inputs and generated files are limited to UTF-8 text/CSV/JSON/Python, 1 MiB per file. The code tool runs standard-library Python in an OS sandbox and returns text artifacts for explicit publication.

The notebook saves its state to `MyDrive/SAGE_Colab_v1/snapshots/`. A new runtime restores the newest intact snapshot and resumes bounded initial training. Existing goals are listed for owner-authorised resume. Press **Stop / revoke** to cancel generation and stop further dispatch. Owner corrections and releases remain separate from model predictions.

**Training:** the small GRU belief/world ensemble, action-value policy and computation controller train automatically on observed simulator transitions. Their parameters, optimiser, dataset manifests and checkpoints persist. A small QLoRA experiment updates and saves real language-model adapter parameters, but its adapter remains disabled until broader evaluation exists. Personal task experiences enter training only if you select the separate consent checkbox. Later small-model and planner candidates require an explicit owner release and support rollback.

The pretrained backbone is Qwen3-8B in 4-bit NF4 on a sufficiently roomy GPU/host, with Qwen3-4B as the automatic memory fallback. The actual free memory is measured before downloading, the immutable model revision is pinned, and a real generation check runs afterward. Set `MODEL_ID` and optionally `MODEL_REVISION` in the notebook to use another compatible causal model with an accepted licence. Retrieval optionally uses all-MiniLM-L6-v2, falling back explicitly to a versioned hash index.

See [implementation coverage and limitations](docs/IMPLEMENTATION_STATUS.md), [verification results](docs/VALIDATION.md), [model research and licences](docs/MODEL_SELECTION.md), and [authority threat model](docs/SECURITY.md). `SAGE_Source.zip` contains the complete source without large pretrained weights.

For development diagnostics only (no LLM download), the source also supports:

```sh
python -m pip install -e .
python -m pytest -q
python -m sage.cli --state ./research_state collect --output trajectories.jsonl
python -m sage.cli --state ./research_state train-world --steps 120
python -m sage.cli --state ./research_state evaluate
```

Large-model loading is deliberately restricted to hosted Colab. Local checks use small CPU models and explicit scripted fixtures; they do not substitute a fake model for the live backbone.
