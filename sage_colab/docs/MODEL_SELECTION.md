# Model decision — researched 6 October 2026

The default is **Qwen/Qwen3-8B**, loaded from revision `b968826d9c46dd6066d109eabc6255188de91218`, using bitsandbytes NF4 double quantisation, a 4,096-token context and bounded output. The fallback is **Qwen/Qwen3-4B**, revision `1cfa9a7208912126459214e8b04321603b3df60c`. Both publisher licence files were fetched and inspected; copies are included here. The publisher identifies both as Apache-2.0. No gated account or paid inference endpoint is used.

The 8B choice requires at least 12 GiB **free** GPU memory and 7 GiB available host RAM before loading. Otherwise the 4B choice requires at least 6 GiB free GPU and 4 GiB available host RAM. The loader requires 22 GiB free runtime disk, keeps context bounded, streams safetensor shards with low-CPU-memory loading, and forbids CPU weight offloading. A CUDA OOM on automatic 8B selection triggers one 4B retry. These thresholds are engineering estimates; the notebook measures actual loading, allocation and generation. The 8B model's roughly 4.1 GB raw 4-bit parameter storage is only a lower bound: unquantised layers, metadata, activations, caches and runtime buffers also consume memory.

Small learning models run on CPU. The retrieval encoder uses CPU. QLoRA tunes only q/v projections in the final four backbone layers, rank 4, sequence length at most 384, at most 16 optimiser steps and a 360-second training budget checked between steps. At least 3 GiB additional free GPU memory is required before this experiment starts; resource failure is explicitly recorded. It saves a candidate rather than deploying unproven adaptation.

Qwen3.5-9B was also considered: its publisher lists Apache-2.0 and multimodal capability. The earlier Qwen3 family was selected to keep the inference/training stack on an established Transformers 4.57.1 integration, reduce architecture-specific kernels, and reserve memory for the other components. This is a compatibility choice, not a claim that Qwen3-8B is the strongest current small model. Actual comparative Colab performance remains to be measured.

The optional **sentence-transformers/all-MiniLM-L6-v2** retrieval model has 22.7M parameters and a publisher-declared Apache-2.0 licence. Startup resolves and records an immutable revision, verifies the licence metadata, and rebuilds vectors when the encoder version changes. Its 384-dimensional embeddings do not themselves establish causal or multimodal correspondence.

Sources:

- [Qwen3-8B publisher model card](https://huggingface.co/Qwen/Qwen3-8B) and [licence](https://huggingface.co/Qwen/Qwen3-8B/blob/main/LICENSE).
- [Qwen3-4B publisher model card](https://huggingface.co/Qwen/Qwen3-4B) and [licence](https://huggingface.co/Qwen/Qwen3-4B/blob/main/LICENSE).
- [Qwen3.5-9B publisher model card](https://huggingface.co/Qwen/Qwen3.5-9B).
- [Transformers 4.57.1 bitsandbytes quantisation documentation](https://huggingface.co/docs/transformers/v4.57.1/quantization/bitsandbytes).
- [PEFT quantisation and adapter training](https://huggingface.co/docs/peft/v0.17.0/en/developer_guides/quantization).
- [MiniLM publisher model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html): free GPU resources and lifetimes are variable and not guaranteed; the interface therefore stays in the notebook rather than running a tunnelled content-generation web application.

Pinned Python library versions are in `requirements-colab.txt`. Colab's installed CUDA-enabled PyTorch is retained and its exact version recorded at runtime. The package pins were checked in a separate local CPU environment; hosted CUDA compatibility is checked automatically in the notebook and is not claimed as locally verified.
