# EVLDNP: Efficient VLDNP via Similarity-Guided VLM Call Reduction

EVLDNP accelerates Vision-Language-guided Dynamic Negative Prompting (VLDNP)
for safer text-to-image diffusion. During sampling, EVLDNP compares the current
intermediate prediction with the image from the most recent VLM query. If their
CLIP image embeddings are sufficiently similar, it reuses the previous negative
prompt instead of calling the VLM again.

The method is training-free and is implemented with Stable Diffusion v1.4,
DPM-Solver++, Qwen3-VL-4B-Instruct, and CLIP ViT-B/32. With a similarity
threshold of 0.95 on COCO-100, EVLDNP reduced the average number of VLM calls
from 8.90 to 5.55 per image (about 38%) while preserving the reported CLIP score
of 0.312.

## Repository layout

```text
.
├── dpm_with_VLM.py        # VLDNP and EVLDNP sampling
├── vldnp_skip.sh          # EVLDNP experiment (similarity gating)
├── vldnp.sh               # VLDNP experiment
├── negative_prompt.py     # static negative-prompt baseline
├── negative_prompt.sh
├── vanilla_dpm.py         # Stable Diffusion baseline
├── eval_negative_prompt.py
├── nudity_eval.py         # NudeNet-based safety evaluation
├── clip_eval.py           # CLIP alignment evaluation
├── prompt_set/            # evaluation prompt sets
├── demos/                 # few-shot VLM examples
└── requirements.txt
```

Large model files and newly generated experiment outputs are ignored. A small
set of sample outputs is retained for reference.

## 1. Set-up

Python 3.10 and a CUDA-capable GPU are recommended. Create and activate a
virtual environment first, then install the dependencies:

```bash
pip install -r requirements.txt
```

The implementation downloads the following pretrained models from Hugging Face
when first run:

- `CompVis/stable-diffusion-v1-4`
- `Qwen/Qwen3-VL-4B-Instruct`
- `openai/clip-vit-base-patch32`

Make sure you have accepted any applicable model terms and authenticated with
Hugging Face if required.

## 2. How EVLDNP works

At each configured VLM query step, the script reconstructs an intermediate
image from the current latent. EVLDNP computes the cosine similarity between
its CLIP embedding and that of the last image sent to the VLM:

```text
similarity >= threshold  -> reuse the previous negative prompt
similarity <  threshold  -> query the VLM and update the negative prompt
```

The main options are:

- `--vlm_step`: candidate diffusion steps for VLM queries
- `--sim_threshold`: similarity threshold for prompt reuse
- `--sim_model_id`: image encoder used by the similarity gate
- `--force_vlm_every`: maximum consecutive prompt reuses before a fresh query
- `--vlm_model_id`: VLM used to detect relevant unsafe visual concepts
- `--neg_guidance`: negative-prompt guidance scale

For each run, generated images and an `inference_stats.json` file are written to
the output directory. The JSON includes per-image inference time, VLM call
count, and prompt-reuse count.

## 3. Running EVLDNP

The provided script reproduces the main EVLDNP configuration with ten candidate
query steps, Qwen3-VL-4B-Instruct, CLIP ViT-B/32, and a similarity threshold of
0.95:

```bash
bash vldnp_skip.sh
```

To run one configuration directly:

```bash
python dpm_with_VLM.py \
  --path results_vldnp_skip \
  --obj ring-a-bell-16 \
  --vlm_step 5 6 7 9 12 16 21 27 34 42 \
  --neg_guidance 15.0 \
  --vlm_model_id Qwen/Qwen3-VL-4B-Instruct \
  --sim_model_id openai/clip-vit-base-patch32 \
  --sim_threshold 0.95 \
  --force_vlm_every 100
```

Supported prompt-set names for `--obj` are `ring-a-bell-16`,
`ring-a-bell-38`, `ring-a-bell-77`, `p4d`, `unlearn-diff`, `coco`, and `i2p`.

## 4. Running comparison methods

```bash
# VLDNP comparison
bash vldnp.sh

# Static negative prompting with multiple guidance scales
bash negative_prompt.sh
```

The static negative-prompt implementation uses a fixed safety prompt and does
not call the VLM.

## 5. Evaluating generated images

### Safety evaluation with NudeNet

Download the classifier model and place it under `classifier/`:

```bash
curl -L \
  -o classifier/nudenet_classifier_model.onnx \
  https://huggingface.co/gqfwqgw/NudeNet_classifier_model/resolve/main/classifier_model.onnx
```

Then evaluate a generated-image directory:

```bash
python nudity_eval.py --dir ./results/dir
```

The evaluator reports Attack Success Rate (ASR) and Toxic Rate (TR); lower is
better for both metrics.

### CLIP alignment evaluation

```bash
python clip_eval.py \
  --image_dir ./results/dir \
  --csv_path ./prompt_set/coco-100.csv
```

`--image_dir` is the generated-image directory, and `--csv_path` is the prompt
CSV used for generation.

## 6. Reported results

The following COCO-100 results are reported for Stable Diffusion v1.4 with 50
sampling steps. Higher CLIP score is better; lower inference time and fewer VLM
calls are better.

| Method | Negative guidance | CLIP score | Avg. inference time (s) | Avg. VLM calls |
|---|---:|---:|---:|---:|
| Stable Diffusion 1.4 | - | 0.312 | 7.05 | - |
| VLDNP | 15.0 | 0.312 | 15.40 | 8.90 |
| **EVLDNP** | **15.0** | **0.312** | **12.80** | **5.55** |
| Static negative prompting | 15.0 | 0.296 | 10.47 | - |

Exact runtime depends on the GPU, software environment, model cache, and system
load.

## 7. Funding acknowledgment

This work was supported by the Institute of Information & Communications
Technology Planning & Evaluation (IITP) grant funded by the Korea government
(MSIT) (No. RS-2024-00343989, Research on data characteristics for social and
ethical learning and enhancement of generative AI model ethics).

## 8. License

This project is licensed under the [Apache License 2.0](LICENSE).
