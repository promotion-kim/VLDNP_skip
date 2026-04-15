# Adaptive Negative-Prompt Bench (Qwen 3-VL)

This repo contains a **quick-benchmark script** that auto-generates concise
*negative prompts* for diffusion images using
[Qwen-3-VL-Instruct](https://huggingface.co/Qwen).  
It walks a directory of *timestep* sub-folders, feeds each image (optionally
with a few-shot context) to the model, and writes the suggested negative
prompt to CSV.

```

.
├── eval\_negative\_prompt.py   ← main script
├── demos/                    ← 3 sample few-shot images
├── coco-100.csv              ← example meta-csv
├── i2p\_violence.csv          ← another meta-csv
├── ring-a-bell-77.csv        ← another meta-csv
└── requirements.txt

````

> **Note**  
> Large diffusion runs live in `gen_ai_project/…` which is ignored by Git.

---

## 1  Set-up

```bash
# create / activate your venv or conda env first
pip install -r requirements.txt
````

The pinned versions avoid the `numpy 2.0` / `transformers` padding bug that
throws **“expected np.ndarray (got numpy.ndarray)”**.

---

## 2  Directory layout

```
gen_ai_project/
└── vanilla_dpm_ori_pred/
    └── or/
        ├── coco-100/
        │   ├── timestep_0/
        │   ├── timestep_1/
        │   └── … timestep_49/
        ├── i2p-violence/
        │   └── timestep_* / …
        └── ring-a-bell-77/
            └── timestep_* / …
```

*Each* `timestep_x/` folder contains PNG files named
`timestep_<x>_<index>.png`.

---

## 3  Running the benchmark

```bash
# generate prompts for **five** images at timesteps 0,10,20,30,40
python eval_negative_prompt.py \
    --data-root gen_ai_project/vanilla_dpm_ori_pred/or \
    --csv coco-100.csv \
    --timesteps 0 10 20 30 40 \
    --limit 5
```

* `--csv`     points at the metadata file (one row per **clean** source image).
* `--timesteps` selects which diffusion steps to evaluate.
* `--limit`   hard-stops after *N* distinct source images
  → **`5 images × 5 timesteps = 25` generations**.

The script writes:

```
neg_prompts.csv
┌──────────────────────────────┬──────────┬─────────────────────────────────────────┐
| img                          | timestep | neg_prompt                              |
├──────────────────────────────┼──────────┼─────────────────────────────────────────┤
| …/timestep_0/timestep_0_0.png|    0     | motion blur, oversaturation, watermark… |
└──────────────────────────────┴──────────┴─────────────────────────────────────────┘
```

## 4 Running Diffusion Model with VLM

```bash
# VLM evaluating during Diffusion sampling.
bash vldnp.sh
bash vldnp_skip.sh
```

Current Negative prompt format: "Nudity, sexual" + "Breast, nipple, genitalia, buttoks, belly, armpits." (detected body parts)

## 5 Running Diffusion Model with Negative Prompting

```bash
# negative prompting evaluation using various negative guidance sclae
bash negative_prompt.sh
```

## 6 Evaluation with Nudenet Classifier (from SAFREE)
You can download Classifier model as follows:
```bash
curl -L \
  -o classifier/nudenet_classifier_model.onnx \
  https://huggingface.co/gqfwqgw/NudeNet_classifier_model/resolve/main/classifier_model.onnx
```
Download and place at 'classifier/'

Evaluation can be done by

```bash
# Nudity evaluation using Nudenet Classifier
python nudity_eval.py \
	--dir ./results/dir
```
* `--dir`	directory of images to be evaluated.
After evaluation, it will output Attack Success Rate and Toxic Rate.
