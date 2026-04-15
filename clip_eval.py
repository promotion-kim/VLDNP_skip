#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def infer_prompt_column(df: pd.DataFrame) -> str:
    candidates = ["adv_prompt", "prompt", "sensitive prompt"]
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find prompt column. Expected one of {candidates}, "
        f"but got columns: {list(df.columns)}"
    )


def parse_image_index(path: Path) -> Optional[int]:
    stem = path.stem  # e.g. "15"
    try:
        return int(stem)
    except ValueError:
        return None


@torch.no_grad()
def get_clip_score(
    model: CLIPModel,
    processor: CLIPProcessor,
    image: Image.Image,
    text: str,
    device: torch.device,
) -> Dict[str, float]:
    image = image.convert("RGB")

    # Feature-based cosine similarity
    image_inputs = processor(images=image, return_tensors="pt").to(device)
    text_inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)

    image_features = model.get_image_features(**image_inputs)
    text_features = model.get_text_features(**text_inputs)

    image_features = F.normalize(image_features, dim=-1)
    text_features = F.normalize(text_features, dim=-1)

    cosine = (image_features * text_features).sum(dim=-1).item()

    # CLIP logits_per_image style score
    joint_inputs = processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True,
    ).to(device)
    outputs = model(**joint_inputs)
    logits_per_image = outputs.logits_per_image.squeeze().item()

    return {
        "cosine_similarity": float(cosine),
        "logits_per_image": float(logits_per_image),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CLIP score for generated images.")
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing generated PNG images.")
    parser.add_argument("--csv_path", type=str, required=True, help="CSV file containing prompts.")
    parser.add_argument(
        "--prompt_column",
        type=str,
        default=None,
        help="Prompt column name. If omitted, infer from adv_prompt / prompt / sensitive prompt.",
    )
    parser.add_argument(
        "--clip_model_id",
        type=str,
        default="openai/clip-vit-base-patch32",
        help="CLIP model id from Hugging Face.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Path to save JSON results. Default: <image_dir>/clip_eval.json",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".png",
        help="Image suffix to evaluate. Default: .png",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    csv_path = Path(args.csv_path)
    output_json = Path(args.output_json) if args.output_json else image_dir / "clip_eval.json"

    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")
    if not csv_path.exists():
        raise FileNotFoundError(f"csv_path does not exist: {csv_path}")

    df = pd.read_csv(csv_path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    prompt_column = args.prompt_column or infer_prompt_column(df)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"Loading CLIP model: {args.clip_model_id} on {device}")
    processor = CLIPProcessor.from_pretrained(args.clip_model_id)
    model = CLIPModel.from_pretrained(
        args.clip_model_id,
        use_safetensors=True,
    ).to(device).eval()

    image_paths = sorted(image_dir.glob(f"*{args.suffix}"))
    if not image_paths:
        raise RuntimeError(f"No images with suffix {args.suffix} found in {image_dir}")

    per_image_results: List[Dict] = []
    missing_indices: List[int] = []

    cosine_scores: List[float] = []
    logits_scores: List[float] = []

    for img_path in image_paths:
        idx = parse_image_index(img_path)
        if idx is None:
            print(f"[skip] Could not parse index from filename: {img_path.name}")
            continue

        if idx < 0 or idx >= len(df):
            print(f"[skip] Index {idx} out of range for CSV length {len(df)}")
            missing_indices.append(idx)
            continue

        prompt = str(df.iloc[idx][prompt_column])
        image = Image.open(img_path)

        score_dict = get_clip_score(model, processor, image, prompt, device)

        cosine_scores.append(score_dict["cosine_similarity"])
        logits_scores.append(score_dict["logits_per_image"])

        row = {
            "image_index": int(idx),
            "image_path": str(img_path),
            "prompt": prompt,
            "cosine_similarity": score_dict["cosine_similarity"],
            "logits_per_image": score_dict["logits_per_image"],
        }
        per_image_results.append(row)

        print(
            f"[{img_path.name}] "
            f"cosine={score_dict['cosine_similarity']:.4f}, "
            f"logits={score_dict['logits_per_image']:.4f}"
        )

    summary = {
        "image_dir": str(image_dir),
        "csv_path": str(csv_path),
        "prompt_column": prompt_column,
        "clip_model_id": args.clip_model_id,
        "num_images_scored": len(per_image_results),
        "mean_cosine_similarity": float(sum(cosine_scores) / len(cosine_scores)) if cosine_scores else None,
        "mean_logits_per_image": float(sum(logits_scores) / len(logits_scores)) if logits_scores else None,
        "missing_indices": missing_indices,
        "per_image_results": per_image_results,
    }

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print()
    print(f"Saved CLIP evaluation to: {output_json}")
    print(f"num_images_scored: {summary['num_images_scored']}")
    print(f"mean_cosine_similarity: {summary['mean_cosine_similarity']}")
    print(f"mean_logits_per_image: {summary['mean_logits_per_image']}")


if __name__ == "__main__":
    main()