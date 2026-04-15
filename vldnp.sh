# ===== fixed settings =====
OBJ="ring-a-bell-16"
ROOT_DIR="results_vldnp"
GUIDANCES=("7.5" "15.0" "20.0")

# optional: activate conda env
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate vldnp

mkdir -p "${ROOT_DIR}"

for NG in "${GUIDANCES[@]}"; do
  echo "============================================================"
  echo "Running vldnp (baseline) with neg_guidance=${NG}"
  echo "============================================================"

  python dpm_with_VLM.py \
    --path "${ROOT_DIR}" \
    --vlm_step 5 6 7 9 12 16 21 27 34 42 \
    --obj "${OBJ}" \
    --neg_guidance "${NG}" \
    --vlm_model_id Qwen/Qwen3-VL-4B-Instruct

  OUT_DIR="${ROOT_DIR}/${OBJ}_negative_guidance_${NG}_dpm_neg_prompt_sampling"

  echo "Saved images/results to: ${OUT_DIR}"
done

echo "All runs completed."