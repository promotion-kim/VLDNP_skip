# ===== fixed settings =====
OBJ="ring-a-bell-16"
ROOT_DIR="results_static_neg"
GUIDANCES=("0.0" "7.5" "15.0" "20.0")

# optional: activate conda env
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate vldnp

mkdir -p "${ROOT_DIR}"

for NG in "${GUIDANCES[@]}"; do
  echo "============================================================"
  echo "Running static negative prompting with neg_guidance=${NG}"
  echo "============================================================"

  python negative_prompt.py \
    --path "${ROOT_DIR}" \
    --obj "${OBJ}" \
    --neg_guidance "${NG}"

  OUT_DIR="${ROOT_DIR}/${OBJ}_negative_guidance_${NG}_dpm_neg_prompt_sampling"

  echo "Saved images/results to: ${OUT_DIR}"
done

echo "All runs completed."