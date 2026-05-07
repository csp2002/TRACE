cd "$(dirname "$0")/.."
for d in kits lits local pancreas; do
  CUDA_VISIBLE_DEVICES=7 python MedSAM/MedSAM_box_neighbor.py --data "$d"
done
