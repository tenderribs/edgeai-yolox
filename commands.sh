# train on synthetic data
python3 -m yolox.tools.train -n yolox_s_tree_pose -b 32 --fp16 --occupy --workers 12 \
    -c './pretrained_models/yolox-s-ti-lite_39p1_57p9_checkpoint.pth' --dataset synth43k

# train on real data
python3 -m yolox.tools.train -n yolox_s_tree_pose -b 32 --fp16 --occupy --workers 12 \
    -c './pretrained_models/2024-03-18_synth43_best.pth' --dataset cana100

# export onnx file
python3 tools/export_onnx.py \
    --output-name pretrained_models/yolox_s_tree_pose.onnx \
    -f exps/default/yolox_s_tree_pose.py \
    -c YOLOX_outputs/yolox_s_tree_pose/best_ckpt.pth

# run inference
python3 kpts_onnx_inference.py \
    --model pretrained_models/yolox_s_cana100_tree_pose.onnx \
    --output_dir YOLOX_outputs/inference \
    --images_path ./datasets/mark_forest \
    --score_thr 0.8