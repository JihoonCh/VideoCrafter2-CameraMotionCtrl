#!/bin/bash
# Usage: bash scripts/run_camera_motion.sh <pan_left|pan_right|tilt_up|tilt_down> [opt|base] [opt_scale]
#   opt  : proposed latent optimization (default)
#   base : VideoCrafter2 baseline with the same prompts / seed
MOTION=${1:-pan_left}
METHOD=${2:-opt}
OPT_SCALE=${3:-40}

ckpt='checkpoints/base_512_v2/model.ckpt'
config='configs/inference_t2v_512_v2.0.yaml'
prompt_file="prompts/${MOTION}.txt"
res_dir="results"

# Latent shift per adjacent frame (latent pixels). Content moves opposite to the camera.
case $MOTION in
    pan_left)  SHIFT_W=1;  SHIFT_H=0 ;;
    pan_right) SHIFT_W=-1; SHIFT_H=0 ;;
    tilt_up)   SHIFT_W=0;  SHIFT_H=-1 ;;
    tilt_down) SHIFT_W=0;  SHIFT_H=1 ;;
    *) echo "Unknown motion: $MOTION"; exit 1 ;;
esac

COMMON_ARGS="--seed 123 --mode base --ckpt_path $ckpt --config $config \
--n_samples 1 --bs 1 --height 320 --width 512 \
--unconditional_guidance_scale 12.0 --ddim_steps 50 --ddim_eta 1.0 \
--prompt_file $prompt_file --fps 28"

if [ "$METHOD" = "base" ]; then
    python3 scripts/evaluation/inference.py $COMMON_ARGS \
        --savedir $res_dir/base_512_v2_${MOTION}
else
    python3 scripts/evaluation/inference_latent_opt.py $COMMON_ARGS \
        --savedir $res_dir/${MOTION}_opt_scale_${OPT_SCALE} \
        --opt_scale $OPT_SCALE --shift_w $SHIFT_W --shift_h $SHIFT_H \
        --opt_steps 20 --opt_iters 3
fi
