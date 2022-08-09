ckpt_paths=/checkpoint/foivos/unsharded_checkpoints/2022-05-27

# for dir in $ckpt_paths/*; do
  # sbatch incoder_single.sh $dir
# done

## sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m5.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0008.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
## sleep 60
sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m20.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0008.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
## sleep 60
## sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m2.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0008.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
## sleep 60
## sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m10.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0008.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
## sleep 60
## sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m1.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0008.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
## sleep 60
# sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m5.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0006.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
# sleep 60
# sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m5.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0004.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
# sleep 60
# sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m20.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0006.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
# sleep 60
# sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m20.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0004.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
# sleep 60
# sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m2.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0006.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
# sleep 60
# sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m2.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0004.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
# sleep 60
# sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m10.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0006.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
# sleep 60
# sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m1.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0006.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
# sleep 60
# sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m1.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0004.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128
# sleep 60
# sbatch incoder_single.sh /checkpoint/foivos/unsharded_checkpoints/2022-05-27/incoder1B_m10.fp16.transformer_lm_gpt.adam.beta0.9_0.98.wd0.01.clip1.0.lr0.0004.warmup1500.sampletok2048.breakeos_blocked.bs8.updatefreq1.seed3.ngpu128

