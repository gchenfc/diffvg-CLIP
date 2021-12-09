
  for i in 32 64 256 1024; do
    python diffvg_clip.py --num_paths $i --num_iter 500
  done

grep "N" results3/*/args.txt | tee "results3/args_all.txt"
