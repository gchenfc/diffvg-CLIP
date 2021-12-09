
for img in 2 3 4 5 6 7 8; do
  for i in 64 256 512 1024; do
    python painterly_rendering.py imgs/res_${img}.png --num_paths $i --max_width 50.0 --num_iter 150
  done
done

grep "N" results/*/args.txt | tee "results/args_all.txt"
