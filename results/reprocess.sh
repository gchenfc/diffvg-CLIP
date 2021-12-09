
# for f in $(ls a-small-*/iter_0499.svg); do
for f in a-small-girl-in-the-grass-plays_*/iter_0499.svg; do
  dir=$(dirname "$f")
  fname=$(basename "$f" .svg)
  fnew="${dir}/${fname}_traj.svg"
  cp $f $fnew
  sed -i 's/stroke-width="[0-9\.]*"/stroke-width="0.4"/g' $fnew
  sed -i 's/stroke-opacity="[0-9\.]*"/stroke-opacity="1"/g' $fnew
  convert $fnew "${dir}/${fname}_traj.png"

  fpng="${dir}/${fname}_traj.png"
  strokes=$(sed "s/.*num_paths=//g;s/).*//g" ${dir}/args.txt)
  name=$(echo $dir | sed "s/_2021.*//g")
  # copy-paste this output into local terminal.
  # Note: may need to create base folder in local machine first
  echo "scp michelangelo:$(pwd)/$fnew ~/Downloads/$name/${strokes}strokes_traj.svg"
  echo "scp michelangelo:$(pwd)/$fpng ~/Downloads/$name/${strokes}strokes_traj.png"
  echo "scp michelangelo:$(pwd)/$f ~/Downloads/$name/${strokes}strokes.svg"
  echo "scp michelangelo:$(pwd)/$dir/${fname}.png ~/Downloads/$name/${strokes}strokes.png"
done

