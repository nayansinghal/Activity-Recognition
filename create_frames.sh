for i in $(find . \( ! -name ".*" \) | cut -d'/'  -f2)
do
    mkdir ../images/$i
    ffmpeg -i  $i -vf scale=256:256 -r 1/1 ../images/"$i"/"$i"_%d.jpg
done
