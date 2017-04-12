for i in $(find . \(  -name "*.avi" \) | cut -d'/'  -f2 | cut -d'.' -f1)
do
	mkdir ../images/$i
	ffmpeg -i  $i.avi -vf scale=256:256 -r 5/1 ../images/"$i"/"$i"_%d.jpg
done