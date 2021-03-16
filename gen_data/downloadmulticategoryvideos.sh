# Check number of arguments
if [ "$#" != "2" ]; then
	echo "Usage: bash downloadmulticategoryvideos.sh <number-of-videos-per-category> <selected-category-file-name>"
	exit 1
fi

while read line
	do
		# could be denied access to some vids, some vids removed 
		bash downloadcategoryids.sh $1 "${line}"
		bash downloadvideos.sh $1 "${line}"
	done < $2

trim=trim
for filename in ./videos/*
do
	filen="${filename%.*}"
	ffmpeg -ss 00:00:00 -t 00:00:20 -i "$filename" -async 1 -vcodec copy -acodec copy "$filen$trim".mp4
	rm "$filename"
done