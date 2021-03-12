# Check if FFMPEG is installed
FFMPEG=ffmpeg
command -v $FFMPEG >/dev/null 2>&1 || {
	echo >&2 "This script requires ffmpeg. Aborting."; exit 1;
}

# Check number of arguments
if [ "$#" -ne 3 ]; then
	echo "Usage: bash generateframesfromvideos.sh <path_to_directory_containing_videos> <path_to_directory_to_store_frames> <frames_format>"
	exit 1
fi


# Parse videos and generate frames in a directory
# 20 fps, 20 seconds, 400 frames per video 
for video in "$1"/*
do
	videoname=$(basename "${video}")
	videoname="${videoname%.*}"
	videoname=${videoname//[%]/x}
	# fps=$(ffmpeg -i "${video}" 2>&1 | sed -n "s/.*, \(.*\) fp.*/\1/p")
	# echo $fps
	fps=20
	#echo videoname:
	#echo $videoname
	mkdir -p $2
	$FFMPEG -i "${video}" -r ${fps} $2/"${videoname}"_frame_%07d.$3
done