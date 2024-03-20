#!/bin/bash

launch () {
	tempfile=.temp/$$/$1.txt
	success=""

	echo launching run $1 : dumping to $tempfile

	while [ -z "$success" ]
	do
		RUN_ID=$1 xmanager launch one-xm-launch.py > $tempfile
		success=$(grep JobState.JOB_STATE_SUCCEEDED $tempfile)
	done

}


mkdir -p .temp/$$/

for i in $(python -c "from entrypoint_histo import get_run_ids; print(*get_run_ids())")
do
	if [ "$1" == "run" ]
	then
		launch $i &
		sleep 3
	else
		echo "------------------------------------------------------------"
		echo run $i
		python entrypoint_histo.py --run_id=$i
	fi
done

echo "------------------------------------------------------------"

wait
