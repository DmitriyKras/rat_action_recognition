# declare classes
declare -a cls=("scratching_back_paw" "body_cleaning" "grooming" "on_back_paws")
declare path="/home/cv-worker/dmitrii/RAT_DATASET/LAB_RAT_ACTIONS_DATASET"

for cl in "${cls[@]}"
do
    for vid in $(ls "$path/$cl/videos/"*sick*)
    do
        mv "$vid" "$path/$cl/videos_sick/"
    done

    for vid in $(ls "$path/$cl/optical_flow/"*sick*)
    do
        mv "$vid" "$path/$cl/optical_flow_sick/"
    done

    for vid in $(ls "$path/$cl/labels_topviewrodents/"*sick*)
    do
        mv "$vid" "$path/$cl/labels_topviewrodents_sick/"
    done

    for vid in $(ls "$path/$cl/labels_ratpose/"*sick*)
    do
        mv "$vid" "$path/$cl/labels_ratpose_sick/"
    done
done