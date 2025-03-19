# declare classes
#declare -a cls=("scratching_back_paw" "body_cleaning" "grooming" "on_back_paws")  # initial exp
declare -a cls=("eating" "sniffing")  # bridge
#declare -a cls=("scratching_back_paw" "body_cleaning" "grooming" "on_back_paws" "eating" "sniffing")  # target exp
declare path="/home/cv-worker/dmitrii/RAT_DATASET/LAB_RAT_ACTIONS_DATASET"

for cl in "${cls[@]}"
do
    for vid in $(ls "$path/$cl/videos_sick/")
    do
        mv "$path/$cl/videos_sick/$vid" "$path/$cl/videos/$vid"
    done

    for vid in $(ls "$path/$cl/optical_flow_sick/")
    do
        mv "$path/$cl/optical_flow_sick/$vid" "$path/$cl/optical_flow/$vid"
    done

    for vid in $(ls "$path/$cl/labels_topviewrodents_sick/")
    do
        mv "$path/$cl/labels_topviewrodents_sick/$vid" "$path/$cl/labels_topviewrodents/$vid"
    done

    for vid in $(ls "$path/$cl/labels_ratpose_sick/")
    do
        mv "$path/$cl/labels_ratpose_sick/$vid" "$path/$cl/labels_ratpose/$vid"
    done

    for vid in $(ls "$path/$cl/kpts_features_sick/")
    do
        mv "$path/$cl/kpts_features_sick/$vid" "$path/$cl/kpts_features/$vid"
    done
done
