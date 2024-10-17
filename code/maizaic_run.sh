#!/bin/bash

working_path="/data/e/dmc/data/24r_06_12_601_605_field26_perp_to_row_low"

hm_method="surf"

angle_files=("$working_path"/*.csv)

#split to group based on boundaries
'''
python create_frame_to_mosaic.py \
	-image_path "$working_path/calibrated" \
	-save_path "$working_path/frames_to_mosaic" \
        -scale 3 \
'''
python split_for_mini.py \
     -image_path "$working_path/frames_to_mosaic" \
     -save_path "$working_path/${hm_method}_mini_partition" \
     -hm "$working_path/homography_matrices/H_${hm_method}.csv" \
     -angle_csv "${angle_files[@]}"

#loop to mosaic all all minis
mini_path="$working_path/${hm_method}_mini_partition"

for mini_hm in "$mini_path"/*.csv;
do
    if [ -f "$mini_hm" ]; then
        echo "processing $mini_hm"

        #extract the group number from the file name
        group_number=$(basename "$mini_hm" | grep -oE '[0-9]+')

        python stitcher.py \
            -image_path "$mini_path/group_$group_number" \
            -hm "$mini_hm" \
            -save_path "$working_path/${hm_method}_mini_mosaics" \
            -scale 3 \
            -fname "group$group_number" \
            -mini_mosaic
    else
        echo "there is no .csv file found in $mini_hm"
    fi
done



#assemble global mosaic
python surf_assembly_global.py \
       -image_path "$working_path/${hm_method}_mini_mosaics" \
       -save_path "$working_path/${hm_method}_global_mosaic"
