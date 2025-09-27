#!/bin/bash
name="$1"

final_path="../final_model_$1"

if [ -e "$final_path" ]; then
    final_path="$final_path-$(date +%Y%m%d-%H%M%S)"
fi


# init value
# origin_data_name="daycare.v2i.yolov8"

origin_data_name="test_script"
augment_assets_path="./augment_assets"
num_add_image=2
num_add_effect=1
train_code_line="./train.py --data_dir --data_dir ./"

keep_paths=(
    "$augment_assets_path"
    "jdsjd"
)

# init augment data
cp -r ../datasets/$origin_data_name ./
cp -r $augment_assets_path/* ./$origin_data_name/train/
cp -r $augment_assets_path/* ./$origin_data_name/valid/
mkdir -p $origin_data_name/train/augmented_images
mkdir -p $origin_data_name/train/augmented_labels
mkdir -p $origin_data_name/valid/augmented_images
mkdir -p $origin_data_name/valid/augmented_labels
cp -r $origin_data_name/train/images/* $origin_data_name/train/augmented_images/
cp -r $origin_data_name/train/labels/* $origin_data_name/train/augmented_labels/
cp -r $origin_data_name/valid/images/* $origin_data_name/valid/augmented_images/
cp -r $origin_data_name/valid/labels/* $origin_data_name/valid/augmented_labels/

# add image data in train and valid
for i in $(seq 1 $num_add_image); do
    cd $origin_data_name/train
    python ./add_image.py
    bash ./clone_data.sh new_data/images ai$i
    bash ./clone_data.sh new_data/labels ai$i
    cp -r new_data/images/* augmented_images/
    cp -r new_data/labels/* augmented_labels/
    rm -rf new_data
    cd ../..

    cd $origin_data_name/valid
    python ./add_image.py
    bash ./clone_data.sh new_data/images ai$i
    bash ./clone_data.sh new_data/labels ai$i
    cp -r new_data/images/* augmented_images/
    cp -r new_data/labels/* augmented_labels/
    rm -rf new_data
    cd ../..
done

# add effect data in train and valid
for i in $(seq 1 $num_add_effect); do
    cd $origin_data_name/train
    python ./add_effect.py
    cp -r labels new_labels
    bash ./clone_data.sh new_images ae$i
    bash ./clone_data.sh new_labels ae$i
    cp -r new_images/* augmented_images/
    cp -r new_labels/* augmented_labels/
    rm -rf new_images
    rm -rf new_labels
    cd ../..

    cd $origin_data_name/valid
    python ./add_effect.py
    cp -r labels new_labels
    bash ./clone_data.sh new_images ae$i
    bash ./clone_data.sh new_labels ae$i
    cp -r new_images/* augmented_images/
    cp -r new_labels/* augmented_labels/
    rm -rf new_images
    rm -rf new_labels
    cd ../..
done

mv $origin_data_name/train/augmented_images/* $origin_data_name/train/images
mv $origin_data_name/train/augmented_labels/* $origin_data_name/train/labels
mv $origin_data_name/valid/augmented_images/* $origin_data_name/valid/images
mv $origin_data_name/valid/augmented_labels/* $origin_data_name/valid/labels
# remove empty folders
rm -rf $origin_data_name/train/augmented_images
rm -rf $origin_data_name/train/augmented_labels
rm -rf $origin_data_name/valid/augmented_images
rm -rf $origin_data_name/valid/augmented_labels

python ./augment_assets/convert_label.py --work_dir ./$origin_data_name --target_class 0

# train and valid data
source ./venv/bin/activate
python $train_code_line


# Store trained model
for path in "${keep_paths[@]}"; do
    if [ -e "$path" ]; then
        cp -r "$path" "$final_path/"
        echo "Stored: $path"
    else
        echo "Not found: $path" >&2
    fi
done
# clean train-dir
# rm -rf $origin_data_name


exit 0