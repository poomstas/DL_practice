# images
wget -c http://images.cocodataset.org/zips/train2017.zip   # train dataset
wget -c http://images.cocodataset.org/zips/val2017.zip     # validation dataset
wget -c http://images.cocodataset.org/zips/test2017.zip    # test dataset
wget -c http://images.cocodataset.org/zips/unlabeled2017.zip


# annotations
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget -c http://images.cocodataset.org/annotations/image_info_test2017.zip
wget -c http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip

# Unzip
unzip -q train2017.zip
unzip -q val2017.zip
unzip -q test2017.zip
unzip -q unlabeled2017.zip
unzip -q annotations_trainval2017.zip
unzip -q stuff_annotations_trainval2017.zip
unzip -q image_info_test2017.zip
unzip -q image_info_unlabeled2017.zip

# Delete zip files
# rm train2017.zip
# rm val2017.zip
# rm test2017.zip
# rm unlabeled2017.zip
# rm annotations_trainval2017.zip
# rm stuff_annotations_trainval2017.zip
# rm image_info_test2017.zip
# rm image_info_unlabeled2017.zip