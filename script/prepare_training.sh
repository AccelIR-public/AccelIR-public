python ./src/data/generate_patches.py --hr_data_dir ./dataset/DIV2K/DIV2K_train_HR --lr_data_dir ./dataset/DIV2K/DIV2K_train_LR_bicubic/X4 \
--target_hr_data_dir ./dataset/patch/DIV2K_train_HR/X4 --target_lr_data_dir ./dataset/patch/DIV2K_train_LR_bicubic/X4

python ./src/data/generate_patches.py --hr_data_dir ./dataset/DIV2K/DIV2K_valid_HR --lr_data_dir ./dataset/DIV2K/DIV2K_valid_LR_bicubic/X4 \
--target_hr_data_dir ./dataset/patch/DIV2K_valid_HR/X4 --target_lr_data_dir ./dataset/patch/DIV2K_valid_LR_bicubic/X4

python ./src/data/generate_info.py --hr_data_dir ./dataset/patch/DIV2K_train_HR/X4 --lr_data_dir ./dataset/patch/DIV2K_train_LR_bicubic/X4 

python ./src/data/generate_info.py --hr_data_dir ./dataset/patch/DIV2K_valid_HR/X4 --lr_data_dir ./dataset/patch/DIV2K_valid_LR_bicubic/X4 
