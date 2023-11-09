import shutil
import os

# Set the paths of the original folder and the new folder
original_folder = "/home/zubairirshad/generalizable-object-representations/ckpts/vanilla_SF1_3view/3viewtest_SF1/3view"
new_folder = "/path/to/new/folder"

# new_folder = os.path.join(original_folder, 'copy')
# if not os.path.exists(new_folder):
#     os.mkdir(new_folder)

# imgs = os.listdir(original_folder)


# Copy the images from the original folder to the new folder
# for src in imgs:
#     # Set the source and destination paths for the image
#     src = os.path.join(original_folder, src)
#     dst = os.path.join(new_folder, src)
    
#     # Copy the image to the new folder
#     shutil.copy(src, dst)

#     original_file_path = os.path.join(original_folder, f"{i}.jpg")
#     # Get the new file path
#     new_file_path = os.path.join(reversed_directory_path, f"{84-i}.jpg")
#     # Copy the file
#     shutil.copyfile(original_file_path, new_file_path)
for i in range(43):
    # Get the original file path
    original_file_path = os.path.join(original_folder, f"image{i:03d}.jpg")
    # Get the new file path
    new_file_path = os.path.join(original_folder, f"image{85-i:03d}.jpg")
    # new_file_path = os.path.join(original_folder, f"image{i+43:03d}.jpg")
    # Copy the file
    shutil.copyfile(original_file_path, new_file_path)
# # Rename the images in the new folder in reverse order
# for i in range(43, 85):
#     # Set the source and destination paths for the image
#     src = os.path.join(new_folder, str(i-43) + ".jpg")
#     dst = os.path.join(original_folder, str(i) + ".jpg")
    
#     # Copy the reversed image to the original folder
#     shutil.copy(src, dst)