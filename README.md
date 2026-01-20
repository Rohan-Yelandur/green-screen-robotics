# Green Screen Replacement
This repo contains scripts to replace the green screen in hdf5 files with background images. Allows for synthetically multiplying the number of demonstrations in a dataset.

## Usage

`python green_screen.py <input_h5_file_or_folder> <background_image_or_folder>`

`python h5_viewer.py <input_h5_file>`

`python extract_video.py <input_h5_file>` -> outputs mp4


Change the HSV Configuration in green_screen.py to change the color of the green screen.

## Techniques Used

### 1. HSV Color Space
Isolates green hues more reliably than RGB by separating color information from brightness, handling shadows better.

### 2. Morphological Operations
Fills small holes and removes noise to ensure a clean, solid subject mask.