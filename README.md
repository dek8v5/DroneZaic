# DroneZaic: End-to-End Pipeline for Mosaicking UAV-Captured Aerial Imagery

DroneZaic is an advanced mosaicking algorithm designed to process freely flown aerial imagery captured by UAVs. 

## 5 Novel Contributions

1. **Dynamic Sampling**
   - Optimizes frame selection based on UAV movements using optical flow.
   - Sampling based on overlap for forward, backward, and sideways, and more frequency on more complicated movements such as rotation and zoom.

2. **Automatic Lens Calibration**
   - automatic calibration of the camera lens and accounts for gimbal misalignment due to wear and tear.
   - For a more accurate image alignment and reduce distortion in the final mosaic.

3. **Homography Estimation with Cornetv3**
   - Utilizes the Cornetv3 architecture for a more robust and faster homography estimation.
   - Improves accuracy by trained the new model with a more balanced movements, different intensities, augmented with synthetic color changes, and different maize growth and soil colors.

4. **Shot Detection via UAV Movement Analysis**
   - Detects shot boundaries by monitoring changes in UAV movement.
   - Grouping the same UAV movements in the same mini mosaics.

5. **Mini Mosaicking to Minimize Error Accumulation**
   - Processes images in smaller sections (mini mosaics) to limit error propagation.
   - Higher accuracy in large-scale mosaics.

## Getting Started

### Prerequisites
- Python 2.x
- OpenCV
- Tensorflow
- NumPy
- Other dependencies as specified in `requirements.txt`

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/dek8v5/DroneZaic.git
   ```
2. Navigate to the project directory:
   ```bash
   cd DroneZaic/code
   ```
3. Install the required packages
   ```bash
   sudo install -r requirement.txt
   ```

### Running MaiZaic
1. locate your video that you want to mosaic. Ensure your video is supported and (optionally) locate the accompanying .SRT file where GPS metadata is embedded.

2. dynamic sample first
The `dynamic_sampling.py` script extracts keyframes from UAV videos based on optical flow. It now also supports optional GPS extraction from DJI `.SRT` metadata for georeferenced processing.
#### With GPS Metadata (DJI `.SRT` Format)
**Supported formats:** `JPG`, `TIF`  
**Input requirements:** DJI `.MOV` file and corresponding `.SRT` file
``` bash
python dynamic_sampling/dynamic_sampling.py -video /media/dek8v5/f/aerial_imaging/images/23r/grace/23.6/DJI_0205.MOV -save_path /home/dek8v5/Documents/cornetv2/data_ori/FINAL_CORNETV2_DATASET/1_gps_jpg_23r_06_23_205_seedling_parallel_1pass/jpeg -srt /media/dek8v5/f/aerial_imaging/images/23r/grace/23.6/DJI_0205.SRT -win 100 -scale 3 -fname 23r_06_23 -format jpg
```
**Expected .SRT line format (based on our DJI drone):**
``` bash
110
00:00:04,547 --> 00:00:04,589
<font size="36">FrameCnt : 110, DiffTime : 42ms
2023-06-23 11:26:04,097,455
[iso : 100] [shutter : 1/3200.0] [fnum : 280] [ev : -0.7] [ct : 5502] [color_md : default] [focal_len : 280] [latitude : 38.904148] [longtitude : -92.281307] [altitude: 274.239014] </font>
```

#### Without GPS
``` bash
python dynamic_sampling/dynamic_sampling.py -video /media/dek8v5/f/aerial_imaging/images/23r/grace/23.6/DJI_0205.MOV -save_path /home/dek8v5/Documents/cornetv2/data_ori/FINAL_CORNETV2_DATASET/1_gps_jpg_23r_06_23_205_seedling_parallel_1pass/png -win 100 -scale 3 -fname 23r_06_23 -format png
```

3. calibrate the frame extracted from the dynamic sampling
The `calibration.py` script performs automatic lens calibration to correct for lens distortion and gimbal misalignment. Calibration is essential for accurate frame alignment and high-quality mosaicking.

The lens and gimbal parameters are **unique for each drone** and should ideally be derived from a checkerboard calibration process. If you haven't already obtained these parameters, you can generate them using the `automatic_calibration.py` script.

   ```bash
   python calibration.py -image_path /path/to/your/raw/frames  -save_path /path/to/output_directory
   ```

#### Overwriting Raw Frames

To overwrite the original raw frames with their calibrated versions, set `-image_path` and `-save_path` to the same directory:

```bash
python calibration.py -image_path /path/to/raw/frames -save_path /path/to/raw/frames
```

4. run the calibrated frames to the homography estimator of your choice
   ```bash
   example using surf:
   
   cd surf
   python surf_homography_estimation.py -image_path /path/to/your/calibrated/frames -save_path /path/to/the/working/directory -scale int (for resizing the frame dimension)
   
   ```
      
5. run the mini-mosaic script
#### With Step-by-Step Duplication
This mode saves all intermediate steps, including raw frames, calibrated frames, and an additional copy of the calibrated frames grouped into shots. It is useful for debugging, inspection, and retaining full processing history.

   ```bash
    ./maizaic_run.sh -p /path/to/project_dir -h asift -d true

    # or

    ./maizaic_run.sh --working_path /path/to/project_dir --hm_method asift --mode_duplicate true
   
   ```
#### Without Duplication (Overwrites intermediate steps):
This mode replaces the raw frames with their calibrated versions and moves the calibrated frames into grouped shots. It is optimized for minimal storage usage and does not retain intermediate or duplicate files.

   ``` bash
   ./maizaic_run.sh -p /path/to/project_dir -h asift -d false

   # or

   ./maizaic_run.sh --working_path /path/to/project_dir --hm_method asift --mode_duplicate false
   ```
### Citation
If you use _DroneZaic_ or any part of this pipeline, please cite our work.

#### the paper
```bibtex
@article{kharismawati2025dronezaic,
  author    = {Dewi Endah Kharismawati and Toni Kazic},
  title     = {\emph{DroneZaic}: a robust end-to-end pipeline for mosaicking freely flown aerial video of agricultural fields},
  journal   = {The Plant Phenone Journal},
  doi       = {10.1002/ppj2.70033}
  year      = {2025}
}
```

#### for the dataset
```bibtex
@misc{kharismawati2025dronezaicdata,
  author    = {Dewi Endah Kharismawati and Toni Kazic},
  title     = {\emph{DroneZaic Dataset}: a robust end-to-end pipeline for mosaicking freely flown aerial video of agricultural fields},
  publisher = {Dryad},
  doi       = {10.5061/dryad.r4xgxd2q7},
  year      = {2025}
}
```
