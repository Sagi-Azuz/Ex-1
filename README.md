# 3D Volume Plane Projection

## Overview

This project implements a 3D plane projection system that extracts 2D images from a 3D volumetric dataset.

The program receives:
- A 3D volume stored in NRRD format.
- A CSV file containing plane definitions using either three or four 3D points.

For each plane, the program validates its geometry and generates a 2D projection when the plane is valid. Invalid planes are handled by generating an Error Frame.

All generated frames are combined into an MP4 video while preserving the original order of the planes in the CSV file.
---

## Project Demo

<img width="500" alt="output_280-ezgif com-gif-to-mp4-converter" src="https://github.com/user-attachments/assets/4b6836b6-1a91-4073-8a18-5d72f372a42f" />

---
## Problem Definition

The purpose of this project is to develop an algorithm that can extract 2D projections from arbitrary planes defined inside a 3D volume.

The system must validate the geometric properties of each plane, sample the corresponding positions in the volume, and generate an output image.

The system must also handle invalid plane definitions without stopping the processing of the remaining valid planes.

## Project Scope

### Included

- Loading 3D volumetric data from NRRD files.
- Reading plane definitions from CSV files.
- Supporting planes defined by three or four points.
- Validating plane geometry using the required geometric conditions.
- Searching for a valid ordering of plane points when necessary.
- Calculating 2D plane projections.
- Sampling voxel values using trilinear interpolation.
- Assigning zero to samples outside the volume.
- Generating Error Frames for invalid planes.
- Creating an MP4 video containing all frames in their original order.
- Measuring projection performance.

### Outside the Scope

- 3D volume reconstruction.
- Medical diagnosis or clinical interpretation.
- Conversion of physical coordinates into voxel coordinates.

## Input and Output

### Input

**NRRD Volume**

A 3D volumetric dataset containing voxel intensity values.

**CSV File**

A CSV file containing plane definitions. Each row describes a plane using either three or four points.

Each point contains three coordinates:

`(x, y, z)`

The coordinates are interpreted as voxel indices.

### Output

An MP4 video containing one frame for every plane in the input CSV file.

- Valid planes produce 2D projection images.
- Invalid planes produce Error Frames.
- The original CSV plane order is preserved.

## Algorithm Overview

The algorithm consists of the following stages:

1. **Volume Loading** – Load and validate the 3D NRRD volume.
2. **Plane Parsing** – Read plane definitions from the CSV file.
3. **Geometry Validation** – Verify that each plane satisfies the required geometric conditions.
4. **Point Ordering** – Search for a valid point ordering when the original ordering is invalid.
5. **Plane Projection** – Calculate the 2D projection dimensions and sample positions.
6. **Voxel Sampling** – Calculate intensity values using trilinear interpolation.
7. **Error Handling** – Generate an Error Frame when a plane is invalid.
8. **Video Generation** – Combine all frames into an MP4 video while preserving the input order.

### Trilinear Interpolation

The project uses trilinear interpolation to estimate voxel intensity values at non-integer coordinates.

The method considers the eight neighboring voxels surrounding the sampling position and calculates a weighted intensity value.

This allows smoother sampling compared with using only the nearest voxel.

Samples outside the volume are assigned a value of zero.

## Experiment & Results

As required by the assignment guidelines, we conducted a projection experiment using the `MRHead_1.nrrd` volumetric dataset and the `lecturer_example.csv` plane definitions file. 

The program successfully processed the input, validated the geometric properties of the provided planes, and generated the corresponding 2D projections. Invalid planes within the dataset triggered the error-handling mechanism, properly generating distinct Error Frames without interrupting the execution. 

Below are specific frame results extracted during the experiment:

**Frame 1 – Valid Plane Projection**

<img width="500" alt="image" src="https://github.com/user-attachments/assets/0a857cd9-94b4-432d-9e74-91f8ac4bcbab" />

**Frame 2 – Valid Plane Projection**

<img width="500" alt="image" src="https://github.com/user-attachments/assets/78303ebe-106a-4085-9d5a-77ccd9b8ff92" />

**Frame 3 – Invalid Plane (Error Frame Handling)**

<img width="500" alt="image" src="https://github.com/user-attachments/assets/cd61f2af-941e-4a76-858f-6de48a0bfa08" />

**Frame 4 – Additional Valid Projection**

<img width="500" alt="Screenshot" src="https://github.com/user-attachments/assets/51838688-a521-4e1c-a360-2f5df88dd1b7" />

All generated frames were sequentially compiled into the final output video. The runtime performance met the assignment benchmarks, sustaining an average projection speed well above the 30 FPS minimum threshold.

**Complete MP4 Output Video:**

<img width="500" alt="ezgif com-video-to-gif-converter (1)" src="https://github.com/user-attachments/assets/ea6575bb-f103-41e2-8e92-c39831beeeb1" />

## Repository Structure

```text
Ex-1/
│
├── project.py
├── test_project.py
├── README.md
│
├── data/
│   ├── nrrd/
│   │   ├── MRHead_1.nrrd
│   │   ├── biscuit_cake.nrrd
│   │   ├── croissant.nrrd
│   │   ├── onion.nrrd
│   │   └── swiss_roll.nrrd
│   │
│   └── csv/
│       ├── Ex0.csv - Sheet1.csv
│       ├── biscuit_cake.csv
│       ├── croissant.csv
│       ├── edge_cases.csv
│       ├── head.csv
│       ├── lecturer_example.csv
│       ├── onion.csv
│       └── swiss_roll.csv
│
└── docs/
    ├── SRS.md
    └── algorithm-design-revised.md

```

## How to Run

The project is executed using Python through the Windows `py` launcher.

### Command

```bash
py project.py <input_volume.nrrd> <input_planes.csv> <output_video.mp4>
```

### Example

```bash
py project.py data/nrrd/MRHead_1.nrrd data/csv/lecturer_example.csv output.mp4
```

The program generates the output video at the specified path.


## Testing

The project includes unit tests for the main components:

- Euclidean distance calculations.
- Voxel access and volume boundaries.
- Trilinear interpolation.
- Volume validation.
- Three-point and four-point plane geometry.
- Plane ordering.
- Invalid plane handling.
- Plane projection.
- Video generation.

To run the tests from the repository root:

```bash
py -m pytest -v test_project.py
```

## Performance

The program measures projection computation performance.

The performance report includes:
- Number of valid projections.
- Projection computation time.
- Average projection FPS.
- Processed pixels per second.
- Video generation time.

The performance targets are:
- Minimum: 30 projections per second.
- Desired: 100 projections per second.

Numba warm-up is performed before projection timing so that initial compilation time is excluded from the projection FPS measurement.

## Assumptions and Limitations

- CSV coordinates are interpreted as voxel indices `(x, y, z)`.
- NRRD spatial metadata is displayed but is not used to convert physical coordinates.
- Samples outside the volume are assigned zero.
- Invalid planes generate Error Frames.
- The output video preserves the original order of all input planes.
- Projection values are rounded, clipped to the range 0–255, and converted to 8-bit grayscale for video output.
- The generated images are mathematical projections and are not intended for medical diagnosis.

## Documentation

Additional technical documentation is available in the `docs/` directory:


- [Software Requirements Specification](docs/SRS.md)
- [Algorithm Design](docs/algorithm-design-revised.md)

## Authors

Developed by Sagi Azuz and Liel Avidor as part of an academic project.

