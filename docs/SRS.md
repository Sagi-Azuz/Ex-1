# Software Requirements Specification (SRS)

## Projection of Planes from a 3D Volume

### 1. System Purpose

The software reads a 3D volume in NRRD format and plane definitions from
a CSV file. It computes a 2D projection image for each valid plane and
combines the resulting images into an MP4 video.

### 2. Input and Output

The software is executed from the command line:

``` bash
python project.py file.nrrd file.csv file.mp4
```

-   `file.nrrd` --- A 3D volume and its metadata.
-   `file.csv` --- Plane definitions, where each row contains either 3
    or 4 points with X, Y, and Z coordinates.
-   `file.mp4` --- The output video file path.

**Input assumption:** Coordinates in the CSV file are interpreted as
voxel indices within the volume. NRRD metadata is displayed for
informational purposes but is not used to convert physical coordinates.

### 3. Functional Requirements

**FR-1 — Load Volume:** Read the NRRD file using C index order,
verify that the volume is 3D, ensure that its dimensions are within
the range 1–1023, and verify that the input values are non-negative
and within the supported range. Convert the volume data to `uint16`
before processing.

**FR-2 --- Read Plane Definitions:** Read the CSV file, skip the header
row and empty rows, and parse planes defined by either 3 or 4 points.

**FR-3 --- Validate Geometry:** Check the geometric conditions for
planes defined by 3 or 4 points using `EPS = 0.001`. If the original
point order is invalid, the software checks possible permutations of the
points and accepts the first valid ordering found.

**FR-4 — Handle Invalid Planes:** A plane for which no valid point
ordering is found is reported, and an error frame is added to the
output video in its corresponding position. Processing continues
with the remaining planes. Errors in the input structure or file
loading stop the processing.

**FR-5 — Compute Projections:** Generate a 2D projection matrix
for each valid plane. The projection dimensions are determined
by the geometric distances between the defining points. Sampling
positions are calculated using midpoint sampling along the
corresponding plane segments.

**FR-6 --- Sample Volume Values:** Sample volume values using trilinear
interpolation. Points outside the volume, as well as neighboring voxels
outside the required interpolation boundaries, receive a value of 0.

**FR-7 --- Generate Video:** Create an MP4 video at 60 frames per
second. Each projection is centered inside a black frame of uniform
dimensions. Video dimensions are adjusted to even pixel values.
Projection values are rounded, clipped to the range 0--255, and
converted to 8-bit values.

**FR-8 — Preserve Order:** The video frames must follow the order
of the plane definitions in the CSV file. Invalid planes must be
represented by error frames in their original positions.

**FR-9 --- Report Performance:** Display computation times, the number
of projections, projection dimensions, projections per second, and pixel
processing throughput. Numba warm-up is excluded from projection timing
but included in the total processing time.

### 4. Non-functional Requirements

**NFR-1 --- Performance:** The minimum performance target is 30
projections per second, with a desired target of 100 projections per
second. This metric refers to projection computation, not video
playback.

**NFR-2 --- Efficiency:** Use Numba to accelerate geometric and sampling
computations, and NumPy/OpenCV for data processing and video generation.

**NFR-3 --- Maintainability:** Separate input loading, geometry
validation, projection computation, and video generation into distinct
components.

### 5. Items Requiring Further Verification

-   Verify that the point ordering for four-point planes matches the
    assignment's exact definition and projection formulas.
-   Verify whether physical coordinates are required. The current
    implementation uses CSV coordinates directly as voxel indices.
-   Check whether a different value mapping is required for video
    output, since values above 255 are clipped to 255.
-   Verify whether the FPS target applies only to projection computation
    or to the entire processing pipeline.

### 6. Implementation Decisions

- **Coordinate System:** Coordinates provided in the CSV file are
  interpreted as voxel indices (X, Y, Z). Physical coordinates from
  NRRD metadata are not converted.

- **Data Type:** The volume data is converted to `uint16` before
  projection processing. Input values must be non-negative and
  within the `uint16` range.

- **Interpolation:** Trilinear interpolation is used to calculate
  values at non-integer sampling positions. Out-of-volume voxel
  accesses return 0.

- **Video Intensity Mapping:** Projection values are rounded and
  clipped to the range 0–255 before being converted to 8-bit
  grayscale images.

- **Performance Measurement:** Performance is measured using
  projection computation time, excluding Numba warm-up time.
  Video encoding and playback are measured separately.