# Algorithm Design --- Plane Projection onto a 3D Volume

## 1. Algorithm Purpose

The algorithm receives a 3D volume `M` and a plane defined by either
three or four points. It computes a 2D projection image `E` from the
volume values sampled along the plane.

## 2. Coordinate System and Volume Structure

The volume is stored in an array with shape `(sz, sy, sx)`. Therefore,
voxel access follows the order `M[z, y, x]`.

Coordinates from the CSV file are used directly as voxel coordinates
`(x, y, z)`. NRRD metadata fields such as `space`, `space origin`, and
`space directions` are displayed but are not used to convert
coordinates.

## 3. Point Validation and Ordering

For each plane, the applicable geometric conditions for three-point or
four-point definitions are checked using a tolerance of `EPS = 0.001`.

If the original point order does not satisfy the conditions, possible
permutations of the points are checked. The first valid ordering found
is used to compute the projection. If no valid ordering is found, the
plane is skipped.

## 4. Projection Image Dimensions

Let:

-   `H` --- Projection image height in pixels.
-   `W` --- Projection image width in pixels.

For a plane defined by three points:

-   `H` is calculated from the distance between `a` and `b`.
-   `W` is calculated from the distance between `b` and `c`.

For a plane defined by four points:

-   `H` is calculated from the distance between `a` and `c`.
-   `W` is calculated from the distance between `c` and `d`.

Each dimension is rounded using `round()` and limited to a minimum of
one pixel:

\[ H = `\max`{=tex}(1, `\operatorname{round}`{=tex}(D_H)) \]

\[ W = `\max`{=tex}(1, `\operatorname{round}`{=tex}(D_W)) \]

## 5. Row and Sample Computation on the Plane

For each row `j`, where `0 \leq j < H`, a relative position is
calculated at the center of the corresponding height segment:

\[ u = `\frac{j + 0.5}{H}`{=tex} \]

The endpoints of the row in 3D space are calculated according to the
plane type.

**Three-point plane:**

\[ A_1 = a + u(b-a) \]

\[ A_2 = a + u(c-a) \]

**Four-point plane:**

\[ A_1 = a + u(c-a) \]

\[ A_2 = b + u(d-b) \]

The distance between the row endpoints is then calculated, and the
number of samples is determined as follows:

\[ N = `\max`{=tex}(1,
`\operatorname{round}`{=tex}(`\operatorname{dist}`{=tex}(A_1,A_2))) \]

The position of each sample along the row is calculated at the center of
an equally sized segment:

\[ v = `\frac{i + 0.5}{N}`{=tex} \]

\[ P = A_1 + v(A_2-A_1) \]

where `i` is the sample index, in the range `0 \leq i < N`.

## 6. Sample Placement in the Output Image

Each row is horizontally centered within the projection image. The
starting column is calculated by subtracting the number of samples from
the image width, dividing the remaining space by two, and rounding down
to the nearest integer:

\[ x\_{`\text{start}`{=tex}} =
`\left`{=tex}`\lfloor `{=tex}`\frac{W-N}{2}`{=tex}
`\right`{=tex}`\rfloor`{=tex} \]

Each sample is stored in the column:

\[ x = x\_{`\text{start}`{=tex}} + i \]

Samples whose column positions are outside the range `0` to `W-1` are
not stored. The projection matrix is initialized with zeros, so areas
that do not receive a sample remain zero.

## 7. Sampling Values from the Volume

If a sample point is outside the volume boundaries, its value is set to
0. Otherwise, its value is calculated using trilinear interpolation from
the eight neighboring voxels.

If the sample point lies exactly at a voxel location, the voxel's value
is returned directly. Voxels outside the volume boundaries are assigned
a value of 0 during interpolation as well.

## 8. Video Generation

After all valid projections have been computed:

1.  The maximum height and width among all projections are determined.
2.  If either video dimension is odd, it is increased by one pixel.
3.  Each projection is centered inside a black frame of uniform
    dimensions.
4.  Projection values are rounded, clipped to the range 0--255, and
    converted to `uint8`.
5.  The frames are written to an MP4 file using OpenCV at 60 FPS.

## 9. Performance Measurement

Before projection timing begins, the Numba functions are warmed up.
Warm-up time is excluded from projection FPS measurements but included
in the total processing time.

The projection computation time, number of projections, projections per
second, and pixels per second are measured. Video generation time is
measured separately.
