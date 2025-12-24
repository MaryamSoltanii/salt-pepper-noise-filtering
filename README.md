# Salt & Pepper Noise Filtering (Min, Max, Median)

## Overview
This repository demonstrates **manual removal of Salt & Pepper noise**
using classic **nonlinear spatial filters** implemented from scratch.

The following filters are implemented without using OpenCV built-in functions:
- Min Filter
- Max Filter
- Median Filter
- Min → Max (Opening-like)
- Max → Min (Closing-like)

The image is padded manually using **replicate padding**.

## Noise Model
- Salt & Pepper noise
- 5% Salt (white pixels)
- 5% Pepper (black pixels)
- Positions chosen randomly

## Implemented Steps
1. Load grayscale image (Lena)
2. Add Salt & Pepper noise (exact distribution)
3. Apply replicate padding manually
4. Apply nonlinear filters with 3×3 kernels
5. Compare filtering results visually

## Files
