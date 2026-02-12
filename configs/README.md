# Tracking Presets

This directory contains preset configuration files for different model organisms. These presets set appropriate default values for tracking parameters optimized for each species.

## Available Presets

### `default.json` - Zebrafish (General Purpose)
General-purpose defaults optimized for zebrafish larvae tracking. Good starting point for most aquatic organisms.

### `obiroi.json` - Oryzias bripes (Medaka Fish)
Optimized for medaka fish behavioral tracking. Adjusted for their size and swimming patterns.

### `dmelanogaster.json` - Drosophila melanogaster (Fruit Fly)
Optimized for fruit fly tracking in behavioral arenas. Higher FPS, smaller body size, dark-on-light background.

### `custom.json` - User Custom (gitignored)
Your personal saved defaults. Created when you click "Save as Custom" in the GUI. This file is gitignored so your custom settings won't be committed to version control.

## Creating Custom Presets

You can create additional preset files for your specific organism:

1. Copy an existing preset file as a template
2. Rename it (e.g., `celegans.json`)
3. Edit the `preset_name` and `description` fields
4. Adjust parameter values for your organism
5. Place it in this directory
6. It will appear automatically in the GUI preset selector

## Parameter Guidelines

**Body Size**: Typical body length/diameter in pixels at native resolution
**Resize Factor**: 0.5-1.0, lower for high-res videos to speed up processing
**FPS**: Match your camera's frame rate
**Detection Method**: "yolo" for general use, "background" for stationary cameras
**Size Filtering**: Enable to reject debris/reflections outside expected size range
**Max Targets**: Expected maximum number of animals in frame
**Max Distance**: Maximum movement between frames (in body lengths)

## Sharing Presets

To share optimized settings with colleagues:
1. Export your preset file from this directory
2. Share the JSON file via email/wiki/repository
3. Recipients place it in their `configs/` directory
4. Preset appears in their GUI automatically

## Version Control

- `default.json`, `obiroi.json`, `dmelanogaster.json` are version controlled
- `custom.json` is gitignored (personal settings)
- Add your custom presets to `.gitignore` if they contain sensitive paths
