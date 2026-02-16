# Motion Scenario Metadata Schema (Draft)

## Camera / Motion
- camera_translation_speed: numeric
- camera_rotation_speed: numeric
- acceleration_level: numeric or categorical
- motion_type: categorical (ego / object / mixed)

## Scene Difficulty
- occlusion_ratio: numeric [0,1]
- motion_blur_level: numeric or categorical
- illumination_change: numeric or categorical
- texture_level: categorical (low/med/high)
- dynamic_objects_ratio: numeric [0,1]

## Notes
- Source of each feature: (dataset-provided / computed)
- Missing values: (how handled)
