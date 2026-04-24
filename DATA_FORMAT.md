# Real-Robot Data Inventory & Format

Exact filesystem paths and exact array schemas for every artifact under
`real_robot/`. No pipeline description.

`{vid}` everywhere is a video_id of the form
`real_robot__{benchmark}__{task_id}__{policy}__{trial}`,
e.g. `real_robot__workroom_v1__blue_mug_box__MolmoBot-Img__1`.

---

## 0. Source videos (read-only)

```
/weka/prior-default/chenhaoz/home/MotionPlanner/molmobot_videos/droid_eval/
  {benchmark}/{task_id}/{policy}_{trial}_{succ|fail}.mp4
```
- 4 benchmarks: `workroom_v1`, `kitchen_v3`, `bedroom_v1`, `robomolmo_princeton`
- 5 policies: `MolmoBot`, `MolmoBot-Img`, `MolmoBot-Pi0`, `pi05-DROID`, `pi0-DROID`
- All clips: 1280×720 RGB, ~15.2 fps, h.264, ~2–5 MB.

---

## 1. Task list

```
tmp/real_robot_tasks.json
tmp/real_robot_shard{00..09}.json
```
List of dicts:
```json
{
  "video_id": "real_robot__workroom_v1__blue_mug_box__MolmoBot-Img__1",
  "video_path": "/weka/.../MolmoBot-Img_1_succ.mp4",
  "language_instruction": "",
  "has_annotation": false,
  "prompts": [],
  "fps": 15.0,
  "benchmark": "workroom_v1",
  "task_id": "blue_mug_box",
  "policy": "MolmoBot-Img",
  "trial": 1
}
```

---

## 2. Captions / prompts JSON

```
tmp/combined_prompts_real_robot{tag}.json     # one per dataset_tag
tmp/real_robot_captions.json                  # alternate location
```
Per-`{vid}` entry:
```json
{
  "skip": false,
  "molmo2_8b_recaption": "blue ceramic coffee mug",
  "object_text":         "blue ceramic coffee mug",
  "point_prompt":        "point to ... gripped and picked up by the robot gripper",
  "final_obj":           "blue ceramic coffee mug",
  "per_object_list":     ["blue ceramic coffee mug"],
  "per_object_prompts":  ["point to ..."],
  "molmo2_prompt":       "point to ...",
  "llm_objects":         []
}
```
(Some keys may be absent depending on which writer produced the file; consumers should look for `molmo2_8b_recaption` ∥ `object_text` ∥ `final_obj`.)

---

## 3. Query points

```
vipe/runs/{vid}/querypoints/
  query_points/{vid}_molmo2_{obj_tag}_f{frame}_f{frame}.npz
  query_points/{vid}_molmo2_meta.json
  masks/{vid}_*.png        # optional
  debug_png/{vid}_*.png    # optional
```
`{obj_tag}` is the alphanumeric-only sanitization of the object string.

`*_f{frame}_f{frame}.npz`:
| key | dtype | shape | meaning |
|---|---|---|---|
| `query_points` | int32 | (N, 3) | columns = `(frame_idx, x, y)`, source resolution. N=100 per object per seed-frame. |
| `dim` | int32 | (2,) | `(H, W)` source resolution. |

`*_molmo2_meta.json`: text record of the prompt, raw point output, and SAM3 mask area.

---

## 4. ViPE outputs

```
vipe/.vipe_input/{vid}.mp4                           # 480p re-encode
vipe/vipe_results/rgb/{vid}.mp4                      # 480p, h264
vipe/vipe_results/depth/{vid}.zip                    # one EXR per frame, single Z channel (meters, camera frame)
vipe/vipe_results/pose/{vid}.npz
vipe/vipe_results/intrinsics/{vid}.npz
vipe/vipe_results/pose/{vid}.npz.vipe_orig           # backup of original ViPE poses (if overwritten)
```

`pose/{vid}.npz`:
| key | dtype | shape | meaning |
|---|---|---|---|
| `data` | float32 | (T, 4, 4) | per-frame **camera-to-world** matrices. |
| `inds` | int64 | (T,) | source frame indices. |

`intrinsics/{vid}.npz`:
| key | dtype | shape | meaning |
|---|---|---|---|
| `data` | float32 | (T, 4) | per-frame `(fx, fy, cx, cy)` in **480p** image space. |
| `inds` | int64 | (T,) | source frame indices. |

---

## 5. 2D tracks (intermediate)

```
vipe/track_output/{vid}/
  qp_frame_{f}.npz                  # query pts re-scaled to 480p (intermediate)
  {vid}_f{f}.npz                    # per-seed-frame fwd+bwd 2D tracks
  {vid}_merged.npz                  # all per-seed-frame tracks concatenated
```

`{vid}_merged.npz`:
| key | dtype | shape | meaning |
|---|---|---|---|
| `tracks` | float32 | (T, N, 2) | per-frame `(x, y)` in **source** resolution. N = 100 × #seed-frames. |
| `visibility` | bool | (T, N) | per-frame visibility flag. |
| `dim` | int64 | (2,) | `(H, W)` source resolution. |

---

## 6. 3D tracks (raw, pre-filter)

```
vipe/colmap_output/{dataset_tag}/{vid}_merged_3d_tracks.npz
```
`{dataset_tag}` ∈ {`real_robot_shard00`..`real_robot_shard09`, `real_robot_smoke`}.

| key | dtype | shape | meaning |
|---|---|---|---|
| `points_3d` | float32 | (N, T, 3) | 3D points in camera frame (= world if poses are identity). NaN for invalid backprojection. |
| `visibility` | bool | (N, T, 1) | per-point per-frame visibility. |

---

## 7. Final tracks (consumable output)

```
vipe/final_tracks/real_robot/
  {vid}_2d.npz
  {vid}_3d.npz
  {vid}_filter_meta.npz
```

`{vid}_2d.npz`:
| key | dtype | shape | meaning |
|---|---|---|---|
| `tracks` | float32 | (T, N, 2) | filtered 2D tracks at source resolution. |
| `visibility` | bool | (T, N) | per-frame visibility. |
| `dim` | int64 | (2,) | `(H, W)` source resolution. |

`{vid}_3d.npz`:
| key | dtype | shape | meaning |
|---|---|---|---|
| `points_3d` | float32 | (N, T, 3) | filtered + smoothed 3D points (camera frame, meters). NaN where dropped or invisible. |
| `visibility` | bool | (N, T, 1) | per-frame visibility. |

`{vid}_filter_meta.npz` — diagnostic:
| key | dtype | shape | meaning |
|---|---|---|---|
| `P_original`        | float32 | (N, T, 3) | unsmoothed lifted 3D points. |
| `P_smoothed`        | float32 | (N, T, 3) | smoothed (= `_3d['points_3d']`). |
| `visibility_all`    | bool    | (N, T)    | union visibility before filter. |
| `trust_weights`     | float32 | (N, T)    | per-point per-frame ray-optimization weight. |
| `keep_mask`         | bool    | (N,)      | True = track kept. |
| `drop`              | bool    | (N,)      | True = track dropped. |
| `sub_object_labels` | int64   | (N,)      | object index. |

---

## 8. Visualization website inputs

```
viz_website/static/data/real_robot/{vid}.npz
viz_website/static/data/manifest.json
```

`{vid}.npz`:
| key | dtype | shape | meaning |
|---|---|---|---|
| `pc_xyz`     | float16 | (≤25000, 3) | dense scene point cloud (depth backprojection, subsampled). |
| `pc_colors`  | uint8   | (≤25000, 3) | RGB color per point. |
| `cam_pos`    | float16 | (T, 3)      | camera position trail (meters). |
| `cam_fwd`    | float16 | (T, 3)      | camera forward (Z) axis per frame. |
| `cam0_c2w`   | float32 | (4, 4)      | frame-0 camera-to-world matrix. |
| `intrinsics` | float32 | (4,)        | `(fx, fy, cx, cy)` of frame-0 (480p space). |
| `pts3d`      | float16 | (N, T, 3)   | filtered 3D track points. N capped to 150. |
| `vis3d`      | bool    | (N, T)      | per-point visibility. |
| `tracks2d`   | float16 | (T, N, 2)   | 2D tracks (source resolution). |
| `vis2d`      | bool    | (T, N)      | per-frame visibility. |
| `dim`        | int32   | (2,)        | `(H, W)` of 2D tracks. |

`manifest.json`:
```json
{
  "egodex":     [...],
  "hdepic":     [...],
  "xperience":  [...],
  "real_robot": [
    {
      "id":         "real_robot__workroom_v1__blue_mug_box__MolmoBot-Img__1",
      "benchmark":  "workroom_v1",
      "task":       "blue_mug_box",
      "policy":     "MolmoBot-Img",
      "trial":      "1",
      "group":      "workroom_v1/blue_mug_box",
      "text":       "<recaption | object string>",
      "rgb_url":    "/video/real_robot/<id>",
      "data_url":   "/static/data/real_robot/<id>.npz",
      "rgb_path":   "/weka/prior-default/.../<source>.mp4"
    }
  ]
}
```

---

## 9. GitHub Pages snapshot

```
/tmp/gentraj-pages/
  index.html
  .nojekyll
  static/data/manifest.json                # real_robot only, URLs rewritten
  static/data/real_robot/{vid}.json        # NaN/Inf → null
  videos/real_robot/{vid}.mp4              # copy of source MP4
```

Repo : https://github.com/jianingnz/gentraj-real-robot-viz
Live : https://jianingnz.github.io/gentraj-real-robot-viz/

`{vid}.json` schema is identical to the per-clip NPZ above (§8), JSON-encoded.

---

## 10. Logs

```
outputs/real_robot/smoke_test.log
outputs/real_robot/gantry_experiment_ids.txt
```

---

## 11. Coordinate / unit conventions

- 2D tracks: source resolution `(H=720, W=1280)`, `(x, y) = (col, row)`, integer pixel space (float in `_2d.npz`).
- 3D tracks: camera frame, **meters**, `+Z = forward, +X = right, +Y = down` (OpenCV convention).
- Camera poses (`pose/*.npz['data']`): camera-to-world 4×4 SE(3).
- Intrinsics: `(fx, fy, cx, cy)` in **480p** image space.
- Frame indexing: contiguous `[0, T)` aligned to the 480p re-encode (`vipe/.vipe_input/{vid}.mp4`); `inds` arrays give the source-frame index for each kept frame.
