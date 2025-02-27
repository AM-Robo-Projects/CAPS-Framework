# Code Citations

## License: unknown
https://github.com/skumra/robotic-grasping/tree/183c6f68c44c1c7ff0f07707e2db6fcfd6840d2d/hardware/camera.py

```
class RealSenseCamera:
       def __init__(self,
                    device_id,
                    width=640,
                    height=480,
                    fps=6):
           self.device_id = device_id
           self.width = width
           self.height = height
           self.fps = fps

           self.pipeline = None
           self.scale = None
           self.
```


## License: unknown
https://github.com/SeyedHamidreza/cognitive_robotics_manipulation/tree/e5431da52bb8379c78d04a5c2246ed161fba1d3b/network/hardware/camera.py

```
)

   class RealSenseCamera:
       def __init__(self,
                    device_id,
                    width=640,
                    height=480,
                    fps=6):
           self.device_id = device_id
           self.width = width
           self.height = height
           self.fps = fps

           self.pipeline = None
           self.scale = None
           self
```


## License: LGPL_2_1
https://github.com/hsp-panda/grasping-benchmarks-panda/tree/f05b5d3099f0ba98f50a084fd29be7b2b6c97836/grasping_benchmarks/cv_grasps/cv_grasp_planner.py

```
self.pipeline.wait_for_frames()

           align = rs.align(rs.stream.color)
           aligned_frames = align.process(frames)
           color_frame = aligned_frames.first(rs.stream.color)
           aligned_depth_frame = aligned_frames.get_depth_frame()

           depth_image = np.asarray(aligned_depth_frame.get_data
```


## License: MIT
https://github.com/chansoopark98/Tensorflow-Keras-Semantic-Segmentation/tree/7baa7cc6c909be33818ea592cf96947ef5aeb8f4/utils/realsense_camera.py

```
(self):
           images = self.get_image_bundle()

           rgb = images['rgb']
           depth = images['aligned_depth']

           fig, ax = plt.subplots(1, 2, squeeze=False)
           ax[0, 0].imshow(rgb)
           m
```

