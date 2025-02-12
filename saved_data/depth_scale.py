import pyrealsense2 as rs

# Start the pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth)

# Start the pipeline with configuration
pipeline.start(config)

# Get the depth sensor
depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()

# Retrieve the depth scale
depth_scale = depth_sensor.get_depth_scale()
print("Depth Scale (meters per unit):", depth_scale)

# Stop the pipeline
pipeline.stop()

