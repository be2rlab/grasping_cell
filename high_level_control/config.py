
# Setup
timeout = 1
# initial_position = [0.0, 0.5, 0.35] 
initial_position = [0.0, 0.543674, 0.458738] 
initial_orientation = [1, 0, 0, 0]
# Object recognition
initial_box_coords = [0, 0, 0.3] # in camera frame
conf_thresh = 0.55
dist_thresh = 55
max_retry = 3

change_position_shift = 0.05


# Learning process
plane_obj_coordinates = [0.5, 0.0, 0.05]
plane_obj_orientation = [0.7071068, -0.7071068, 0.0, 0.0]
learn_n_points = 5

# learning spiral curve settings
spiral_z_min = 0.1 + plane_obj_coordinates[2]
# spiral_z_max = 0.4 + plane_obj_coordinates[2]
spiral_z_max = 0.3 + plane_obj_coordinates[2]
max_radius = 0.25

# spiral_z_min = 0.26 + plane_obj_coordinates[2]
# spiral_z_max = 0.3 + plane_obj_coordinates[2]
# max_radius = 0.44

frec_mult = 50

# Planning
plan_n_grasps = 5


################## best spiral parameters

# plane_obj_coordinates = [0.5, 0.0, 0.1]
# plane_obj_orientation = [0.0, -0.7071068, 0.7071068, 0.0]
# learn_n_points = 10

# # learning spiral curve settings
# spiral_z_min = 0.1 + plane_obj_coordinates[2]
# spiral_z_max = 0.4 + plane_obj_coordinates[2]
# max_radius = 0.3