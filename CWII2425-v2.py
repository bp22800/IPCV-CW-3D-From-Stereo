'''
Department of Computer Science, University of Bristol
COMS30030: Image Processing and Computer Vision

3-D from Stereo: Coursework Part 2
3-D simulator

Yuhang Ming yuhang.ming@bristol.ac.uk
Andrew Calway andrew@cs.bris.ac.uk
'''

import cv2
import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import argparse


'''
Interaction menu:
P  : Take a screen capture.
D  : Take a depth capture.

Official doc on visualisation interactions:
http://www.open3d.org/docs/latest/tutorial/Basic/visualization.html
'''

def transform_points(points, H):
    '''
    transform list of 3-D points using 4x4 coordinate transformation matrix H
    converts points to homogeneous coordinates prior to matrix multiplication
    
    input:
      points: Nx3 matrix with each row being a 3-D point
      H: 4x4 transformation matrix
    
    return:
      new_points: Nx3 matrix with each row being a 3-D point
    '''
    # compute pt_w = H * pt_c
    n,m = points.shape
    if m == 4:
        new_points = points
    else:
        new_points = np.concatenate([points, np.ones((n,1))], axis=1)
    new_points = H.dot(new_points.transpose())
    new_points = new_points / new_points[3,:]
    new_points = new_points[:3,:].transpose()
    return new_points

def check_dup_locations(y, z, loc_list):
    for (loc_y, loc_z) in loc_list:
        if loc_y == y and loc_z == z:
            return True


# print("here", flush=True)
if __name__ == '__main__': 

    ####################################
    ### Take command line arguments ####
    ####################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--num', dest='num', type=int, default=6, 
                        help='number of spheres')    
    parser.add_argument('--sph_rad_min', dest='sph_rad_min', type=int, default=10, 
                        help='min sphere  radius x10')
    parser.add_argument('--sph_rad_max', dest='sph_rad_max', type=int, default=16, 
                        help='max sphere  radius x10')
    parser.add_argument('--sph_sep_min', dest='sph_sep_min', type=int, default=4, 
                       help='min sphere  separation')
    parser.add_argument('--sph_sep_max', dest='sph_sep_max', type=int, default=8, 
                       help='max sphere  separation')
    parser.add_argument('--display_centre', dest='bCentre', action='store_true',
                        help='open up another visualiser to visualise centres')
    parser.add_argument('--coords', dest='bCoords', action='store_true')

    args = parser.parse_args()

    if args.num<=0:
        print('invalidnumber of spheres')
        exit()

    if args.sph_rad_min>=args.sph_rad_max or args.sph_rad_min<=0:
        print('invalid max and min sphere radii')
        exit()
    	
    if args.sph_sep_min>=args.sph_sep_max or args.sph_sep_min<=0:
        print('invalid max and min sphere separation')
        exit()
	
    ####################################
    #### Setup objects in the scene ####
    ####################################

    # create plane to hold all spheres
    h, w = 24, 12
    # place the support plane on the x-z plane
    box_mesh=o3d.geometry.TriangleMesh.create_box(width=h,height=0.05,depth=w)
    box_H=np.array(
                 [[1, 0, 0, -h/2],
                  [0, 1, 0, -0.05],
                  [0, 0, 1, -w/2],
                  [0, 0, 0, 1]]
                )
    box_rgb = [0.7, 0.7, 0.7]
    name_list = ['plane']
    mesh_list, H_list, RGB_list = [box_mesh], [box_H], [box_rgb]

    # create spheres
    prev_loc = []
    GT_cents, GT_rads = [], []
    for i in range(args.num):
        # add sphere name
        name_list.append(f'sphere_{i}')

        # create sphere with random radius
        size = random.randrange(args.sph_rad_min, args.sph_rad_max, 2)/10
        sph_mesh=o3d.geometry.TriangleMesh.create_sphere(radius=size)
        mesh_list.append(sph_mesh)
        RGB_list.append([0., 0.5, 0.5])

        # create random sphere location
        step = random.randrange(int(args.sph_sep_min),int(args.sph_sep_max),1)
        x = random.randrange(int(-h/2+2), int(h/2-2), step)
        z = random.randrange(int(-w/2+2), int(w/2-2), step)
        while check_dup_locations(x, z, prev_loc):
            x = random.randrange(int(-h/2+2), int(h/2-2), step)
            z = random.randrange(int(-w/2+2), int(w/2-2), step)
        prev_loc.append((x, z))

        GT_cents.append(np.array([x, size, z, 1.]))
        GT_rads.append(size)
        sph_H = np.array(
                    [[1, 0, 0, x],
                     [0, 1, 0, size],
                     [0, 0, 1, z],
                     [0, 0, 0, 1]]
                )
        H_list.append(sph_H)

    # arrange plane and sphere in the space
    obj_meshes = []
    for (mesh, H, rgb) in zip(mesh_list, H_list, RGB_list):
        # apply location
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )
        # paint meshes in uniform colours here
        mesh.paint_uniform_color(rgb)
        mesh.compute_vertex_normals()
        obj_meshes.append(mesh)

    # add optional coordinate system
    if args.bCoords:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1., origin=[0, 0, 0])
        obj_meshes = obj_meshes+[coord_frame]
        RGB_list.append([1., 1., 1.])
        name_list.append('coords')


    ###################################
    #### Setup camera orientations ####
    ###################################

    # set camera pose (world to camera)
    # # camera init 
    # # placed at the world origin, and looking at z-positive direction, 
    # # x-positive to right, y-positive to down
    # H_init = np.eye(4)      
    # print(H_init)

    # camera_0 (world to camera)
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H0_wc = np.array(
                [[1,            0,              0,  0],
                [0, np.cos(theta), -np.sin(theta),  0], 
                [0, np.sin(theta),  np.cos(theta), 20], 
                [0, 0, 0, 1]]
            )

    # camera_1 (world to camera)
    theta = np.pi * (80+random.uniform(-10, 10))/180.
    H1_0 = np.array(
                [[np.cos(theta),  0, np.sin(theta), 0],
                 [0,              1, 0,             0],
                 [-np.sin(theta), 0, np.cos(theta), 0],
                 [0, 0, 0, 1]]
            )
    theta = np.pi * (45*5+random.uniform(-5, 5))/180.
    H1_1 = np.array(
                [[1, 0,            0,              0],
                [0, np.cos(theta), -np.sin(theta), -4],
                [0, np.sin(theta), np.cos(theta),  20],
                [0, 0, 0, 1]]
            )
    H1_wc = np.matmul(H1_1, H1_0)
    render_list = [(H0_wc, 'view0.png', 'depth0.png'), 
                   (H1_wc, 'view1.png', 'depth1.png')]

#####################################################
    # NOTE: This section relates to rendering scenes in Open3D, details are not
    # critical to understanding the lab, but feel free to read Open3D docs
    # to understand how it works.
    
    # set up camera intrinsic matrix needed for rendering in Open3D
    img_width=640
    img_height=480
    f=415 # focal length
    # image centre in pixel coordinates
    ox=img_width/2-0.5 
    oy=img_height/2-0.5
    K = o3d.camera.PinholeCameraIntrinsic(img_width,img_height,f,f,ox,oy)

    # Rendering RGB-D frames given camera poses
    # create visualiser and get rendered views
    cam = o3d.camera.PinholeCameraParameters()
    cam.intrinsic = K
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=img_width, height=img_height, left=0, top=0)
    for m in obj_meshes:
        vis.add_geometry(m)
    ctr = vis.get_view_control()
    for (H_wc, name, dname) in render_list:
        cam.extrinsic = H_wc
        ctr.convert_from_pinhole_camera_parameters(cam,True)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(name, True)
        vis.capture_depth_image(dname, True)
    vis.run()
    vis.destroy_window()
##################################################

    # load in the images for post processings
    img0 = cv2.imread('view0.png', -1)
    dep0 = cv2.imread('depth0.png', -1)
    img1 = cv2.imread('view1.png', -1)
    dep1 = cv2.imread('depth1.png', -1)

    # visualise sphere centres
    pcd_GTcents = o3d.geometry.PointCloud()
    pcd_GTcents.points = o3d.utility.Vector3dVector(np.array(GT_cents)[:, :3])
    pcd_GTcents.paint_uniform_color([1., 0., 0.])
    if args.bCentre:
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=640, height=480, left=0, top=0)
        for m in [obj_meshes[0], pcd_GTcents]:
            vis.add_geometry(m)
        vis.run()
        vis.destroy_window()

    
    ###################################
    '''
    Task 3: Circle detection
    Hint: use cv2.HoughCircles() for circle detection.
    https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html#ga47849c3be0d0406ad3ca45db65a25d2d

    Write your code here
    '''
    ###################################

    filenames = ['view0.png', 'view1.png']

    image_circle_lookup = {}

    # Detect circles for each image specified in filenames
    for filename in filenames:
        # Load image and convert to grey
        image = cv2.imread(filename, 1)
        image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Blur the image
        image_grey = cv2.GaussianBlur(image_grey, (7, 7), 1.5, 1.5)

        # Detect circles
        min_dist = 10
        min_radius = 5
        max_radius = 100
        # circles = cv2.HoughCircles(image_grey, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dist, param1=50, param2=40, minRadius=min_radius, maxRadius=max_radius)[0,:]
        circles = cv2.HoughCircles(image_grey, cv2.HOUGH_GRADIENT_ALT, dp=1.5, minDist=min_dist, param1=300, param2=0.9, minRadius=min_radius, maxRadius=max_radius)[0,:]

        # Store circles for this image for later use
        image_circle_lookup[filename] = circles

        # Draw circles
        for circle in circles:
            x0 = round(circle[0])
            y0 = round(circle[1])
            r  = round(circle[2])
            colour = (255, 0, 0)
            thickness = 2
            point_size = 3
            cv2.circle(image, (x0, y0), point_size, colour, -1) # centre point
            cv2.circle(image, (x0, y0), r, colour, thickness) # circle
        
        # Save image
        cv2.imwrite(filename, image)

    ###################################
    '''
    Task 4: Epipolar line
    Hint: Compute Essential & Fundamental Matrix
          Draw lines with cv2.line() function
    https://docs.opencv.org/4.x/d6/d6e/group__imgproc__draw.html#ga7078a9fae8c7e7d13d24dac2520ae4a2
    
    Write your code here
    '''
    ###################################

    # Reference VC -> camera_0 -> view0.png
    # Viewing VC   -> camera_1 -> view1.png

    # Get circles for reference VC
    reference_circles = image_circle_lookup['view0.png']

    # Get focal length of both cameras
    f = K.get_focal_length()[0]

    # Get M_L and M_R matrices
    M_L = np.linalg.inv(K.intrinsic_matrix)
    M_L[0,0] *= f
    M_L[1,1] *= f
    M_R = M_L

    # Get transformation from camera_0 to camera_1
    cam_0_to_1 = np.matmul(H1_wc, np.linalg.inv(H0_wc))

    # Get R and T
    R = cam_0_to_1[:-1,:-1]
    RT = -cam_0_to_1[:-1,3]
    T = np.matmul(np.linalg.inv(R), RT)

    # Construct S matrix
    S = np.array([
        [ 0,   -T[2], T[1]],
        [ T[2], 0,   -T[0]],
        [-T[1], T[0], 0]
    ])

    # Calculate essential matrix
    E = np.matmul(R, S)

    # Calculate fundamental matrix
    F = np.matmul(np.matmul(M_R.T, E), M_L)

    for circle in reference_circles:
        x0 = circle[0]
        y0 = circle[1]
        p_hat_L = np.array([x0, y0, f])

        Fp_hat_L = np.matmul(F, p_hat_L)

        line_start_x = 0
        line_end_x = K.width

        # Compute y given x using the epipolar line equation
        def get_y(x):
            return (-(x * Fp_hat_L[0]) - (f * Fp_hat_L[2])) / Fp_hat_L[1]

        line_start_y = get_y(line_start_x)
        line_end_y = get_y(line_end_x)

        line_start = (round(line_start_x), round(line_start_y))
        line_end = (round(line_end_x), round(line_end_y))

        image = cv2.imread('view1.png', 1) # read image
        image = cv2.line(image, line_start, line_end, (0,0,255), 1) # draw epipolar line onto image
        cv2.imwrite('view1.png', image) # write image with line

    ###################################
    '''
    Task 5: Find correspondences

    Write your code here
    '''
    ###################################

    # Method: For each circle centre in reference VC, calculate epipolar line constraint value p_R^T * F * p_L for this
    # with each circle centre in viewing VC, and the one that gives a value closest to 0 is the one to match

    viewing_circles = image_circle_lookup['view1.png']

    circle_matches = []

    for circle1 in reference_circles:
        best_circle = None
        best_value = float('inf')

        for circle2 in viewing_circles:
            p_hat_L = np.array([circle1[0], circle1[1], f])
            p_hat_R = np.array([circle2[0], circle2[1], f])
            value = np.matmul(p_hat_R, np.matmul(F, p_hat_L)) # epipolar line constraint value p_R^T * F * p_L

            if abs(value) < abs(best_value):
                best_value = value
                best_circle = circle2
        
        # Add coloured points to each image to show matchings
        reference_image = cv2.imread('view0.png', 1)
        viewing_image = cv2.imread('view1.png', 1)
        colour = (random.randrange(256), random.randrange(256), random.randrange(256))
        cv2.circle(reference_image, (round(circle1[0]), round(circle1[1])), 5, colour, -1)
        cv2.circle(viewing_image, (round(best_circle[0]), round(best_circle[1])), 5, colour, -1)
        cv2.imwrite('view0.png', reference_image)
        cv2.imwrite('view1.png', viewing_image)

        # Save matching for later use
        circle_matches.append((circle1, best_circle))

    ###################################
    '''
    Task 6: 3-D locations of sphere centres

    Write your code here
    '''
    ###################################

    show_on_image = True

    estimated_centres = []

    for match in circle_matches:
        circle1 = match[0]
        circle2 = match[1]

        # Get pixel coords of circle centres
        p_hat_L = np.array([circle1[0], circle1[1], f])
        p_hat_R = np.array([circle2[0], circle2[1], f])
        
        # Convert from pixel coords to image plane coords
        p_L = np.matmul(M_L, p_hat_L)
        p_R = np.matmul(M_R, p_hat_R)

        # Construct H matrix
        H = np.array([
            p_L,
            -np.matmul(R.T, p_R),
            -np.cross(p_L, np.matmul(R.T, p_R))
        ]).transpose()

        # Calculate values a, b and c
        a, b, c = np.matmul(np.linalg.inv(H), T)
        
        # Get the midpoint of the two vectors that represents the approximation of the 3D point relative to reference VC
        vec1 = a * p_L
        vec2 = b * np.matmul(R.T, p_R) + T
        P_hat = (vec1 + vec2) / 2

        # Get the 3D point relative to the world coordinate system
        H0_cw = np.linalg.inv(H0_wc)
        P_hat = np.append(P_hat, [1]) # convert into homogenous coordinate
        point = np.matmul(H0_cw, P_hat)[:-1]

        estimated_centres.append(point)
    
    if show_on_image:
        # Project 3D points onto reference VC and draw points onto image
        for point in estimated_centres:
            P_L = np.matmul(H0_wc, np.append(point, [1]))[:-1]
            Z = P_L[2]
            p_L = f * (P_L / Z)
            p_hat_L = np.matmul(np.linalg.inv(M_L), p_L)

            x0 = round(p_hat_L[0])
            y0 = round(p_hat_L[1])
            
            image = cv2.imread('view0.png', 1)
            colour = (0, 255, 0)
            thickness = 2
            cv2.circle(image, (x0, y0), 7, colour, thickness)
            cv2.imwrite('view0.png', image)
        
        # Project 3D points onto viewing VC and draw points onto image
        for point in estimated_centres:
            P_R = np.matmul(H1_wc, np.append(point, [1]))[:-1]
            Z = P_R[2]
            p_R = f * (P_R / Z)
            p_hat_R = np.matmul(np.linalg.inv(M_R), p_R)

            x0 = round(p_hat_R[0])
            y0 = round(p_hat_R[1])
            
            image = cv2.imread('view1.png', 1)
            colour = (0, 255, 0)
            thickness = 2
            cv2.circle(image, (x0, y0), 7, colour, thickness)
            cv2.imwrite('view1.png', image)

    ###################################
    '''
    Task 7: Evaluate and Display the centres

    Write your code here
    '''
    ###################################

    ground_truth_centres = np.array(GT_cents)[:,:3].tolist()

    # List of ground_truth_centres indices such that at index i containing index j, estimated_centres[i] corresponds with ground_truth_centres[j]
    ground_truth_order = []

    # Compute the errors (distances between estimated and ground truth centres)
    distance_sum = 0
    for est_centre in estimated_centres:
        # The closest ground truth centre is considered to correspond with the estimated centre
        smallest_distance = float('inf')
        closest_gt = None
        for gt_centre in ground_truth_centres:
            distance = math.sqrt(np.sum(np.square(est_centre - gt_centre)))
            if distance < smallest_distance:
                smallest_distance = distance
                closest_gt = gt_centre
        
        distance_sum += smallest_distance

        # Store ground truth order for later use with radii error
        ground_truth_index = ground_truth_centres.index(gt_centre)
        ground_truth_order.append(ground_truth_index)
        
        # Remove corresponding ground truth centre so it is not considered by any other estimated centres
        ground_truth_centres.remove(closest_gt)

        # Show the error
        print('Error (distance from ground truth) of estimated centre', est_centre, 'is:', smallest_distance)
    
    # Compute and show the mean error
    mean_distance = distance_sum / len(estimated_centres)
    print('Mean error:', mean_distance)


    ground_truth_centres = np.array(GT_cents)[:,:3].tolist()

    # Ground truth sphere centre point cloud (red)
    pcd_GT_cents = o3d.geometry.PointCloud()
    pcd_GT_cents.points = o3d.utility.Vector3dVector(np.array(ground_truth_centres))
    pcd_GT_cents.paint_uniform_color([1., 0., 0.])

    # Estimated sphere centre point cloud (green)
    pcd_est_cents = o3d.geometry.PointCloud()
    pcd_est_cents.points = o3d.utility.Vector3dVector(np.array(estimated_centres))
    pcd_est_cents.paint_uniform_color([0., 1., 0.])

    # Visualise both ground truth and estimated sphere centres
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, left=0, top=0)
    for m in [obj_meshes[0], pcd_GT_cents, pcd_est_cents]:
        vis.add_geometry(m)
    vis.run()
    vis.destroy_window()

    ###################################
    '''
    Task 8: 3-D radius of spheres

    Write your code here
    '''
    ###################################

    estimated_radii = []

    # For each circle detected in reference image, get corresponding 3D estimated point
    # and use perspective projection to calculate the radius of the sphere
    for i, circle in enumerate(reference_circles):
        point = estimated_centres[i]

        # Bring 3D point into reference VC's coordinate space
        point = np.append(point, [1])
        point = np.matmul(H0_wc, point)

        # Get pixel coords of sphere centre and pixel length of radius
        x0 = circle[0]
        y0 = circle[1]
        r = circle[2]

        # Get pixel coords of point on edge of circle vertically above centre in pixel image
        x = x0
        y = y0 - r
        p_hat_L = np.array([x, y, f])

        # Get image plane coords of point on edge of circle
        p_L = np.matmul(M_L, p_hat_L)

        # Get estimated radius of sphere
        y = p_L[1]
        Y = point[1]
        Z = point[2]
        estimated_radius = abs(((Z * y) / f) - Y)

        estimated_radii.append(estimated_radius)

    ###################################
    '''
    Task 9: Display the spheres

    Write your code here:
    '''
    ###################################

    print()

    # Compute the errors (differences between estimated and ground truth radii)
    error_sum = 0
    for i, est_radius in enumerate(estimated_radii):
        # Use ground truth that is already matched with this estimated sphere from the sphere centre error code above in Task 7
        gt_index = ground_truth_order[i]
        gt_radius = GT_rads[gt_index]
        
        difference = abs(gt_radius - est_radius)
        error_sum += difference

        est_centre = estimated_centres[i]

        # Show the error
        print('Error (difference from ground truth) of estimated radius for sphere centre', est_centre, 'is:', difference)
    
    # Compute and show the mean error
    mean_error = error_sum / len(estimated_radii)
    print('Mean error:', mean_error)


    # Note: The spheres are visualised below with a line set (basically a wireframe) instead of a mesh, so that we can see through the spheres
    # Note: Zoom closely to each sphere to see both the green and red spheres in full to compare them (from afar the wireframes can look like solid spheres)

    def getSphereLineSet(centre, radius, colour):
        # Create sphere with given radius
        mesh = o3d.geometry.TriangleMesh.create_sphere(radius=radius)

        # Create matrix H
        x = centre[0]
        y = centre[1]
        z = centre[2]
        H = np.array([
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1]
        ])

        # Arrange sphere in the space using H
        mesh.vertices = o3d.utility.Vector3dVector(
            transform_points(np.asarray(mesh.vertices), H)
        )

        # Compute vertex normals
        mesh.compute_vertex_normals()

        # Convert mesh into line set
        line_set = o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

        # Paint lines in uniform colours
        line_set.paint_uniform_color(colour)
        
        return line_set

    # Ground truth spheres (red)
    ground_truth_spheres = []
    ground_truth_centres = np.array(GT_cents)[:,:3].tolist()
    for i in range(len(ground_truth_centres)):
        gt_centre = ground_truth_centres[i]
        gt_radius = GT_rads[i]
        colour = [1.0, 0.0, 0.0]
        line_set = getSphereLineSet(gt_centre, gt_radius, colour)
        ground_truth_spheres.append(line_set)

    # Estimated spheres (green)
    estimated_spheres = []
    for i in range(len(estimated_centres)):
        est_centre = estimated_centres[i]
        est_radius = estimated_radii[i]
        colour = [0.0, 1.0, 0.0]
        line_set = getSphereLineSet(est_centre, est_radius, colour)
        estimated_spheres.append(line_set)

    # Visualise both ground truth and estimated spheres
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=640, height=480, left=0, top=0)
    vis.add_geometry(obj_meshes[0])
    for m in ground_truth_spheres:
        vis.add_geometry(m)
    for m in estimated_spheres:
        vis.add_geometry(m)
    vis.run()
    vis.destroy_window()

    ###################################
    '''
    Task 10: Evaluate performance using different sphere sizes and separations

    Write your code here:
    '''
    ###################################

    ...

    ###################################
    '''
    Task 11: Investigate impact of noise added to relative pose

    Write your code here:
    '''
    ###################################

    ...
