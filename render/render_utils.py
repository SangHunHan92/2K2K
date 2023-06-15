import numpy as np
import trimesh
import collections
from PIL import Image
import torch
import cv2
from trimesh.remesh import grouping
from scipy.spatial.transform import Rotation as R
from pyrender import PerspectiveCamera, OrthographicCamera, SpotLight, Mesh, Scene, OffscreenRenderer

def rotate_mesh(vertices, angle, global_angle, faces, vertex_colors, axis='x'):
    vertices_re = (np.zeros_like(vertices))
    if axis == 'y':  # pitch
        rotation_axis = np.array([1, 0, 0])
    elif axis == 'x':  # yaw
        rotation_axis = np.array([0, 1, 0])
    elif axis == 'z':  # roll
        rotation_axis = np.array([0, 0, 1])
    else:  # default is x (yaw)
        rotation_axis = np.array([0, 1, 0])

    for i in range(vertices.shape[0]):
        vec = vertices[i, :]
        rotation_degrees = angle
        rotation_radians = np.radians(rotation_degrees)

        rotation_vector = rotation_radians * rotation_axis
        rotation_vector -= global_angle
        rotation = R.from_rotvec(rotation_vector)
        rotated_vec = rotation.apply(vec)
        vertices_re[i, :] = rotated_vec

    mesh = trimesh.Trimesh(vertices=vertices_re, faces=faces, vertex_colors=vertex_colors)

    return mesh

def load_obj_mesh(mesh_file, tex_file):
    vertex_data = []
    norm_data = []
    uv_data = []
    dict = collections.defaultdict(int)

    face_data = []
    face_norm_data = []
    face_uv_data = []

    if isinstance(mesh_file, str):
        f = open(mesh_file, "r")
    else:
        f = mesh_file

    for line in f:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith('#'):
            continue
        values = line.split()
        if not values:
            continue
        if values[0] == 'v':
            v = list(map(float, values[1:4]))
            vertex_data.append(v)
        elif values[0] == 'vn':
            vn = list(map(float, values[1:4]))
            norm_data.append(vn)
        elif values[0] == 'vt':
            vt = list(map(float, values[1:3]))
            uv_data.append(vt)
        elif values[0] == 'f':
            # quad mesh
            if len(values) > 4:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)
                f = list(map(lambda x: int(x.split('/')[0]), [values[3], values[4], values[1]]))
                face_data.append(f)
            # tri mesh
            else:
                f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                face_data.append(f)

            # deal with texture
            if len(values[1].split('/')) >= 2:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[1]), [values[3], values[4], values[1]]))
                    face_uv_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[1]) != 0:
                    f_c = list(map(lambda x: int(x.split('/')[1]), values[1:4]))
                    face_uv_data.append(f_c)
                    f = list(map(lambda x: int(x.split('/')[0]), values[1:4]))
                    dict[f[0] - 1] = f_c[0] - 1
                    dict[f[1] - 1] = f_c[1] - 1
                    dict[f[2] - 1] = f_c[2] - 1
                else:
                    face_uv_data.append([1, 1, 1])

            # deal with normal
            if len(values[1].split('/')) == 3:
                # quad mesh
                if len(values) > 4:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)
                    f = list(map(lambda x: int(x.split('/')[2]), [values[3], values[4], values[1]]))
                    face_norm_data.append(f)
                # tri mesh
                elif len(values[1].split('/')[2]) != 0:
                    f = list(map(lambda x: int(x.split('/')[2]), values[1:4]))
                    face_norm_data.append(f)

    vertex_colors = []
    for k in range(len(vertex_data)):
        if k in dict:
            vertex_colors.append(uv_data[dict[k]])
        else:
            vertex_colors.append([0.0, 0.0])

    texture = Image.open(tex_file)
    w, h = texture.size[0], texture.size[1]

    vertices = np.array (vertex_data)
    visuals = np.array (vertex_colors)
    faces = np.array (face_data) - 1
    face_uvs = np.array (face_uv_data) - 1

    vertices, faces, mid_uvs = subdivide (vertices, uv_data, faces, face_uvs)
    uvs = np.vstack ((visuals, mid_uvs))

    vertex_colors = uvs
    texture = np.flip(texture, 0)
    vertex_colors = [[int(item[0]*w), int(item[1]*h)] for item in vertex_colors]
    vertex_colors = [texture[item[1], item[0], :] for item in vertex_colors]
    visuals = np.array (vertex_colors)
    mesh = trimesh.Trimesh (vertices=vertices, faces=faces, vertex_colors=visuals, process=True)


    # if with_texture and with_normal:
    #     uvs = np.array(uv_data)
    #     face_uvs = np.array(face_uv_data) - 1
    #     norms = np.array(norm_data)
    #     if norms.shape[0] == 0:
    #         norms = compute_normal(vertices, faces)
    #         face_normals = faces
    #     else:
    #         norms = normalize_v3(norms)
    #         face_normals = np.array(face_norm_data) - 1
    #     return vertices, visuals, faces, norms, face_normals, uvs, face_uvs, mesh
    #
    # if with_texture:
    #     uvs = np.array(uv_data)
    #     face_uvs = np.array(face_uv_data) - 1
    #     return vertices, faces, uvs, face_uvs
    #
    # if with_normal:
    #     norms = np.array(norm_data)
    #     norms = normalize_v3(norms)
    #     face_normals = np.array(face_norm_data) - 1
    #     return vertices, faces, norms, face_normals

    return mesh

def pers_get_depth_maps(mesh, scene, res, fov, item_name):
    # scene.camera.resolution = [res, res] #[1024, 1024]
    # scene.camera.fov = fov * (scene.camera.resolution /
    #                          scene.camera.resolution.max()) #[50, 50]
    # scene.camera_transform[0:3, 3] = 0.0
    # scene.camera_transform[2, 3] = 1.0
    scene.set_camera(angles=(0,0,0), center=(0,0,0), distance=1, resolution=[res, res], fov=[50., 50.])
    pers_origins, pers_vectors, pers_pixels = scene.camera_rays()
    pers_points, pers_index_ray, pers_index_tri = mesh.ray.intersects_location(
        pers_origins, pers_vectors, multiple_hits=True)
    pers_depth = trimesh.util.diagonal_dot(pers_points - pers_origins[0],
                                           pers_vectors[pers_index_ray])
    pers_colors = mesh.visual.face_colors[pers_index_tri]

    pers_pixel_ray = pers_pixels[pers_index_ray]
    pers_depth_far = np.zeros(scene.camera.resolution, dtype=np.float32) # 0
    pers_color_far = np.zeros((res, res, 3), dtype=np.float32)

    pers_depth_near = np.ones(scene.camera.resolution, dtype=np.float32) * res # 1
    pers_color_near = np.zeros((res, res, 3), dtype=np.float32)

    denom = np.tan(np.radians(fov) / 2.0) * 2
    pers_depth_int = (pers_depth - 1)*(res/denom) + res / 2

    for k in range(pers_pixel_ray.shape[0]):
        u, v = pers_pixel_ray[k, 0], pers_pixel_ray[k, 1]
        if pers_depth_int[k] > pers_depth_far[v, u]: # > 0
            # pers_color_far[v, u, ::-1] = pers_colors[k, 0:4] / 255.0
            pers_color_far[v, u, ::-1] = pers_colors[k, 0:3] / 255.0
            pers_depth_far[v, u] = pers_depth_int[k]
        if pers_depth_int[k] < pers_depth_near[v, u]: # < 1
            pers_depth_near[v, u] = pers_depth_int[k]
            # pers_color_near[v, u, ::-1] = pers_colors[k, 0:4] / 255.0
            pers_color_near[v, u, ::-1] = pers_colors[k, 0:3] / 255.0

    pers_depth_near = pers_depth_near * (pers_depth_near != res)
    pers_color_near = np.flip(pers_color_near, 0)
    pers_depth_near = np.flip(pers_depth_near, 0)
    pers_color_far = np.flip(pers_color_far, 0)
    pers_depth_far = np.flip(pers_depth_far, 0)

    return pers_color_near, pers_depth_near, pers_depth_far, pers_color_far

def get_depth_maps(mesh, scene, res, fov, dir, item_name):
    # scene.camera.resolution = [res, res]
    # scene.camera.fov = fov * (scene.camera.resolution /
    #                           scene.camera.resolution.max())
    # scene.camera_transform[0:3, 3] = 0.0
    # scene.camera_transform[2, 3] = 1.0
    # scene2 = mesh.scene()
    scene.set_camera(angles=(0,0,0), center=(0,0,0), distance=1, resolution=[res, res], fov=[50., 50.])
    origins, vectors, pixels = get_camera_rays(scene.camera, dir)

    points, index_ray, index_tri = mesh.ray.intersects_location(
        origins, vectors, multiple_hits=True)
    depth = trimesh.util.diagonal_dot(points - origins[index_ray], vectors[index_ray])
    colors = mesh.visual.face_colors[index_tri]

    pixel_ray = pixels[index_ray]
    depth_far = np.zeros(scene.camera.resolution, dtype=np.float32)
    depth_near = np.ones(scene.camera.resolution, dtype=np.float32) * res
    color_far = np.zeros((res, res, 3), dtype=np.float32)
    color_near = np.zeros((res, res, 3), dtype=np.float32)

    denom = np.tan(np.radians(fov) / 2.0) * 2
    depth_int = (depth - 1)*(res/denom) + res / 2

    for k in range(pixel_ray.shape[0]):
        u, v = pixel_ray[k, 0], pixel_ray[k, 1]
        if depth_int[k] > depth_far[v, u]:
            color_far[v, u, ::-1] = colors[k, 0:3] / 255.0
            depth_far[v, u] = depth_int[k]
        if depth_int[k] < depth_near[v, u]:
            depth_near[v, u] = depth_int[k]
            color_near[v, u, ::-1] = colors[k, 0:3] / 255.0

    depth_near = depth_near * (depth_near != res)

    if dir == 'side':
        depth_near = np.rot90(depth_near, k=1)
        depth_far = np.rot90(depth_far, k=1)
        color_near = np.rot90(color_near, k=1)
        color_far = np.rot90(color_far, k=1)
    return depth_near, depth_far, color_near, color_far

def get_camera_rays(camera, dir):
    res = camera.resolution[0]
    v = np.tan(np.radians(camera.fov[0]) / 2.0)
    # v *= 1 - (1/res)

    # create a grid of vectors
    if dir == 'front':
        xy = grid_linspace(
            bounds=[[-v, -v], [v, v]],
            count=res)
        pixels = grid_linspace(
            bounds=[[0, res - 1], [res - 1, 0]],
            count=res).astype(np.int64)
        vectors = np.column_stack((np.zeros_like(xy[:, :]), -np.ones_like(xy[:, :1])))
        origins = np.column_stack((xy, np.ones_like(xy[:, :1])))

    elif dir == 'side':
        yz = grid_linspace(
            bounds=[[-v, v], [v, -v]],
            count=res)
        pixels = grid_linspace(
            bounds=[[0, res - 1], [res - 1, 0]],
            count=res).astype(np.int64)
        vectors = np.column_stack((np.ones_like(yz[:, :1]), np.zeros_like(yz[:, :])))
        origins = np.column_stack((-np.ones_like(yz[:, :1]), yz))

    elif dir == 'up':  # will be updated.
        xz = grid_linspace(
            bounds=[[-v, -v], [v, v]],
            count=res)
        pixels = grid_linspace(
            bounds=[[0, res - 1], [0, res - 1]],
            count=res).astype(np.int64)
        vectors = np.column_stack((np.zeros_like(xz[:, :1]),
                                   np.ones_like(xz[:, :1]),
                                   np.zeros_like(xz[:, :1])))
        origins = np.column_stack((xz[:, 0], -np.ones_like(xz[:, :1]), xz[:, 1]))

    return origins, vectors, pixels

def grid_linspace(bounds, count):
    """
    Return a grid spaced inside a bounding box with edges spaced using np.linspace.

    Parameters
    ------------
    bounds: (2,dimension) list of [[min x, min y, etc], [max x, max y, etc]]
    count:  int, or (dimension,) int, number of samples per side

    Returns
    ---------
    grid: (n, dimension) float, points in the specified bounds
    """
    bounds = np.asanyarray(bounds, dtype=np.float64)
    if len(bounds) != 2:
        raise ValueError('bounds must be (2, dimension!')

    count = np.asanyarray(count, dtype=np.int64)
    if count.shape == ():
        count = np.tile(count, bounds.shape[1])

    grid_elements = [np.linspace(*b, num=c) for b, c in zip(bounds.T, count)]
    grid = np.vstack(np.meshgrid(*grid_elements, indexing='ij')
                     ).reshape(bounds.shape[1], -1).T
    return grid

def lights(cam_scene, angle):
    intensity_1 = np.random.randint(8, 10, size=1)
    # intensity_2 = np.random.randint(2, 3, size=1)
    random_channel = np.random.randint(1, 2, size=2)
    random_color = np.random.uniform(0.7, 0.8, [2, ])

    spot_color = np.ones(3)
    point_color = np.ones(3)

    spot_color[random_channel[1]] = random_color[1]
    spot_color[random_channel[0]] = random_color[0]

    point_color[random_channel[0]] = random_color[0]
    point_color[random_channel[1]] = random_color[1]

    spot_light = SpotLight(color=spot_color, intensity=intensity_1,
                   innerConeAngle=np.pi/10, outerConeAngle=np.pi/6)
    # # point_light = PointLight(color=np.ones(3), intensity=intensity_2)
    # point_light = DirectionalLight(color=point_color, intensity=intensity_2)

    light_pose1 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    light_pose2 = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 2.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    # add light sources.
    light_dir1 = np.random.randint(0, 360, size=360)
    light_dir2 = np.random.randint(0, 360, size=360)

    # light_dir1 = [0, 180, -90, 90]
    # light_dir2 = [-180, -165, -140, -125, -110, -95, -80, -65, -50, -35, -20, -5,
    #               5, 20, 35, 50, 65, 80, 95, 110, 125, 140, 165, 180]
    light1 = []
    light2 = []
    for idx in range(len(light_dir1)):
        light1.append(np.linalg.multi_dot([rotationy(light_dir1[idx]), light_pose1]))
    for idx in range(len(light_dir2)):
        light2.append(np.linalg.multi_dot([rotationy(light_dir2[idx]), light_pose2]))

    # for light_idx in range(len(light1)):
    #     cam_scene.add(point_light, pose=light1[light_idx])
    for light_idx in range(len(light2)):
        cam_scene.add(spot_light, pose=light2[light_idx])

    return cam_scene

def add_lights(mesh, cam_res, angle, scene, fov):
    scene.camera.resolution = [cam_res, cam_res]
    scene.camera.fov = [fov, fov] * (scene.camera.resolution /
                             scene.camera.resolution.max())
    mesh.scene = scene
    points_mesh = Mesh.from_trimesh(mesh, smooth=False)
    camera = OrthographicCamera(xmag=0.47, ymag=0.47)
    camera_pose = scene.camera_transform

    camera_dir = np.asarray([0])  # front
    cam_pose = []
    cam_scene = []
    colors = []

    r = OffscreenRenderer(cam_res, cam_res)

    for idx in range(camera_dir.shape[0]):
        cam_pose.append(np.linalg.multi_dot([rotationy(camera_dir[idx]), camera_pose]))
        cam_scene.append(Scene(bg_color=[0.0, 0.0, 0.0]))
        cam_scene[idx].add(points_mesh)
        cam_scene[idx].add(camera, pose=cam_pose[idx])
        cam_scene[idx] = lights(cam_scene[idx], angle)

        color, depth = r.render(cam_scene[idx])
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)

        if idx == 1 or idx == 3:
            color = np.flip(color, 1)
        colors.append(color)

        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(color)
        # plt.show()

    return colors[0]

def pers_add_lights(mesh, cam_res, angle, scene, fov):
    scene.camera.resolution = [cam_res, cam_res]
    scene.camera.fov = fov * (scene.camera.resolution /
                             scene.camera.resolution.max())
    mesh.scene = scene
    points_mesh = Mesh.from_trimesh(mesh, smooth=False)
    pers_camera = PerspectiveCamera(yfov=np.pi/3.6, aspectRatio=1.0)
    camera_pose = scene.camera_transform

    camera_dir = np.asarray([0])  # front
    cam_pose = []
    pers_cam_scene = []
    pers_colors = []

    for idx in range(camera_dir.shape[0]):
        cam_pose.append(np.linalg.multi_dot([rotationy(camera_dir[idx]), camera_pose]))
        pers_cam_scene.append(Scene(ambient_light=[0.01, 0.01, 0.1], bg_color=[0.0, 0.0, 0.0]))
        pers_cam_scene[idx].add(points_mesh)
        pers_cam_scene[idx].add(pers_camera, pose=cam_pose[idx])
        pers_cam_scene[idx] = lights(pers_cam_scene[idx], angle)

        pers_r = OffscreenRenderer(cam_res, cam_res)
        pers_color, pers_depth = pers_r.render(pers_cam_scene[idx])

        pers_color = cv2.cvtColor(pers_color, cv2.COLOR_BGR2RGB)

        if idx == 1 or idx == 3:
            pers_color = np.flip(pers_color, 1)
        pers_colors.append(pers_color)

    return pers_colors[0]

def subdivide(vertices, uv, faces, face_uvs, face_index=None):
    def faces_to_edges(faces, return_index=False):
        faces = np.asanyarray (faces)

        # each face has three edges
        edges = faces[:, [0, 1, 1, 2, 2, 0]].reshape ((-1, 2))

        if return_index:
            # edges are in order of faces due to reshape
            face_index = np.tile (np.arange (len (faces)), (3, 1)).T.reshape (-1)
            return edges, face_index
        return edges

    if face_index is None:
        face_index = np.arange(len(faces))
    else:
        face_index = np.asanyarray(face_index)

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_index]

    # find the unique edges of our faces subset
    edges = np.sort(faces_to_edges(faces_subset), axis=1)
    uv_edges = np.sort(faces_to_edges(face_uvs), axis=1)

    dict = collections.defaultdict(tuple)
    for k in range(len(edges)):
        dict[tuple(edges[k])] = uv_edges[k]

    unique, inverse = grouping.unique_rows(edges)
    # then only produce one midpoint per unique edge
    mid = vertices[edges[unique]].mean(axis=1)
    mid_idx = inverse.reshape((-1, 3)) + len(vertices)

    mid_uvs = [dict[tuple(i)] for i in edges[unique]]
    mid_uvs = [[(uv[i[0]][0]+uv[i[1]][0])/2, (uv[i[0]][1]+uv[i[1]][1])/2] for i in mid_uvs]

    # the new faces_subset with correct winding
    f = np.column_stack([faces_subset[:, 0],
                         mid_idx[:, 0],
                         mid_idx[:, 2],
                         mid_idx[:, 0],
                         faces_subset[:, 1],
                         mid_idx[:, 1],
                         mid_idx[:, 2],
                         mid_idx[:, 1],
                         faces_subset[:, 2],
                         mid_idx[:, 0],
                         mid_idx[:, 1],
                         mid_idx[:, 2]]).reshape((-1, 3))
    # add the 3 new faces_subset per old face
    new_faces = np.vstack((faces, f[len(face_index):]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[:len(face_index)]

    new_vertices = np.vstack((vertices, mid))
    # new_uv = np.vstack((uv, mid_uvs))

    return new_vertices, new_faces, mid_uvs

def param_to_tensor(param, device):
    for key in param.keys():
        try :
            # whatever numpy or tensor, working well
            param[key] = torch.reshape(torch.as_tensor(param[key], device=device), (1, -1))
            param[key].requires_grad=True
        except :
            continue
    return param

def rotationx(theta):
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, np.cos(theta / 180 * np.pi), np.sin(theta / 180 * np.pi), 0.0],
        [0.0, -np.sin(theta / 180 * np.pi), np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def rotationy(theta):
    return np.array([
        [np.cos(theta / 180 * np.pi), 0.0, np.sin(theta / 180 * np.pi), 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-np.sin(theta / 180 * np.pi), 0.0, np.cos(theta / 180 * np.pi), 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

def location(x, y, z):
    return np.array([
        [1.0, 0.0, 0.0, x],
        [0.0, 1.0, 0.0, y],
        [0.0, 0.0, 1.0, z],
        [0.0, 0.0, 0.0, 1.0]
    ])
