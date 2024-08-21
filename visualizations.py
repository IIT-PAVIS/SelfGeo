import os
import open3d as o3d
import seaborn as sns
import colorcet as cc

def pc_to_pcd(pc):
    palette_PC = sns.color_palette()
    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
    pcd.translate(pc[0])
    pcd.paint_uniform_color(palette_PC[7])

    ''' Add points in the original point cloud'''
    for i in range(len(pc)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)  ## 0.005
        point.translate(pc[i])
        point.paint_uniform_color(palette_PC[7])
        pcd += point

    return pcd


def kp_to_pcd(kp):
    palette = sns.color_palette("bright")
    palette_dark = sns.color_palette("dark")
    pcd = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)
    pcd.translate(kp[0])
    pcd.paint_uniform_color(palette[0])

    for i in range(1, len(kp)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.035)  # ablation: 0.035, figures: 0.050
        point.translate(kp[i])
        if i == 7:
            point.paint_uniform_color(palette_dark[7])
        else:
            point.paint_uniform_color(palette[i])
        pcd += point
    return pcd


def save_keypoints(pc, kp, folder="visualizations", name="", save=True):
    '''
        Parameters
        ----------
        pc          Input point cloud  [2048, 3]
        kp          Estimated key-points  [10, 3]
        folder      Directory to save the images
        name        Name of the output image
        save        if true, the image will save in "output_dir", otherwise, the output will display on screen

        Returns     show the key-points/point cloud
        -------
    '''

    # pdb.set_trace()

    palette_PC = sns.color_palette("dark", 25)
#    palette = sns.color_palette("bright", 25)
    palette = sns.color_palette(cc.glasbey_light, n_colors=13)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc)
    # palette = sns.color_palette("bright", 25)  # create color palette
    pcd1.paint_uniform_color(palette_PC[7])

    key_points = []
    for i in range(len(kp)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        point.translate(kp[i])
        point.paint_uniform_color(palette[i])
        key_points.append(point)

    if save:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd1)
        for i in range(len(key_points)):
            vis.add_geometry(key_points[i])
        vis.poll_events()
        vis.update_renderer()

        if not os.path.exists(folder):
            os.makedirs(folder)
        vis.capture_screen_image("{}/{}.png".format(folder, name))
        vis.destroy_window()
    else:
        o3d.visualization.draw_geometries([pcd1, *key_points])

def save_two_keypoints(pc, kp, kp2, folder="visualizations_pck", name="", save=True):
    '''
        Parameters
        ----------
        pc          Input point cloud  [2048, 3]
        kp          Estimated key-points  [10, 3]
        kp2         Reference key-points  [10, 3]
        folder      Directory to save the images
        name        Name of the output image
        save        if true, the image will save in "output_dir", otherwise, the output will display on screen

        Returns     show the key-points/point cloud
        -------
    '''


    palette_pc = sns.color_palette("dark")
    palette = sns.color_palette(cc.glasbey_light, n_colors=13)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pc)
    pcd1.paint_uniform_color(palette_pc[7])

    key_points = []
    for i in range(len(kp)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # 0.02
        point.translate(kp[i])
        point.paint_uniform_color(palette[i])
        key_points.append(point)

    key_points2 = []
    for i in range(len(kp2)):
        point = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)  # 0.02
        point.translate(kp2[i])
        point.paint_uniform_color(palette[i])
        key_points2.append(point)

    if save:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd1)
        for i in range(len(key_points)):
            vis.add_geometry(key_points[i])
        for i in range(len(key_points2)):
            vis.add_geometry(key_points2[i])
        vis.poll_events()
        vis.update_renderer()

        ''' Added these lines for DeformingThings4D (Animal) dataset '''
        out_dir = '{}/{}'.format(folder, name.split('_')[-2] + '_' + name.split('_')[-1])
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        vis.capture_screen_image("{}/{}.png".format(out_dir, name))

        vis.destroy_window()

    else:
        o3d.visualization.draw_geometries([pcd1, *key_points, *key_points2])
