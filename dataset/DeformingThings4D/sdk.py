# .anime parsing code

'''

Loading the .anime files one by one from the DeformingThing4D dataset
Normalizing (-0.5 to +05) unit box and downsampliing to 2048 points
Creating geodesic distances from the downsampled PCDs

saving downsampled PCDs and corresponding geodesic distances to the output directory

input dir: datasets/DeformingThings4D/animals
output dir: datasets/DeformingThings4D/animal_pc_geodesics
        |_ pcd
        |_ geodesics
        

output dir: datasets/DeformingThings4D/correspondences_for_pck
        |_ */*.npz

        the .npz files contains PCD of frame 0 and the offsets
        Using the first PCD and offsets, we can create next frames (PCDs)


Mohammad Zohaib
ZohaibMohammad.GitHub.io
Zohaib.Mohammad@hotmail.com

November 02, 2023


'''


import os
import numpy as np
import trimesh
import glob as glob
from tqdm import tqdm
import pdb
import open3d as o3d
import seaborn as sns
from natsort import natsorted
from sklearn import neighbors
from sklearn.utils.graph import graph_shortest_path



def anime_read( filename):
    """
    filename: .anime file
    return:
        nf: number of frames in the animation
        nv: number of vertices in the mesh (mesh topology fixed through frames)
        nt: number of triangle face in the mesh
        vert_data: [nv, 3], vertice data of the 1st frame (3D positions in x-y-z-order)
        face_data: [nt, 3], riangle face data of the 1st frame
        offset_data: [nf-1,nv,3], 3D offset data from the 2nd to the last frame
    """
    f = open(filename, 'rb')
    nf = np.fromfile(f, dtype=np.int32, count=1)[0]
    nv = np.fromfile(f, dtype=np.int32, count=1)[0]
    nt = np.fromfile(f, dtype=np.int32, count=1)[0]
    vert_data = np.fromfile(f, dtype=np.float32, count=nv * 3)
    face_data = np.fromfile(f, dtype=np.int32, count=nt * 3)
    offset_data = np.fromfile(f, dtype=np.float32, count=-1)
    vert_data = vert_data.reshape((-1, 3))
    face_data = face_data.reshape((-1, 3))
    offset_data = offset_data.reshape((nf - 1, nv, 3))
    return nf, nv, nt, vert_data, face_data, offset_data


def visualize_pcd(pc):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([pcd])
    

def normalize_pc(pc):
    pc = pc - pc.mean(0)
    pc /= np.max(np.linalg.norm(pc, axis=-1)) # -1 to 1
    return pc/2  # # -0.5 to 0.5 (Unit box)
    

def farthest_point_sample_centroids(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
       
    point = point[centroids.astype(np.int32)]
    return point, centroids.astype(np.int32)


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
       
    point = point[centroids.astype(np.int32)]
    return point


def gen_geo_dists(pc):
    graph = neighbors.kneighbors_graph(pc, 20, mode='distance', include_self=False)
    return graph_shortest_path(graph, directed=False)



def visualize_gd_kp_pc_color(pc, gds, out_dir="", name="", save=False):
	'''
	pc:         input pc [2048x3]
	gds:        geodesics of the input pc [2048]

	we consider first batch and keypoint "kp_no"

	'''

	palette = sns.color_palette("bright")

	ind = 0
	keypoint = o3d.geometry.TriangleMesh.create_sphere(radius=0.008)
	keypoint.translate(pc[ind])             # estimted keypoint
	keypoint.paint_uniform_color(palette[ind])

	gd_kp = gds[ind,:]              # geodesic distance of the estimated kp w.r.t. other points in PCD
	gd_norm = (((max(gd_kp) - gd_kp ) / max(gd_kp)) *9).astype(int)     # normalize 0-1 and multiple by 10 (0-10)
	colors = []
	for x in gd_norm:
		colors.append(palette[x])

	colors=np.asarray(colors)

	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(pc)
	pcd.colors = o3d.utility.Vector3dVector(colors)


	if save:
		if not os.path.exists(out_dir):
			os.makedirs(out_dir)
		vis = o3d.visualization.Visualizer()
		vis.create_window()
		vis.add_geometry(pcd)
		vis.add_geometry(keypoint)
		vis.poll_events()
		vis.update_renderer()

		vis.capture_screen_image("{}/{}.png".format(out_dir, name))
		vis.destroy_window()
	else:
		o3d.visualization.draw_geometries([pcd, keypoint])


def create_pc_geodesics():
    path = "/media/r11-01/Transcend/datasets/DeformingThings4D/animals/bear3EP_Agression/bear3EP_Agression.anime"
    
    xt = np.asarray([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype='f')
    yt = np.asarray([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype='f')
    
    _,_,_,vert_data, face_data, frame = anime_read(path)
    mesh = trimesh.Trimesh(vert_data, face_data)  
    pc = normalize_pc(farthest_point_sample(mesh.sample(16000), 2048))
    pc = (yt[:3,:3]@ xt[:3,:3] @ pc.T).T
#   visualize_pcd(pc)

    geo_desics = gen_geo_dists(pc)


def read_npz_files():
    
    path = '/media/r11-01/mz/datasets/deformable_datasets/iccv23/DeformingThings4D-original/animal_pc_correspondences/bear/bear3EP_SwimleftRM.npz'
    file = np.load(path)
    pcd0 = file['pcd0']
    offsets = file['offsets']
    
    for offset in offsets:
        visualize_pcd(pcd0 + offset)



def create_pcds_and_correspondences():
    '''  To save the PC and correspondings   '''

    ''' 
        SET the paths 
        You may nned to generate the splits.txt
    '''
    # category = 'deerK5L'
    # category_lst = '/media/r11-01/Transcend/datasets/DeformingThings4D/processed_pcd_geodesics/splits/deerK5L_test.txt'
    # data_dir = '/media/r11-01/mz/datasets/deformable_datasets/iccv23/DeformingThings4D-original/animals'
    # output_dir = '/media/r11-01/mz/datasets/deformable_datasets/iccv23/DeformingThings4D-original/animal_pc_correspondences'

    BASEDIR = os.path.dirname(os.path.abspath(__file__))

    # category = 'tigerD8H'
    # category_lst = BASEDIR + '/processed_data/splits/tigerD8H_test.txt'
    data_dir = BASEDIR +  '/original_dataset'
    output_dir = BASEDIR + '/correspondences_for_pck'

    categories = os.listdir(data_dir)
    categories_iter = tqdm(categories)

    pdb.set_trace()

    for category in categories_iter:
        output_path = os.path.join(output_dir, category)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        dir_ = data_dir + '/' + category
        _,_,_, vert_data, face_data, offset_data = anime_read(os.path.join(dir_, dir_.split('/')[-1]+'.anime'))
        
        vert_, index_ = farthest_point_sample_centroids(vert_data, 2048)  # PCD [2048x3], indexes [2048]
        offsets_ = offset_data[:, index_]
        
        xt = np.asarray([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype='f')
        yt = np.asarray([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype='f')
        pcd0 = (yt[:3,:3] @ xt[:3,:3] @ vert_.T).T   

        off = []
        for i, offset in enumerate(offsets_):
            off.append((yt[:3,:3] @ xt[:3,:3] @ offset.T).T)

        off = np.asarray(off)
        
        np.savez(output_path+"/{}".format(category), pcd0=pcd0, offsets=off)


def create_pcds_geodesics():
    '''
    original file with some updates to save goedesics and PCDs

    '''

    BASEDIR = os.path.dirname(os.path.abspath(__file__))


    # data_dir = '/media/r11-01/mz/datasets/deformable_datasets/iccv23/DeformingThings4D-original/animals'
    # output_dir = '/media/r11-01/mz/datasets/deformable_datasets/iccv23/DeformingThings4D-original/animal_pc_correspondences'
    
    data_dir = BASEDIR +  '/original_dataset'
    output_dir = BASEDIR + '/pcds_geodesics'

    list_paths = glob.glob(data_dir+'/*')
    list_paths = natsorted(list_paths)
    list_paths_iter = tqdm(list_paths)

    for path in list_paths_iter:
        _,_,_,vert_data, face_data, offset_data = anime_read(os.path.join(path, path.split('/')[-1]+'.anime'))
        for i, offset in enumerate(offset_data):
            mesh = trimesh.Trimesh(vert_data + offset, face_data) 

            xt = np.asarray([[-1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype='f')
            yt = np.asarray([[0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1]], dtype='f')
            pcd = normalize_pc(farthest_point_sample(mesh.sample(16000), 2048))
            pcd = (yt[:3,:3] @ xt[:3,:3] @ pcd.T).T
            geo_desics = gen_geo_dists(pcd)

            name = path.split('/')[-1]
            output_pcd_path = os.path.join(output_dir, "pcds/{}".format(name))
            output_geodesics_path = os.path.join(output_dir, "geodesics/{}".format(name))
        #    output_images_path = os.path.join(output_dir, "images/{}".format(name))

            if not os.path.exists(output_pcd_path):
                os.makedirs(output_pcd_path)
            
            if not os.path.exists(output_geodesics_path):
                os.makedirs(output_geodesics_path)

            np.save(output_pcd_path+"/{}_{}".format(i,name), pcd)
            np.save(output_geodesics_path+"/{}_{}".format(i,name), geo_desics)
        
        
        

if __name__ == '__main__':
    
    create_pcds_geodesics()
    # create_pcds_and_correspondences()
    # read_npz_files()
    
    






