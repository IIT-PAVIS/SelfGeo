# CAPE dataset

**Download** the sample of the processed CAPE dataset from the following link: 

**[https://drive.google.com/drive/folders/1NP9Ow8CbKAVhhmrHHlZ_MAhle5ehilPj?usp=sharing](https://drive.google.com/drive/folders/1NP9Ow8CbKAVhhmrHHlZ_MAhle5ehilPj?usp=sharing)**

- The processed sample set contains PCDs and geodesics. 
- The data processing steps are described for the Deforming Things 4D dataset.
- Download and save it to the dataset folder.
- We use the following structure for the CAPE dataset:

```
CAPE/
  |_ train
      |__ geodesic
            |__ *.npy (containing the geodesics)
            ...
      |__ pcd
            |__ *.npy (containing the point clouds)
            ...      
  |_ test
      |__ geodesic
            |__ *.npy (containing the geodesics)
            ...
      |__ pcd
            |__ *.npy (containing the point clouds)
            ...      
  |_ val
      |__ geodesic
            |__ *.npy (containing the geodesics)
            ...
      |__ pcd
            |__ *.npy (containing the point clouds)
            ...      
```

