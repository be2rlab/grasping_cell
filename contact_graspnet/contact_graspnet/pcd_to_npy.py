import glob
pcd = glob.glob("/home/diana/Documents/pcd/pcd1/*.pcd")
pcd.sort()
from pypcd import pypcd
import pprint
from pathlib import Path
import numpy as np
for idx in range(len(pcd)):
    cloud=pypcd.PointCloud.from_path(pcd[idx])
    pprint.pprint(cloud.get_metadata())
    new = cloud.pc_data.copy() # copy the data to the new array
    acc=np.array([list(new) for new in new]) # Turn the meta to an array element and pass the result to an array of ACC
    #acc=np.delete(acc,[4,5],1)
    acc[:,3]=0
    acc[:,[0,1]]=acc[:,[1,0]]
    new_cloud = pypcd.make_xyz_label_point_cloud(acc)
    new_cloud.save_pcd(pcd[idx], compression='binary_compressed')
    cloud2=pypcd.PointCloud.from_path(pcd[idx])
    new2 = cloud2.pc_data.copy()
    acc2=np.array([list(new2) for new2 in new2])
    np.save('/home/diana/Documents/pcd/npy1/' + Path(pcd[idx]).stem + ".npy",acc2)