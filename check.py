import h5py

f = h5py.File('data/modelnet40_ply_hdf5_2048/ply_data_test0.h5', 'r')
print(list(f.keys()))  # 应该输出 ['data', 'label']
print(f['data'].shape)  # 应该是 (N, 2048, 3)
print(f['label'].shape)  # 应该是 (N, 1)
