import h5py

path = '/home/ubuntu/tmp/test_4_act/tonglu0602_cart_abs_data_test/episode_0.hdf5'

# with h5py.File(path, 'r') as f:
#     qpos =f['/observations/qpos'][()]
#     action =f['/action'][()]
#     images_group =f['/observations/images']

#     keys = list(images_group.keys())

#     print(0)

def get_all_keys(file_path):
    all_keys = []
    with h5py.File(file_path, 'r') as f:
        f.visit(lambda name: all_keys.append(name))

    print(all_keys) 

if __name__ == "__main__":
    get_all_keys(path)
    # pass