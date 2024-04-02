import pandas as pd
import h5py


def createHDF5():
    csv_file_path = 'Data/Connor-Data.csv'
    raw_data = pd.read_csv(csv_file_path)

    interval_length = 5

    interval_starts = raw_data['Time (s)'] // interval_length * interval_length
    unique_starts = interval_starts.drop_duplicates().index

    train_data = pd.concat([raw_data.loc[start:start + interval_length - 1].reset_index(drop=True)
                            for start in unique_starts], keys=unique_starts)

    test_data = raw_data

    hdf5_file_name = 'Data/elec292-data.hdf5'

    with h5py.File(hdf5_file_name, 'w') as hdf:
        train_group = hdf.create_group('Train')
        test_group = hdf.create_group('Test')

        train_member1 = train_group.create_group('Member1')
        test_member1 = test_group.create_group('Member1')

        train_member1.create_dataset('segmented_data', data=train_data.to_numpy(), compression="gzip")
        test_member1.create_dataset('original_data', data=test_data.to_numpy(), compression="gzip")


    structure = {}
    with h5py.File(hdf5_file_name, 'r') as hdf:
        def print_name(name):
            print(name)
        hdf.visit(print_name)

        def get_structure(name, node):
            if isinstance(node, h5py.Dataset):
                structure[name] = 'Dataset'
            else:
                structure[name] = {n: 'Group' for n in node}
        hdf.visititems(get_structure)

    structure

def visualize(hdf5_file):
    how 
