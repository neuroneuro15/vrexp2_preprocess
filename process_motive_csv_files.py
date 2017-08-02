import pandas as pd
import h5py
import nested_h5py
import os
from os import path
from glob import glob, iglob

def read_motive_csv(fname):
    df = pd.read_csv(fname, skiprows=1, header=[0, 1, 3, 4], 
                     index_col=[0, 1], tupleize_cols=True)
    df.index.names = ['Frame', 'Time']
    df.columns = [tuple(cols) if 'Unnamed' not in cols[3] else tuple([*cols[:-1] + (cols[-2],)]) for cols in df.columns]
#     df.col
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['DataSource', 'ObjectName', 'CoordinateType', 'Axis'])
    return df


def extract_motive_metadata(motive_csv_fname):
    with open(motive_csv_fname) as f:
        line = f.readline()

    cols = line.strip().split(',')
    session_metadata = {x: y for x, y in zip(cols[::2], cols[1::2])}

    # Attempt to coerce values to numeric data types, if possible
    for key, value in session_metadata.items():
        try:
             session_metadata[key] = float(value) if '.' in value else int(value)
        except ValueError:
            pass 
    return session_metadata


def convert_motive_csv_to_hdf5(csv_fname, h5_fname):
    if not path.exists(path.split(path.abspath(h5_fname))[0]):
        os.makedirs(path.split(path.abspath(h5_fname))[0])
    df = read_motive_csv(csv_fname)
    df = df.reset_index('Time')
    session_metadata = extract_motive_metadata(csv_fname)

    if session_metadata['Total Exported Frames'] != len(df):
        with open('log_csv_to_hdf5.txt', 'a') as f:
            f.write('Incomplete: {}, (csv: {} Frames of {} Motive Recorded Frames\r\n'.format(path.basename(csv_fname), 
                len(df), session_metadata['Total Exported Frames']))

    nested_h5py.write_to_hdf5_group(h5_fname, df, '/raw/')
    
    with h5py.File(h5_fname, 'r+') as f:
        f.attrs.update(session_metadata)


def add_orientation_dataset(h5_fname):
    with h5py.File(h5_fname, 'r+') as f:
        f.copy('/raw', '/preprocessed')
        for name, obj in nested_h5py.walk_h5py_path(f['/preprocessed/']):
            if not isinstance(obj, h5py.Dataset) or not 'Rotation' in name:
                continue
            
            rot_df = pd.DataFrame.from_records(obj.value).set_index('Frame')
            rot_df.columns = rot_df.columns.str.lower()

            oris, ori0 = [], rc.Camera().orientation0
            for idx, row in rot_df.iterrows():
                oris.append(rc.RotationQuaternion(**row).rotate(ori0))

            odf = pd.DataFrame(oris, columns=['X', 'Y', 'Z'], index=rot_df.index)
            f.create_dataset(name=obj.parent.name + '/Orientation', data=odf.to_records())

    with open(path.join(path.dirname(h5_fname), 'ori_added.txt'), 'w'):
        pass


basedir = '/home/nickdg/theta_storage/data/VR_Experiments_Round_2/Converted Motive Files'


csv_fnames = glob(basedir + '/**/*.csv', recursive=True)
new_basedir = path.join(path.commonpath(csv_fnames), '..', 'processed_data')
h5_fnames = [path.join(new_basedir, path.basename(path.splitext(name)[0]), path.basename(path.splitext(name)[0] + '.h5')) for name in csv_fnames]


def task_convert_motive_csv_to_hdf5():
    for csv_fname, h5_fname in zip(csv_fnames, h5_fnames):
        task = {
            'actions': [(convert_motive_csv_to_hdf5, (csv_fname, h5_fname))],
            'targets': [h5_fname],
            'file_dep': [csv_fname],
            'name': 'convert_csv_to_h5: {}'.format(path.basename(h5_fname)),
        }
        yield task

        yield {
            'actions': [(add_orientation_dataset, (h5_fname,))],
            'targets': [path.join(path.dirname(h5_fname), 'ori_added.txt')],
            'file_dep': [h5_fname],
            # 'task_dep': [task['name']],
            'name': 'add_orientation: {}'.format(path.basename(h5_fname)),
        }


    

if __name__ == '__main__':
    import doit
    doit.run(globals())