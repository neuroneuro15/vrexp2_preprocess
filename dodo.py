import os
from os import path
from glob import glob
import json
import numpy as np
import pandas as pd
import h5py
import nested_h5py
import ratcave as rc


def read_motive_csv(fname):
    df = pd.read_csv(fname, skiprows=1, header=[0, 1, 3, 4],
                     index_col=[0, 1], tupleize_cols=True)
    df.index.names = ['Frame', 'Time']
    df.columns = [tuple(cols) if 'Unnamed' not in cols[3] else tuple([*cols[:-1] + (cols[-2],)]) for cols in df.columns]
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
    #df = df.reset_index('Time')
    session_metadata = extract_motive_metadata(csv_fname)

    if session_metadata['Total Exported Frames'] != len(df):
        with open('log_csv_to_hdf5.txt', 'a') as f:
            f.write('Incomplete: {}, (csv: {} Frames of {} Motive Recorded Frames\r\n'.format(path.basename(csv_fname),
                len(df), session_metadata['Total Exported Frames']))

    nested_h5py.write_to_hdf5_group(h5_fname, df, '/raw/', 
        compression='gzip', compression_opts=7)

    with h5py.File(h5_fname, 'r+') as f:
        f.attrs.update(session_metadata)

    return None


def add_orientation_dataset(h5_fname, **kwargs):

    with h5py.File(h5_fname, 'r+') as f:
        f.copy('/raw', '/preprocessed')
        for name, obj in nested_h5py.walk_h5py_path(f['/preprocessed/']):
            if not isinstance(obj, h5py.Dataset) or not 'Rotation' in name:
                continue

            rot_df = pd.DataFrame.from_records(obj.value).set_index(['Frame', 'Time'])
            rot_df.columns = rot_df.columns.str.lower()

            oris, ori0 = [], rc.Camera().orientation0
            for _, row in rot_df.iterrows():
                oris.append(rc.RotationQuaternion(**row).rotate(ori0))

            odf = pd.DataFrame(oris, columns=['X', 'Y', 'Z'], index=rot_df.index)
            f.create_dataset(name=obj.parent.name + '/Orientation',
                data=odf.to_records(), compression='gzip', compression_opts=7)

    return None


def rotate_and_offset(array, rotation_matrices, mean_pos, pos_offset):
    dat = array.copy()
    dat -= mean_pos
    for matrix in rotation_matrices:
        dat = dat @ matrix
    dat[:, 1] += mean_pos[1] + pos_offset[1]
    return dat


def unrotate_objects(h5_fname, group='/preprocessed/Rigid Body', source_object_name='Arena', 
    add_rotation=10.5, index_cols=2, y_offset=-0.66, **kwargs):
    """
    Un-rotate the objects in an hdf5 group by either a set rotationa mount, another object's rotation, or both.

    Arguments:
       -h5_fname (str): filename of hdf5 file to read in.
       -group (str): hdf5 group directory where all objects can be found
       -source_object_name (str): object name to use as un-rotation parent.
       -add_rotation (float): additional amount (degrees) to rotate by.
       -mean_center (bool): if the position should also be moved by the source_object's position.  
       
    """
    source_obj = nested_h5py.read_from_h5_group(h5_fname, path.join(group, source_object_name), index_cols=index_cols)
    mean_pos = source_obj.Position.mean()
    mean_rot = source_obj.Rotation.mean()
    mean_rot.index = mean_rot.index.str.lower()
    rot_mat = rc.RotationQuaternion(**mean_rot).to_matrix()
    manrot = rc.RotationEulerDegrees(x=0, y=add_rotation, z=0).to_matrix()
    rot_matrices = [rot_mat[:-1, :-1], manrot[:-1, :-1]]

    with h5py.File(h5_fname, 'r+') as f:
        
        for name in filter(lambda s: 'position' in s.lower(), f.attrs):
            positions = np.matrix(f.attrs[name])
            positions = [rotate_and_offset(np.matrix(pos), rot_matrices, mean_pos.values, (0., y_offset, 0.)).squeeze() for pos in positions]
            f.attrs[name + ''] = np.array(positions).squeeze()

        bodies = f[group]
        body_paths = [bodies[body].name for body in bodies]

    

    for body in body_paths:
        obj = nested_h5py.read_from_h5_group(h5_fname, path.join(group, body), index_cols=index_cols)
        obj['Position'] = rotate_and_offset(obj.Position.values, rot_matrices, mean_pos.values, (0., y_offset, 0.))
        obj['Orientation'] = rotate_and_offset(obj.Orientation.values, rot_matrices, (0., 0., 0.), (0., 0., 0.))
        
        nested_h5py.write_to_hdf5_group(h5_fname, obj, body + '/', mode='r+', overwrite=True)

    with h5py.File(h5_fname, 'r+') as f:
        for marker_name, dset in f['preprocessed/Rigid Body Marker/'].items():
            obj = pd.DataFrame.from_records(dset['Position'][:]).set_index(['Frame', 'Time'])
            obj[:] = rotate_and_offset(obj[:], rot_matrices, mean_pos.values, (0., y_offset, 0.))
            dset['Position'][:] = obj.to_records()

    return None        


event_log_dir = '/home/nickdg/theta_storage/data/VR_Experiments_Round_2/logs/event_logs/'
settings_log_dir = '/home/nickdg/theta_storage/data/VR_Experiments_Round_2/logs/settings_logs/'


def add_event_log(csv_fname, h5_fname, **kwargs):

    log_fname = path.join(event_log_dir, path.basename(csv_fname))

    if not path.exists(log_fname):
        # Attempt to match name
        for backidx in range(1, 15):
            fname_part = log_fname[:-backidx]
            matches = glob(fname_part + '*')
            if len(matches) == 1:
                log_fname = matches[0]
                break
            if len(matches) > 1:
                print('No matching log found for {}'.format(log_fname))
                return
        else:
            print('No matching log found for {}'.format(log_fname))
            return
    events = pd.read_csv(log_fname, sep=';',)# parse_dates=[0], infer_datetime_format=True)
    events.columns = events.columns.str.lstrip()
    events['Event'] = events.Event.str.lstrip()
    events['EventArguments'] = events.EventArguments.str.lstrip()

    times = pd.read_hdf(h5_fname, '/raw/Rigid Body/Rat/Position').set_index('Frame')['Time']
    event_frames = np.searchsorted(times.values.flatten(), events['MotiveExpTimeSecs'])
    events['Frame'] = event_frames
    events['Time'] = times.loc[event_frames].values

    phase_frames = events[events.Event.str.match('set_')].reset_index().Frame.values

    events.set_index(['Frame', 'Time'], inplace=True)
    event_names = events.Event.values.astype('S')
    del events['Event']
    event_arguments = events.EventArguments.values.astype('S')
    del events['EventArguments']
    del events['DateTime']
    
    with h5py.File(h5_fname, 'r+') as f:
        f.create_dataset('/events/eventlog', data=events.to_records(),
            compression='gzip', compression_opts=7)
        f.create_dataset('/events/eventNames', data=event_names)
        f.create_dataset('/events/eventArguments', data=event_arguments)
        if len(phase_frames) > 0:
            f.create_dataset('/events/phaseStartFrameNum', data=phase_frames)

    return None


def add_settings_log(json_fname, h5_fname, **kwargs):
    """
    Writes a settings log to the hdf5 file as root user attributes, using csv data.

    Arguments:
        -json_fname (str): json filename to read for settings info
        -h5_fname (str): hdf5 filename to write to.
    """
    log_fname = path.join(settings_log_dir, path.basename(json_fname))

    if not path.exists(log_fname):
        # Attempt to match name
        for backidx in range(1, 15):
            fname_part = log_fname[:-backidx]
            matches = glob(fname_part + '*')
            if len(matches) == 1:
                log_fname = matches[0]
                break
            if len(matches) > 1:
                print('No matching log found for {}'.format(log_fname))
                return
        else:
            print('No matching log found for {}'.format(log_fname))
            return

    with open(log_fname) as f:
        sess_data = json.load(f)

    for key, value in sess_data.items():
        if type(value) == bool:
            sess_data[key] = int(value)
        if type(value) == type(None):
            sess_data[key] = 0

    with h5py.File(h5_fname, 'r+') as f:
        f.attrs.update(sess_data)

    return None


def add_softlink_to_markers(h5_fname):
    with open(h5_fname) as f:
        for body_name, body_group in f['preprocessed/Rigid Body'].items():
            body_group.create_group("Markers")
            
            marker_names = [name for name in '/preprocessed/Rigid Body Marker' if body_name in name]
            for marker_name in marker_names:
                body_group['Markers'][marker_name] = h5py.SoftLink('/preprocessed/Rigid Body Marker/'+marker_name)    
    return None


def skip_if_conf_file_exists(conf_fname):
    """
    Skips running function if a given file exists. If not, runs functions and creates the file.

    Arguments:
      -conf_fname (str): filename to check for. (Note: Must be full path)
    """
    def decorator(fun):
        def wrapper(*args, **kwargs):
            
            if path.exists(conf_fname):
                print(fun.__name__ + ' already made for this file.  Skipping step...')
                return
            else:
                output = fun(*args, **kwargs)
                with open(conf_fname, 'w'):
                    pass
                return output
        return wrapper
    return decorator


basedir = '/home/nickdg/theta_storage/data/VR_Experiments_Round_2/Converted Motive Files'


csv_fnames = glob(basedir + '/**/*.csv', recursive=True)
new_basedir = path.join(path.commonpath(csv_fnames), '..', 'processed_data')
h5_fnames = [path.join(new_basedir, path.basename(path.splitext(name)[0]), path.basename(path.splitext(name)[0] + '.h5')) for name in csv_fnames]


def task_preprocess_all_data():
    for csv_fname, h5_fname in zip(csv_fnames[:], h5_fnames):

        dirname = path.dirname(h5_fname)

        if 'test' in csv_fname.lower():
            continue
        if 'habit' in csv_fname.lower():
            continue
        # if not 'spat' in csv_fname.lower():
        #     continue

        convert_task = {
            'actions': [(convert_motive_csv_to_hdf5, (csv_fname, h5_fname))],
            'targets': [h5_fname],
            'file_dep': [csv_fname],
            'name': 'convert_csv_to_h5: {}'.format(path.basename(h5_fname)),
        }
        yield convert_task

        conf_fname = dirname + '/event_log_added.txt'
        add_event_log_if_no_conf_file = skip_if_conf_file_exists(conf_fname)(add_event_log)
        event_task = {
            'actions': [(add_event_log_if_no_conf_file, (csv_fname, h5_fname,))],
            'targets': [conf_fname],
            'task_dep': [convert_task['name']],
            'file_dep': [h5_fname],
            'name': 'add_event_log: {}'.format(path.basename(h5_fname)),
            'verbosity': 2,
        }
        yield event_task

        conf_fname = dirname + '/settings_log_added.txt'
        add_settings_log_if_no_conf_file = skip_if_conf_file_exists(conf_fname)(add_settings_log)
        settings_task = {
            'actions': [(add_settings_log_if_no_conf_file, (csv_fname, h5_fname,))],
            'targets': [conf_fname],
            'task_dep': [event_task['name']],
            'file_dep': [h5_fname],
            'name': 'add_settings_log: {}'.format(path.basename(h5_fname)),
            'verbosity': 2,
        }
        yield settings_task

        conf_fname = dirname + '/ori_added.txt'
        add_orientation_dataset_if_no_conf_file = skip_if_conf_file_exists(conf_fname)(add_orientation_dataset)
        ori_task = {
            'actions': [(add_orientation_dataset_if_no_conf_file, (h5_fname,))],
            'targets': [conf_fname],
            # 'file_dep': [h5_fname],
            'task_dep': [settings_task['name']],
            'name': 'add_orientation: {}'.format(path.basename(h5_fname)),
            'verbosity': 2,
        }
        yield ori_task

        conf_fname = dirname + '/unrotated.txt'
        unrotate_objects_if_no_conf_file = skip_if_conf_file_exists(conf_fname)(unrotate_objects)
        rotate_task = {
            'actions': [(unrotate_objects_if_no_conf_file, (h5_fname,))],
            'targets': [conf_fname],
            # 'file_dep': [h5_fname],
            'task_dep': [ori_task['name']],
            'name': 'unrotate: {}'.format(path.basename(h5_fname)),
            'verbosity': 2,
        }
        yield rotate_task

        conf_fname = dirname + '/softlink_added.txt'
        add_softlinks_if_no_conf_file = skip_if_conf_file_exists(conf_fname)(add_softlink_to_markers)
        softlink_task = {
            'actions': [(add_softlinks_if_no_conf_file, (h5_fname,))],
            'targets': [conf_fname],
            # 'file_dep': [h5_fname],
            'task_dep': [rotate_task['name']],
            'name': 'softlink: {}'.format(path.basename(h5_fname)),
            'verbosity': 2,
        }
        yield softlink_task


if __name__ == '__main__':
    import doit
    doit.run(globals())