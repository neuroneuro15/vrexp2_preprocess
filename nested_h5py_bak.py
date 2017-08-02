import pandas as pd
import h5py
from os import path

def walk_h5py_path(node, dirpath='/'):
    """
    Takes a node from an h5py.File and iterates over each sub, item, 
    yielding the (path, item) of each end node.
    
    Example:
    with h5py.File('myfile.hdf5') as f:
        list(walk_h5py_path(f['/data']))
    >>> [('/data/a/2', <HDF5 dataset>), ('/data/a/1', <HDF5 dataset>),
         ('/data/b/1), <HDF5 dataset)]
    """
    for key, value in node.items():
        currpath = path.join(dirpath, key)
        if isinstance(value, h5py.Dataset):
            yield currpath, value
        else:
            for el in walk_h5py_path(value, dirpath=currpath):
                yield el


def read_from_h5_group(fname, dirpath='/', index_cols=0):
    """
    Walks HDF5 group, building a concatenated MultiIndexed DataFrame.
    
    Arguments:
       - fname (str): HDF5 filename to read from.
       - dirpath (str): Parent group path to start from.
       - index_cols (int): The first N columns to use as an Index 
                   (Assumes that the first N trials have the same names)
      
    """
    ddd = []
    with h5py.File(fname) as f:
        for key, dset in walk_h5py_path(f[dirpath]):
            names = dset.dtype.names
            dd = pd.DataFrame(dset.value)
            ind_names, col_names = names[:index_cols], names[index_cols:]
            if ind_names:
                dd.set_index(list(ind_names), inplace=True)
            cols = [(a,) for a in key.split('/')[1:]] + [col_names]
            dd.columns = pd.MultiIndex.from_product(cols)
            ddd.append(dd)
    return pd.concat(ddd, axis=1)
    

def write_to_hdf5_group(fname, df, dirpath='/'):
    """
    """
    with h5py.File(fname, 'w') as f:
        f.attrs.update(session_metadata)
        for cols in df.columns.droplevel(-1).drop_duplicates():
            f.create_dataset(name=dirpath + '/'.join(cols), data=df[cols].to_records())

