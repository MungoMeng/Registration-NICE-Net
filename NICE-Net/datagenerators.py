import os, sys
import numpy as np
import scipy.ndimage
        
        
def gen_s2s(gen, batch_size=1):

    while True:
        X = next(gen)
        fixed = X[0]
        moving = X[1]          
        
        # Downsample input for deep supervision
        fix_down_2 = scipy.ndimage.interpolation.zoom(fixed, (1,0.5,0.5,0.5,1), order = 1)
        mov_down_2 = scipy.ndimage.interpolation.zoom(moving, (1,0.5,0.5,0.5,1), order = 1)
        fix_down_4 = scipy.ndimage.interpolation.zoom(fixed, (1,0.25,0.25,0.25,1), order = 1)
        mov_down_4 = scipy.ndimage.interpolation.zoom(moving, (1,0.25,0.25,0.25,1), order = 1)
        fix_down_8 = scipy.ndimage.interpolation.zoom(fixed, (1,0.125,0.125,0.125,1), order = 1)
        mov_down_8 = scipy.ndimage.interpolation.zoom(moving, (1,0.125,0.125,0.125,1), order = 1)
        
        volshape = X[0].shape[1:-1]
        zeros = np.zeros((batch_size, *volshape, len(volshape)))
            
        yield ([fixed, moving, mov_down_2, mov_down_4, mov_down_8], [zeros, fixed, zeros, fix_down_2, zeros, fix_down_4, zeros, fix_down_8])
        

def pairs_gen(path, pairs, batch_size=1, random=True):
    """
    random = 1: randomly pick pairs from pairs.npy
    random = 0: pick pairs from pairs.npy one by one
    """
    
    pairs = np.load(path+pairs)
    pairs_num = len(pairs)
    i = 0   
    while True:
        
        if random == False:
            idxes = range(i,i+batch_size)
            i = i+batch_size
            if i == pairs_num:
                i=0   
        
        else: # random == True
            idxes = np.random.randint(pairs_num, size=batch_size)

        # load fixed images
        X_data = []
        for idx in idxes:
            fixed = bytes.decode(pairs[idx][0])
            X = load_volfile(path+fixed, np_var='vol')
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)
        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]

        # load moving images
        X_data = []
        for idx in idxes:
            moving = bytes.decode(pairs[idx][1])
            X = load_volfile(path+moving, np_var='vol')
            X = X[np.newaxis, ..., np.newaxis]
            X_data.append(X)
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data, 0))
        else:
            return_vals.append(X_data[0])
    
        yield tuple(return_vals)


def load_example_by_name(vol_name):
    """
    load a specific volume and segmentation
    np_var: specify the name of the variable in numpy files, if your data is stored in 
        npz files. default to 'vol_data'
    """
    
    X = load_volfile(vol_name, 'vol')
    X = X[np.newaxis, ..., np.newaxis]
    return_vals = [X]

    X_seg = load_volfile(vol_name, np_var='seg')
    X_seg = X_seg[np.newaxis, ..., np.newaxis]
    return_vals.append(X_seg)

    return tuple(return_vals)


def load_volfile(datafile, np_var):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
        if 'nibabel' not in sys.modules:
            try :
                import nibabel as nib  
            except:
                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()
        
    else: # npz
        X = np.load(datafile)[np_var]

    return X
