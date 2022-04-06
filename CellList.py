import numpy as np 

π = np.pi 

def Cell_List(R_ix, cutoff, boxboxcuttoff=1.5):

    R_ix_int = ( R_ix/cutoff ).astype(int)
    lims     = 1 + np.amax( R_ix_int, axis=0 ) - np.amin( R_ix_int, axis=0 )
    boxes    = np.arange(0, np.prod(lims), 1, dtype=int).reshape((lims[0],lims[1],lims[2]))

    box = boxes[R_ix_int[:,0], R_ix_int[:,1], R_ix_int[:,2]] ## get box for each atom
    ind = np.argsort(box) ## find indices beloning to a box
    sor = np.append([0], np.where( np.diff( box[ind] ) == 1 )[0] ) ### find where boxes end, i.e. box[ind] = np.sort( box )
    sor = np.append(sor, box[-1] ) ## !!

    x_  = np.arange(0, lims[0], 1, dtype=int)
    y_  = np.arange(0, lims[1], 1, dtype=int)
    z_  = np.arange(0, lims[2], 1, dtype=int)
    xyz = np.array(np.meshgrid(x_, y_, z_))

    R_Bx  = (xyz.swapaxes(0,3)).reshape((xyz.size//xyz.shape[0] , xyz.shape[0]), order="F")
    R_BCx = np.einsum("Bx, C -> BCx", R_Bx, np.ones(len(R_Bx), dtype=int)) - np.einsum("Cx, B -> BCx", R_Bx, np.ones(len(R_Bx), dtype=int))
    Boxdistances = np.einsum("BCx -> BC", R_BCx**2)**0.5
    II, JJ = np.where( np.logical_and(np.triu( Boxdistances ) > 0, np.triu( Boxdistances ) < boxboxcuttoff) )

    return np.asarray([II, JJ]).T, ind, sor


