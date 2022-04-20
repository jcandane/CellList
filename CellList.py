import numpy as np 

π = np.pi 

def celllist(R_ix, cube_size, boxboxcuttoff=1.5):
    """
    GIVEN:  R_ix (Cartesian Positions)
            cube_size (Cell List Box size)
            **boxboxcuttoff (cutoff for Box Verlet List in units of `cube_size`)
    GET:    IJ (Box Pair-Neighbors)
            ind (indices of atoms sorted by which boxes they belong to)
            sor (sorted array, how to slice the indice array to get members of a box)
    """

    R_ix_int  = ( R_ix/cube_size ).astype(int)
    lims      = 1 + np.amax( R_ix_int, axis=0 ) - np.amin( R_ix_int, axis=0 )
    boxes     = np.arange(0, np.prod(lims), 1, dtype=int).reshape((lims[0],lims[1],lims[2]))

    ### get indices of 
    box = boxes[R_ix_int[:,0], R_ix_int[:,1], R_ix_int[:,2]] ## get box for each atom
    ind = np.argsort(box) ## find indices beloning to a box
    sor = np.append([0], np.where( np.diff( box[ind] ) >= 1 )[0] ) ### find where boxes end, i.e. box[ind] = np.sort( box ) !!! >=   ### if this is greater than 1 (not equal to get rid of box??)
    sor = np.append(sor, len(box)) ## !!

    ### get Box center coordinates
    x_  = np.arange(0, lims[0], 1, dtype=int)
    y_  = np.arange(0, lims[1], 1, dtype=int)
    z_  = np.arange(0, lims[2], 1, dtype=int)
    xyz = np.array(np.meshgrid(x_, y_, z_))

    ### Box Verlet List
    R_Bx  = (xyz.swapaxes(0,3)).reshape((xyz.size//xyz.shape[0] , xyz.shape[0]), order="F")
    Boxdistances = np.linalg.norm( R_Bx[None, :,:] -  R_Bx[:,None,:] , axis=2)
    IJ = np.asarray( np.where( np.logical_and( Boxdistances >= 0, Boxdistances < 1.5) ) ).T

    return IJ, ind, sor

def boxbox(C_ix, AT_i,    IJ, indexes, sorted,     E_nAR, u, router):
    """
    GIVEN:  Atom Character;   R_ix (positions), AT_i (Unary Atom-Types)
            Box Neighborhood; IJ (Box Pairs), indexes (over all atoms), sorted (over all boxes)
            Potential;  E_nAR (Potential; n (derivative), A (pair-type), and R (Radial distance))
                        u (unit of potential vs Bohr)
                        router (router for pair-wise atom types)
    GET:    E (energy), f_ix (force), and g_r (pair-correlation)
    """

    #R_ix  = C_ix * åCu ## convert R_ix from Bohr into tabulated units
    R_ix  = C_ix * u ##
    E     = 0.
    f_ix  = np.zeros(R_ix.shape)
    g_r   = np.zeros(E_nAR[0].shape)
    for i in range(len(IJ)):
        I, J  = IJ[i]
        I_i   = indexes[ sorted[I]:sorted[I+1] ] ## atom indices belonging to box I
        J_j   = indexes[ sorted[J]:sorted[J+1] ] ## atom indices belonging to box J

        ## displacement/distance calculation
        R_ijx    = R_ix[ I_i, None, : ] - R_ix[ None, J_j, : ]
        distance = np.linalg.norm(R_ijx, axis=2)
        R_ijx   *= 1/(distance[:,:,None] + 1e-10)
        N, dx    = distance.astype(int), distance-distance.astype(int)

        ## get pair-atom-types from unary-atom-types
        AT = router[ np.outer(AT_i[I_i], np.ones(len(J_j), dtype=int)) , np.outer(np.ones(len(I_i), dtype=int), AT_i[J_j]) ]

        ## save pair-correlation, pair-energy, & pair-forces
        g_r[AT, N] += 1
        E          += np.sum( E_nAR[ 0, AT, N ] - E_nAR[ 1, AT, N ] * dx/u )/2 ### use instead analytic derivative?? E_nAR[ 1, AT, N ]??
        #E          += np.sum( E_nAR[ 0, AT, N ] - (E_nAR[ 1, AT, N ]/åCu) * dx )/2
        #E          += np.sum(E_nAR[ 0, AT, N ] + (E_nAR[ 0, AT, N+1 ] - E_nAR[ 0, AT, N ] ) * dx)/2 ### use instead analytic derivative?? E_nAR[ 1, AT, N ]??
        f_ijx       =  (E_nAR[ 1, AT, N ] + (E_nAR[ 1, AT, N+1 ] - E_nAR[ 1, AT, N ] ) * dx)[:,:,None] * R_ijx
        f_ix[I_i]  += np.sum( f_ijx, axis=1)
        f_ix[J_j]  -= np.sum( f_ijx, axis=0)

    return E, f_ix, g_r 


