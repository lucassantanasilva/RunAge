#!/usr/bin/env python
# coding: utf-8

# In[1]:


def invdyn3d(rcm, rd, rp, acm, alpha, omega, mass, Icm, Fd, Md, e1, e2, e3):
    """Three-dimensional inverse-dynamics calculations of one segment

    Parameters
    ----------
    rcm   : array_like [x,y,z]
            center of mass position (y is vertical)
    rd    : array_like [x,y,z]
            distal joint position
    rp    : array_like [x,y,z]
            proximal joint position
    acm   : array_like [x,y,z]
            center of mass acceleration 
    alpha : array_like [x,y,z]
            segment angular acceleration at the local coordinate system
    omega : array_like [x,y,z]
            segment angular velocity at the local coordinate system
    mass  : number
            mass of the segment   
    Icm   : array_like [x,y,z]
            rotational inertia around the center of mass of the segment
    Fd    : array_like [x,y,z]
            force on the distal joint of the segment
    Md    : array_like [x,y,z]
            moment of force on the distal joint of the segment in the local basis
    e1    : array_like [x,y,z]
             segment versor in the i axis
    e2    : array_like [x,y,z]
             segment versor in the j axis
    e3    : array_like [x,y,z]
             segment versor in the k axis  
    Returns
    -------
    Fp    : array_like [x,y,z]
            force on the proximal joint of the segment (y is vertical)
    Mp    : array_like [x,y,z]
            moment of force on the proximal joint of the segment in the local basis

    Notes
    -----
    To use this function recursevely, the outputs [Fp, Mp] must be inputed as 
    [-Fp, -Mp] on the next call to represent [Fd, Md] on the distal joint of the
    next segment (action-reaction).
    
    This code was inspired by similar codes written by Ton van den Bogert [1]_ and Marcos Duarte [2]_.

    References
    ----------
    .. [1] http://isbweb.org/data/invdyn/index.html
    .. [2] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/GaitAnalysis2D.ipynb

    """
    
    from numpy import cross
    import numpy as np
     
    g = 9.80665  # m/s2, standard acceleration of free fall (ISO 80000-3:2006)
    # Force and moment of force on the proximal joint
    Fp = mass*acm - Fd - [0, -g*mass, 0]
    
    Mdis = cross(rd-rcm, Fd)
    Mprox = cross(rp-rcm, Fp)    
    #Global do Local
    MdisLocal = np.zeros_like(Mdis)
    MproxLocal = np.zeros_like(Mdis)
    for i in range(Mdis.shape[0]):
        RLGMF_grf = np.vstack((e1[i,:],e2[i,:],e3[i,:]))
        MdisLocal[i,:]=RLGMF_grf@Mdis[i,:]
        MproxLocal[i,:]=RLGMF_grf@Mprox[i,:]
    
    Mp = (Icm@alpha.T).T  + np.cross(omega[0:alpha.shape[0],:], (Icm@omega[0:alpha.shape[0],:].T).T,axis=1) - Md - MdisLocal - MproxLocal
    
    
    return Fp, Mp

