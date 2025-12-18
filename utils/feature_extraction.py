import taufactor as tau         # to compute tau factor of each phase
from taufactor.metrics import volume_fraction, triple_phase_boundary, specific_surface_area
from scipy.stats import lognorm
import porespy as ps            # to do the pore size analysis
import numpy as np

def image_attribute_extraction(img, Res=1):
    """Extract image attributes: volume fraction, specific surface area,
    triple phase boundary, tortuosity factor, and pore radius.
    Parameters
    ----------
    img : ndarray
        3D numpy array of the image with integer values representing different phases.
    Res : float, optional
        Resolution of the image (default is 1).
    Returns
    -------
    vf : ndarray
        Volume fraction of each phase.
    ssa : ndarray
        Specific surface area of each phase.
    tpb : float
        Triple phase boundary normalized by Res^2.
    tortuosity : ndarray
        Tortuosity factor of each phase.
    r_p : ndarray
        Pore radius of each phase.
    """
    vf = volume_fraction(img)
    vf = np.array(list(vf.values()))
    ssa = specific_surface_area(img)
    ssa = np.array(list(ssa.values())) / Res
    tpb = triple_phase_boundary(img.astype(np.float32)).item() / (Res**2)
    tortuosity = np.zeros(3)
    r_p = np.zeros(3)
    for i in range(3):
        s = tau.Solver(img==i)
        s.solve()
        tortuosity[i] = s.tau.item()

        chords = ps.filters.apply_chords_3D(img==i)
        chord_length = ps.metrics.chord_counts(im=chords)
        shape, loc, scale = lognorm.fit(chord_length, floc=0)
        sigma = shape          # this is σ
        mu = np.log(scale)     # this is μ
        r_p[i] = np.exp(mu + 0.5 * sigma**2) * Res

    return vf, ssa, tpb, tortuosity, r_p