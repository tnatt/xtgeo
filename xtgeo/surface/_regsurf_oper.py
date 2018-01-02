"""Various operations"""
from __future__ import print_function, absolute_import

import cxtgeo.cxtgeo as _cxtgeo
from xtgeo.common import XTGeoDialog

xtg = XTGeoDialog()

logger = xtg.functionlogger(__name__)

xtg_verbose_level = xtg.get_syslevel()
if xtg_verbose_level < 0:
    xtg_verbose_level = 0

_cxtgeo.xtg_verbose_file('NONE')


def resample(surf, other):
    """Resample from other surface object to this surf"""

    logger.info('Resampling...')

    svalues = surf.get_zval()

    ier = _cxtgeo.surf_resample(other._ncol, other._nrow,
                                other._xori, other._xinc,
                                other._yori, other._yinc,
                                other._yflip, other._rotation,
                                other.get_zval(),
                                surf._ncol, surf._nrow,
                                surf._xori, surf._xinc,
                                surf._yori, surf._yinc,
                                surf._yflip, surf._rotation,
                                svalues,
                                0,
                                xtg_verbose_level)

    if ier != 0:
        raise RuntimeError('Resampling went wrong, '
                           'code is {}'.format(ier))

    surf.set_zval(svalues)


def distance_from_point(surf, point=(0, 0), azimuth=0.0):

    x, y = point

    svalues = surf.get_zval()

    # call C routine
    ier = _cxtgeo.surf_get_dist_values(
        surf._xori, surf._xinc, surf._yori, surf._yinc, surf._ncol,
        surf._nrow, surf._rotation, x, y, azimuth, svalues, 0,
        xtg_verbose_level)

    if ier != 0:
        surf.logger.error('Something went wrong...')
        raise RuntimeError('Something went wrong in {}'.format(__name__))

    surf.set_zval(svalues)


def get_value_from_xy(surf, point=(0.0, 0.0)):

    xcoord, ycoord = point

    zcoord = _cxtgeo.surf_get_z_from_xy(float(xcoord), float(ycoord),
                                        surf.ncol, surf.nrow,
                                        surf.xori, surf.yori, surf.xinc,
                                        surf.yinc, surf.yflip,
                                        surf.rotation,
                                        surf.get_zval(), xtg_verbose_level)

    if zcoord > surf._undef_limit:
        return None

    return zcoord


def get_xy_value_from_ij(surf, i, j):

    if 1 <= i <= surf.ncol and 1 <= j <= surf.nrow:

        ier, xval, yval, value = (
            _cxtgeo.surf_xyz_from_ij(i, j,
                                     surf.xori, surf.xinc,
                                     surf.yori, surf.yinc,
                                     surf.ncol, surf.nrow, surf._yflip,
                                     surf.rotation, surf.get_zval(),
                                     0, xtg_verbose_level))
        if ier != 0:
            surf.logger.critical('Error code {}, contact the author'.
                                 format(ier))
            raise SystemExit('Error')

    else:
        raise ValueError('Index i and/or j out of bounds')

    if value > surf.undef_limit:
        value = None

    return xval, yval, value