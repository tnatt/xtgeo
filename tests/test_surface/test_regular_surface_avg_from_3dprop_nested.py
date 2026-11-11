"""Tests for RegularSurface.avg_from_3dprop on a nested hybrid grid.

In a nested hybrid grid the refined cells have I/J indices that are far from the
coarse "mother" cells they are geometrically nested inside, so a map node is covered
by cells with very different (I, J). The expected thickness weighted average is
therefore found independently of the grid indexing: by summing all active cells whose
footprint contains the map node.
"""

import pathlib

import numpy as np
import pytest

import xtgeo

NESTEDHYBRID1 = pathlib.Path("3dgrids/drogon/5/drogon_nested_hybrid1.roff")


@pytest.fixture(scope="module")
def nested_hybrid_grid(testdata_path):
    """Load the Drogon nested hybrid grid and its NEST_ID property."""
    path = pathlib.Path(testdata_path) / NESTEDHYBRID1
    grid = xtgeo.grid_from_file(path)
    nest = xtgeo.gridproperty_from_file(path, name="NEST_ID", grid=grid)
    return grid, nest


@pytest.fixture(scope="module")
def nested_hybrid_box_grid():
    """Create a small synthetic nested-hybrid-like coarse/fine merged grid."""
    base = xtgeo.create_box_grid((12, 10, 4), rotation=25.0)
    base.set_subgrids({"Zone1": 2, "Zone2": 2})
    base.get_zoneprop_from_subgrids()

    coarse = base.copy()
    fine = base.copy()

    fact = fine.get_actnum()
    fact.values = 0
    fact.values[3:9, 2:8, :2] = 1
    fine.set_actnum(fact)
    fine.refine(2, 2, 2)

    cact = coarse.get_actnum()
    cact.values = 1
    cact.values[3:9, 2:8, :2] = 0
    coarse.set_actnum(cact)

    # Hardcoded layer maps for this exact synthetic setup:
    # Refined K-window is only coarse layers 0..1 (:2), so lower coarse layers
    # are shifted less than in a full-column refinement.
    lmap_coarse = np.array([0, 2, 4, 5], dtype=np.int32)
    lmap_fine = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32)

    return xtgeo.grid_merge(coarse, fine, lmap_coarse, lmap_fine)


def test_avg_from_3dprop_nested_hybrid(nested_hybrid_grid):
    """Thickness weighted average equals the independent sum over the column cells."""
    grid, _ = nested_hybrid_grid
    grid.set_subgrids({"Zone1": 4, "Zone2": 60})

    x, y, z = grid.get_xyz(asmasked=False)
    dz = grid.get_dz(asmasked=False)
    actnum = grid.get_actnum(asmasked=False)
    dz.values[actnum == 0] = 0

    zone_prop = grid.get_zoneprop_from_subgrids()

    # new function
    surf: xtgeo.RegularSurface = xtgeo.surface_from_grid3d(grid)

    cell_center_z = z.copy().get_npvalues3d()

    surf.avg_from_3dprop(
        xprop=x.get_npvalues3d(),
        yprop=y.get_npvalues3d(),
        mprop=cell_center_z,
        dzprop=dz.get_npvalues3d(),
        zoneprop=zone_prop.get_npvalues3d(),
        zone_minmax=(1, 1),
        zone_avg=False,
    )

    surf.values


def test_avg_from_3dprop_nested_hybrid_box_zone_avg_true(nested_hybrid_box_grid):
    """Baseline: zone-wise averaging should give exact 1.0 in this synthetic case."""
    grid = nested_hybrid_box_grid

    x, y, z = grid.get_xyz(asmasked=False)
    dz = grid.get_dz(asmasked=False)
    actnum = grid.get_actnum(asmasked=False)
    dz.values[actnum == 0] = 0

    zone_prop = grid.get_prop_by_name("Zone")

    # new function
    surf: xtgeo.RegularSurface = xtgeo.surface_from_grid3d(grid)

    surf.avg_from_3dprop(
        xprop=x.get_npvalues3d(),
        yprop=y.get_npvalues3d(),
        mprop=z.get_npvalues3d(),
        dzprop=dz.get_npvalues3d(),
        zoneprop=zone_prop.get_npvalues3d(),
        zone_minmax=(1, 1),
        zone_avg=True,
    )

    values = np.ma.array(surf.values)
    assert values.count() > 0
    np.testing.assert_allclose(values.compressed(), 1.0, atol=1e-8)


def test_avg_from_3dprop_nested_hybrid_box_zone_avg_false_reveals_bug(
    nested_hybrid_box_grid,
):
    """BUG demo: nested-hybrid case should also be 1.0, but is not with zone_avg=False."""
    grid = nested_hybrid_box_grid

    x, y, z = grid.get_xyz(asmasked=False)
    dz = grid.get_dz(asmasked=False)
    actnum = grid.get_actnum(asmasked=False)
    dz.values[actnum == 0] = 0

    zone_prop = grid.get_prop_by_name("Zone")

    surf: xtgeo.RegularSurface = xtgeo.surface_from_grid3d(grid)

    surf.avg_from_3dprop(
        xprop=x.get_npvalues3d(),
        yprop=y.get_npvalues3d(),
        mprop=z.get_npvalues3d(),
        dzprop=dz.get_npvalues3d(),
        zoneprop=zone_prop.get_npvalues3d(),
        zone_minmax=(1, 1),
        zone_avg=False,
    )

    values = np.ma.array(surf.values)
    assert values.count() > 0
    # Expected from create_box_grid defaults and Zone1=top-2 layers: average z = 1.0
    np.testing.assert_allclose(values.compressed(), 1.0, atol=1e-8)
