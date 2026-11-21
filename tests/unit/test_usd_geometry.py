"""Scaled CAD imports retain the independently computed world geometry and anchors."""
import numpy as np
import pytest

pytest.importorskip('pxr', reason='offline USD asset-building extra not installed')
from pxr import Sdf, Usd, UsdGeom, UsdPhysics, Vt

from scripts.prepare_usd import normalize


def points(mesh):
    matrix = np.array(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(0))
    return np.c_[np.asarray(mesh.GetPointsAttr().Get()), np.ones(4)] @ matrix


def fixture():
    stage = Usd.Stage.CreateInMemory()
    body = UsdGeom.Xform.Define(stage, '/body')
    body.AddTranslateOp().Set((2, 3, 4))
    body.AddRotateZOp().Set(31)
    body.AddScaleOp().Set((.02, .03, .04))
    UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
    mesh = UsdGeom.Mesh.Define(stage, '/body/mesh')
    mesh.AddTranslateOp().Set((3, 1, -2))
    mesh.AddRotateYOp().Set(19)
    mesh.AddScaleOp().Set((-1.2, 2, 1))
    mesh.CreatePointsAttr([(0,0,0), (1,0,0), (0,1,0), (0,0,1)])
    mesh.CreateFaceVertexCountsAttr([3,3,3,3])
    mesh.CreateFaceVertexIndicesAttr([0,2,1,0,1,3,0,3,2,1,2,3])
    mesh.CreateExtentAttr([(0,0,0),(1,1,1)])
    uv = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'faceVarying')
    uv.Set(Vt.Vec2fArray([(float(i),float(i+1)) for i in range(12)]))
    joint = UsdPhysics.PrismaticJoint.Define(stage, '/joint')
    joint.CreateBody1Rel().SetTargets(['/body'])
    joint.CreateLocalPos1Attr((1,2,3))
    return stage, body, mesh, joint, uv


def test_metric_bake_preserves_vertices_winding_uv_bounds_and_anchor():
    stage, body, mesh, joint, uv = fixture()
    before = points(mesh)
    indices = np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    corner_uv = np.array(uv.Get()).reshape(-1,3,2)
    anchor = np.r_[joint.GetLocalPos1Attr().Get(),1] @ np.array(body.ComputeLocalToWorldTransform(0))
    normalize(stage)
    np.testing.assert_allclose(points(mesh), before, atol=1e-6)
    np.testing.assert_array_equal(np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),indices[:,::-1])
    np.testing.assert_array_equal(np.array(uv.Get()).reshape(-1,3,2),corner_uv[:,::-1])
    np.testing.assert_allclose(np.r_[joint.GetLocalPos1Attr().Get(),1] @ np.array(body.ComputeLocalToWorldTransform(0)),anchor,atol=1e-6)
    for item in (body,mesh):
        rotation = np.array(item.ComputeLocalToWorldTransform(0))[:3,:3]
        np.testing.assert_allclose(rotation@rotation.T,np.eye(3),atol=1e-6)
        assert np.linalg.det(rotation) == pytest.approx(1)
    extent = np.array(mesh.GetExtentAttr().Get())
    local = np.array(mesh.GetPointsAttr().Get())
    np.testing.assert_allclose(extent,[local.min(0),local.max(0)])


def test_singular_source_names_bad_prim():
    stage, body, *_ = fixture()
    body.GetPrim().GetAttribute('xformOp:scale').Set((0,1,1))
    with pytest.raises(ValueError, match='/body:.*singular'):
        normalize(stage)


def test_unsupported_geometry_is_rejected_before_editing_stage():
    stage, body, *_ = fixture()
    UsdGeom.Cube.Define(stage, '/cube')
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='/cube: unsupported USD geometry Cube'):
        normalize(stage)
    assert stage.GetRootLayer().ExportToString() == before


def test_mirrored_rigid_frame_refuses_unconverted_physics():
    stage, body, *_ = fixture()
    body.GetPrim().GetAttribute('xformOp:scale').Set((-1,1,1))
    with pytest.raises(ValueError, match='/body: mirrored rigid frame'):
        normalize(stage)
