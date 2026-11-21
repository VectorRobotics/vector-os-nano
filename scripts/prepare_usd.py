# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024-2026 Vector Robotics
"""Offline USD preparation for the MuJoCo importer; original assets remain immutable."""
import numpy as np
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade, Vt


def normalize(stage):
    """Bake scale/shear into meshes, retaining world geometry and joint anchor points.

    MuJoCo 4.0.1's USD reader divides geom translations by scaled body frames;
    rigid frames also avoid its default matrix-factorization epsilon for tiny CAD units.
    Input is an editable, flattened stage, never the upstream source layer.
    """
    cache = UsdGeom.XformCache(0)
    prims = list(stage.Traverse())
    for prim in prims:
        if prim.IsA(UsdGeom.Gprim) and not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Cylinder)):
            raise ValueError(f'{prim.GetPath()}: unsupported USD geometry {prim.GetTypeName()}; convert to a mesh first')
    old = {str(p.GetPath()): np.array(cache.GetLocalToWorldTransform(p)) for p in prims}
    new = {}
    for path, matrix in old.items():
        if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix[:3, :3])) == 0:
            raise ValueError(f'{path}: non-finite or singular USD transform')
        if stage.GetPrimAtPath(path).HasAPI(UsdPhysics.RigidBodyAPI) and np.linalg.det(matrix[:3, :3]) < 0:
            raise ValueError(f'{path}: mirrored rigid frame requires joint/inertia conversion')
        u, _, vt = np.linalg.svd(matrix[:3, :3])
        if np.linalg.det(u @ vt) < 0:
            u[:, -1] *= -1
        rigid = np.eye(4)
        rigid[:3, :3], rigid[3, :3] = u @ vt, matrix[3, :3]
        new[path] = rigid
    for p in prims:
        path = str(p.GetPath())
        if p.GetRelationship('material:binding'):
            UsdShade.MaterialBindingAPI.Apply(p)
        if p.IsA(UsdGeom.Xformable):
            xf = UsdGeom.Xformable(p)
            parent = new.get(str(p.GetParent().GetPath()), np.eye(4))
            xf.ClearXformOpOrder()
            xf.AddTransformOp(opSuffix='metric').Set(Gf.Matrix4d(new[path] @ np.linalg.inv(parent)))
            xf.SetResetXformStack(False)
        if p.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(p)
            points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
            bake = old[path] @ np.linalg.inv(new[path])
            result = points @ bake[:3, :3] + bake[3, :3]
            mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(result.astype(np.float32)))
            mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([result.min(0), result.max(0)], dtype=np.float32)))
            assert np.allclose(result @ new[path][:3, :3] + new[path][3, :3],
                               points @ old[path][:3, :3] + old[path][3, :3], atol=1e-6)
            normals = mesh.GetNormalsAttr().Get()
            if normals:
                normals = np.array(normals) @ np.linalg.inv(bake[:3, :3]).T
                normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-20)
                mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
            if np.linalg.det(bake[:3, :3]) < 0:
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
                offsets = np.r_[0, np.cumsum(counts)]
                order = np.concatenate([np.arange(a, b)[::-1] for a, b in zip(offsets[:-1], offsets[1:])])
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
                mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(indices[order]))
                if normals is not None and len(normals) and mesh.GetNormalsInterpolation() == 'faceVarying':
                    mesh.GetNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals[order].astype(np.float32)))
                for var in UsdGeom.PrimvarsAPI(p).GetPrimvars():
                    if var.GetInterpolation() != 'faceVarying':
                        continue
                    value = var.GetIndices() if var.IsIndexed() else var.Get()
                    if value:
                        transformed = type(value)([value[int(i)] for i in order])
                        var.SetIndices(transformed) if var.IsIndexed() else var.Set(transformed)
        elif p.IsA(UsdGeom.Cylinder):
            cyl = UsdGeom.Cylinder(p)
            axis = {'X': 0, 'Y': 1, 'Z': 2}[cyl.GetAxisAttr().Get()]
            scale = np.linalg.norm(old[path][:3, :3], axis=1)
            radial = np.delete(scale, axis)
            assert np.isclose(*radial), 'Elliptical cylinders need a mesh conversion'
            cyl.GetRadiusAttr().Set(cyl.GetRadiusAttr().Get() * radial[0])
            cyl.GetHeightAttr().Set(cyl.GetHeightAttr().Get() * scale[axis])
        if p.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(p)
            for side in (0, 1):
                targets = getattr(joint, f'GetBody{side}Rel')().GetTargets()
                if not targets:
                    continue
                body = str(targets[0])
                attr = getattr(joint, f'GetLocalPos{side}Attr')()
                point = np.r_[attr.Get(), 1] @ old[body] @ np.linalg.inv(new[body])
                attr.Set(Gf.Vec3f(*point[:3]))
    return stage


def main():
    """Write a prepared copy for the native MuJoCo USD importer."""
    import argparse
    from pathlib import Path
    from pxr import Usd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.output.resolve():
        parser.error('output must differ from the immutable upstream source')
    stage = Usd.Stage.Open(args.source.as_posix())
    prepared = normalize(Usd.Stage.Open(stage.Flatten()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not prepared.GetRootLayer().Export(args.output.as_posix()):
        raise RuntimeError(f'USD export failed: {args.output}')


if __name__ == '__main__':
    main()
