"""Deterministic printed wrappers on the existing rigid package envelopes."""
import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


LABELS = {
    'tissue_refill': ('TISSUE', 'REFILL', 'SOFT PAPER / SEALED REFILL'),
    'sponge_pack_1': ('KITCHEN', 'SPONGES', 'SEALED PACK / DAILY CLEANING'),
    'sponge_pack_2': ('KITCHEN', 'SPONGES', 'SEALED PACK / DAILY CLEANING'),
    'zip_bag_box': ('STORAGE', 'BAGS', 'RESEALABLE / KITCHEN STORAGE'),
}
FONT = Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')


def wrapper_texture(directory, name, color):
    """Six printed panels; no texture or geometry is borrowed from a real brand."""
    title, subtitle, detail = LABELS[name]
    color = np.asarray(color, dtype=float)
    if color.shape != (4,) or not np.isfinite(color).all() or np.any((color < 0) | (color > 1)):
        raise ValueError('expected finite RGBA in [0, 1]')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name + '-wrapper.png')
    rng = np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little'))
    base = .55*np.array([242, 239, 228]) + .45*color[:3]*255
    paper = np.clip(base + rng.normal(0, .7, (1024, 1536, 1)), 0, 255).astype('uint8')
    image = Image.fromarray(paper)
    draw = ImageDraw.Draw(image)
    accent = tuple((color[:3] * 180).astype(int))
    ink = (37, 45, 39)
    fonts = {size: ImageFont.truetype(str(FONT), size) for size in (15, 20, 24, 48)}
    for face in range(6):
        x, y = face % 3 * 512, face // 3 * 512
        draw.rectangle((x, y, x+511, y+70), fill=accent)
        draw.text((x+32, y+25), 'DAILY / HOME', fill='white', font=fonts[20])
        draw.text((x+32, y+118), title, fill=ink, font=fonts[48])
        draw.text((x+32, y+181), subtitle, fill=ink, font=fonts[48])
        draw.line((x+32, y+259, x+480, y+259), fill=accent, width=4)
        draw.text((x+32, y+282), detail, fill=ink, font=fonts[15])
        draw.text((x+32, y+325), 'Keep dry. Store neatly.', fill=ink, font=fonts[20])
        # Printed fold and seal marks stay inside the unchanged rigid envelope.
        for offset in (5, 10, 15):
            draw.line((x+offset, y+370, x+offset, y+505), fill=(211, 207, 195))
        draw.text((x+32, y+433), 'HOUSEHOLD ESSENTIALS', fill=accent, font=fonts[24])
        draw.text((x+32, y+472), name.replace('_', ' ').upper(), fill=ink, font=fonts[15])
    image.save(path)
    return path


def add_wrapper(spec, body, collision, directory):
    """Add UV visuals only: the box, mass, inertia and contact parameters are retained."""
    import mujoco as mj

    size = np.asarray(collision.size, dtype=float)
    if (collision.type != mj.mjtGeom.mjGEOM_BOX or size.shape != (3,)
            or not np.isfinite(size).all() or np.any(size <= 0)):
        raise ValueError('wrapper requires a positive box envelope')
    name = collision.name
    texture = wrapper_texture(directory, name, collision.rgba)
    vertices, faces, uv, normals = [], [], [], []
    for axis in range(3):
        for sign in (-1, 1):
            face = len(vertices) // 4
            normal = np.eye(3)[axis] * sign
            up = np.eye(3)[2 if axis != 2 else 0]
            right = np.cross(up, normal)
            points = normal * size + np.array([-right-up, right-up, right+up, -right+up]) * size
            start = len(vertices)
            vertices.extend(points)
            faces.extend([[start, start+1, start+2], [start, start+2, start+3]])
            normals.extend([normal] * 4)
            # Pixel padding prevents the adjacent atlas panel bleeding at a seam.
            uv.extend((np.array([[2, 510], [510, 510], [510, 2], [2, 2]])
                       + [face % 3 * 512, face // 3 * 512]) / [1536, 1024])
    spec.add_texture(name=name+'_print', type=mj.mjtTexture.mjTEXTURE_2D,
                     file=str(texture.resolve()), embed=mj.mjtEmbed.mjEMBED_FALSE)
    slots = [''] * int(mj.mjtTextureRole.mjNTEXROLE)
    slots[int(mj.mjtTextureRole.mjTEXROLE_RGB)] = name+'_print'
    spec.add_material(name=name+'_paper', textures=slots, rgba=[1, 1, 1, 1],
                      roughness=.72, metallic=0, specular=.12, shininess=.08)
    spec.add_mesh(name=name+'_wrapper', uservert=np.asarray(vertices).ravel(),
                  usernormal=np.asarray(normals).ravel(), userface=np.asarray(faces).ravel(),
                  usertexcoord=np.asarray(uv).ravel(), inertia=mj.mjtMeshInertia.mjMESH_INERTIA_SHELL)
    body.add_geom(name=name+'_visual', type=mj.mjtGeom.mjGEOM_MESH, meshname=name+'_wrapper',
                  material=name+'_paper', pos=collision.pos, quat=collision.quat,
                  contype=0, conaffinity=0, density=0, group=1)
    collision.group = 3
    return dict(texture=str(texture), texture_sha256=hashlib.sha256(texture.read_bytes()).hexdigest(),
                font_sha256=hashlib.sha256(FONT.read_bytes()).hexdigest(),
                geometry='exact box envelope; rigid sealed wrapper, no flexible material simulation')
