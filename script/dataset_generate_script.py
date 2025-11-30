import os
import sys
import math

import bpy
import mathutils


def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def import_and_normalize_obj(obj_path: str):
    bpy.ops.wm.obj_import(filepath=obj_path)
    objs = [o for o in bpy.context.selected_objects if o.type == 'MESH']
    if not objs:
        raise RuntimeError(f"No mesh imported from {obj_path}")

    if len(objs) > 1:
        bpy.ops.object.select_all(action='DESELECT')
        for o in objs:
            o.select_set(True)
        bpy.context.view_layer.objects.active = objs[0]
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
    else:
        obj = objs[0]

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

    coords = [obj.matrix_world @ mathutils.Vector(corner)
              for corner in obj.bound_box]
    min_corner = mathutils.Vector((
        min(c.x for c in coords),
        min(c.y for c in coords),
        min(c.z for c in coords),
    ))
    max_corner = mathutils.Vector((
        max(c.x for c in coords),
        max(c.y for c in coords),
        max(c.z for c in coords),
    ))
    center = (min_corner + max_corner) / 2.0
    bpy.ops.transform.translate(value=(-center.x, -center.y, -center.z))

    size = (max_corner - min_corner).length
    if size > 0:
        s = 2.0 / size
        bpy.ops.transform.resize(value=(s, s, s))

    return obj


def setup_camera_and_lights(img_size=256):
    scene = bpy.context.scene

    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)

    cam.location = (0.0, -3.4, 1.8)
    cam.rotation_euler = (math.radians(65), 0.0, 0.0)
    cam_data.lens = 55.0

    scene.camera = cam
    scene.render.engine = 'BLENDER_EEVEE'
    scene.render.resolution_x = img_size
    scene.render.resolution_y = img_size

    key_data = bpy.data.lights.new(name="KeyLight", type='SUN')
    key_data.energy = 1.2
    key = bpy.data.objects.new("KeyLight", key_data)
    bpy.context.collection.objects.link(key)
    key.location = (4.0, -4.0, 6.0)

    fill_data = bpy.data.lights.new(name="FillLight", type='SUN')
    fill_data.energy = 0.5
    fill = bpy.data.objects.new("FillLight", fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (-4.0, -2.0, 3.0)

    world = scene.world
    world.use_nodes = True
    node_tree = world.node_tree

    bg = node_tree.nodes.get("Background")
    if bg is None:
        bg = node_tree.nodes.new(type="ShaderNodeBackground")
    bg.inputs[0].default_value = (0.15, 0.18, 0.22, 1.0)
    bg.inputs[1].default_value = 1.0

    world_output = None
    for node in node_tree.nodes:
        if node.type == 'OUTPUT_WORLD':
            world_output = node
            break
    if world_output is None:
        world_output = node_tree.nodes.new(type='ShaderNodeOutputWorld')

    for link in list(node_tree.links):
        if link.to_node == world_output and link.to_socket.name == "Surface":
            node_tree.links.remove(link)
    node_tree.links.new(bg.outputs['Background'], world_output.inputs['Surface'])

    return cam


def angle_str(angle: float) -> str:
    s = f"{angle:.4f}"
    return s.rstrip("0").rstrip(".")


def render_yaw_pitch_grid_for_obj(obj_path, out_dir,
                                  yaw_list, pitch_list,
                                  img_size=256):
    root_out = os.path.abspath(out_dir)
    gen_out = os.path.join(root_out, "generated")
    os.makedirs(gen_out, exist_ok=True)

    clear_scene()
    obj = import_and_normalize_obj(obj_path)
    cam = setup_camera_and_lights(img_size=img_size)
    scene = bpy.context.scene

    for yaw_deg in yaw_list:
        for pitch_deg in pitch_list:
            yaw_rad = math.radians(yaw_deg)
            pitch_rad = math.radians(pitch_deg)
            obj.rotation_euler = (pitch_rad, yaw_rad, 0.0)

            y_str = angle_str(yaw_deg)
            p_str = angle_str(pitch_deg)

            img_path = os.path.join(gen_out, f"view_y{y_str}_p{p_str}.png")
            scene.render.filepath = img_path
            bpy.ops.render.render(write_still=True)

            cam_pose_path = os.path.join(gen_out, f"cam_y{y_str}_p{p_str}.txt")
            with open(cam_pose_path, "w") as f:
                for row in cam.matrix_world:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

            obj_pose_path = os.path.join(gen_out, f"obj_y{y_str}_p{p_str}.txt")
            with open(obj_pose_path, "w") as f:
                for row in obj.matrix_world:
                    f.write(" ".join(f"{v:.6f}" for v in row) + "\n")

            print(f"[OK] {obj_path} yaw={y_str}, pitch={p_str} -> {img_path}")


def main():
    argv = sys.argv
    if "--" not in argv:
        raise RuntimeError(
            "Usage: blender -b -P full_pipeline.py -- SHAPENET_ROOT OUTPUT_ROOT [SYNSET_ID] [MAX_OBJS]"
        )
    argv = argv[argv.index("--") + 1:]
    if len(argv) < 2:
        raise RuntimeError(
            "Usage: blender -b -P full_pipeline.py -- SHAPENET_ROOT OUTPUT_ROOT [SYNSET_ID] [MAX_OBJS]"
        )

    shapenet_root = os.path.abspath(argv[0])
    output_root = os.path.abspath(argv[1])

    target_synset = None
    if len(argv) >= 3:
        target_synset = argv[2]
    max_objs = None
    if len(argv) >= 4:
        max_objs = int(argv[3])

    print(f"[INFO] SHAPENET_ROOT={shapenet_root}")
    print(f"[INFO] OUTPUT_ROOT={output_root}")
    print(f"[INFO] target_synset={target_synset}")
    print(f"[INFO] max_objs_per_synset={max_objs}")

    yaw_list = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
    pitch_list = [-20.0, 0.0, 20.0, 40.0]

    print(f"[INFO] yaw_list={yaw_list}")
    print(f"[INFO] pitch_list={pitch_list}")

    synsets = sorted(os.listdir(shapenet_root))

    for syn in synsets:
        if target_synset is not None and syn != target_synset:
            continue

        syn_dir = os.path.join(shapenet_root, syn)
        if not os.path.isdir(syn_dir):
            continue

        print(f"\n=== Processing Synset {syn} ===")
        obj_ids = sorted(os.listdir(syn_dir))
        processed = 0

        for obj_id in obj_ids:
            if max_objs is not None and processed >= max_objs:
                print(f"[INFO] reached max_objs={max_objs} for synset {syn}")
                break

            obj_dir = os.path.join(syn_dir, obj_id)
            models_dir = os.path.join(obj_dir, "models")
            if not os.path.isdir(models_dir):
                continue

            obj_path = os.path.join(models_dir, "model_normalized.obj")
            if not os.path.isfile(obj_path):
                continue

            out_dir = os.path.join(output_root, syn, obj_id)
            gen_dir = os.path.join(out_dir, "generated")
            if os.path.exists(gen_dir):
                print(f"[SKIP] already has {gen_dir}")
                processed += 1
                continue

            try:
                print(f"[RUN] syn={syn}, obj={obj_id}")
                render_yaw_pitch_grid_for_obj(
                    obj_path,
                    out_dir,
                    yaw_list,
                    pitch_list,
                    img_size=256,
                )
                processed += 1
            except Exception as e:
                print(f"[ERROR] Failed on {obj_path}: {e}")

        print(f"[INFO] Synset {syn} processed objects: {processed}")


if __name__ == "__main__":
    main()