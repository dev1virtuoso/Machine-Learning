import bpy

bpy.ops.object.select_all(action='DESELECT')
bpy.ops.object.select_by_type(type='MESH')
bpy.ops.object.delete()

bpy.ops.mesh.primitive_cube_add(size=1, enter_editmode=False, align='WORLD', location=(0, 0, 0))

cube = bpy.context.object
cube.scale = (0.2, 0.2, 0.2)

material = bpy.data.materials.new(name="Identifier Material")
material.diffuse_color = (1, 0, 0)
cube.data.materials.append(material)

bpy.ops.render.render(write_still=True)
