from dataclasses import dataclass

import bpy
from mathutils import Vector

OBJ_NAME = "brick"
COLLECTION_NAME = "bricks"


# Function to delete an object by name
def delete_object_by_name(obj_name, collection=None):
    # Try to get the object by name
    obj = bpy.data.objects.get(obj_name)
    if obj:
        # If a specific collection is given, unlink the object from that collection
        if collection and obj.name in collection.objects:
            collection.objects.unlink(obj)

        # Delete the object
        bpy.data.objects.remove(obj)


def delete_objects_in_collection(collection_name):
    collection = bpy.data.collections.get(collection_name)
    if collection:
        # Iterate over the objects in the collection and delete them
        for obj in collection.objects:
            bpy.data.objects.remove(obj, do_unlink=True)


def get_collection(collection_name):
    if collection_name in bpy.data.collections:
        return bpy.data.collections[collection_name]
    else:
        return bpy.data.collections.new(collection_name)


def init():
    bricks = get_collection(COLLECTION_NAME)
    if bricks.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(bricks)

    delete_objects_in_collection(COLLECTION_NAME)


#    # This line ensures that the script can run without error if there's already a cube in the scene.
#    bpy.ops.object.select_all(action='DESELECT')
#    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))

#    # Optionally, you can rename the cube
#    cube = bpy.context.active_object
#    cube.name = OBJ_NAME
#    # Deselect all objects first
#    bpy.ops.object.select_all(action='DESELECT')

#    # Access the object by its name and select it
#    cube = bpy.data.objects.get(OBJ_NAME)
#    if cube:
#        cube.select_set(True)

#        # Make the cube the active object
#        bpy.context.view_layer.objects.active = cube
init()

input_file = bpy.data.texts["2023-22.input"].as_string().splitlines()

CONTROL_1 = """\
1,0,1~1,2,1
0,0,2~2,0,2
0,2,3~2,2,3
0,0,4~0,2,4
2,0,5~2,2,5
0,1,6~2,1,6
1,1,8~1,1,9
""".splitlines()


class Brick:
    start: Vector
    end: Vector

    def __init__(self, start: Vector, end: Vector):
        self.start = start
        self.end = end
        bpy.ops.mesh.primitive_cube_add(size=2, location=(1, 1, 1))
        brick = bpy.context.active_object
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
        width = end.x - start.x + 1
        length = end.y - start.y + 1
        height = end.z - start.z + 1
        brick.scale = Vector((width / 2, length / 2, height / 2))
        brick.name = "brick"
        brick.location = start - Vector((0, 0, 1))
        self.obj = brick


def parse_line(line):
    start, end = line.split("~", 1)
    start = Vector(tuple(int(x) for x in start.split(",", 2)))
    end = Vector(tuple(int(x) for x in end.split(",", 2)))
    return start, end


brick_pos = [parse_line(line) for line in input_file]
bricks = [Brick(start, end) for start, end in brick_pos]

max_z = max(end.z for start, end in brick_pos)
# bpy.ops.object.empty_add(type="SINGLE_ARROW", location=(0, 0, max_z))
levels = [set() for _ in range(max_z)]
