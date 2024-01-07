from dataclasses import dataclass
from typing import Callable, Iterator, NamedTuple
from itertools import chain
from functools import cache
import time
from pprint import pp
import heapq

import bmesh
import bpy
from mathutils import Vector

OBJ_NAME = "brick"
COLLECTION_NAME = "bricks"
FOCUS_MAT_1 = bpy.data.materials.get("focus1")
FOCUS_MAT_2 = bpy.data.materials.get("focus2")
FOCUS_MAT_3 = bpy.data.materials.get("focus3")
FOCUS_MAT_4 = bpy.data.materials.get("focus4")
DEFAULT_MAT = bpy.data.materials.get("brick")

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
            if obj.data.materials:
                obj.data.materials[0] = None
            bpy.data.objects.remove(obj, do_unlink=True)


def get_collection(collection_name):
    if collection_name in bpy.data.collections:
        return bpy.data.collections[collection_name]
    else:
        return bpy.data.collections.new(collection_name)


def init():
    bpy.context.preferences.edit.use_global_undo = False
    bricks = get_collection(COLLECTION_NAME)
    if bricks.name not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(bricks)
    delete_objects_in_collection(COLLECTION_NAME)


Vec = NamedTuple("Vec", [("x", int), ("y", int), ("z", int)])
Vec.__sub__ = lambda self, other: Vec(
    self.x - other.x, self.y - other.y, self.z - other.z
)
Vec.__add__ = lambda self, other: Vec(
    self.x + other.x, self.y + other.y, self.z + other.z
)


class Brick:
    id_: int
    start: Vector
    end: Vector
    at_rest: bool
    obj = None

    def __init__(self, id: int, start: Vec, end: Vec):
        self.id_ = id
        self.start = start
        self.end = end
        self.at_rest = False
        self.can_destroy = False

    def __repr__(self):
        return (
            f"[{self.id_}: {'rested' if self.at_rest else ''} {self.start} {self.end}]"
        )

    def __hash__(self):
        return hash(self.id_)

    def levels(self) -> list[int]:
        return [x for x in range(int(self.start.z) - 1, int(self.end.z))]

    def level_above(self) -> int:
        return self.levels()[-1] + 1

    def level_below(self) -> int:
        return self.levels()[0] - 1

    def positions(self) -> Iterator[Vec]:
        for x in range(self.start.x, self.end.x + 1):
            for y in range(self.start.y, self.end.y + 1):
                for z in range(self.start.z, self.end.z + 1):
                    yield Vec(x, y, z)

    def bottom_positions(self) -> Iterator[Vec]:
        for x in range(self.start.x, self.end.x + 1):
            for y in range(self.start.y, self.end.y + 1):
                yield Vec(x, y, self.start.z)

    def top_positions(self) -> Iterator[Vec]:
        for x in range(self.start.x, self.end.x + 1):
            for y in range(self.start.y, self.end.y + 1):
                yield Vec(x, y, self.end.z)

    def move_down(self, level_map):
        current_levels = self.levels()
        for lvl in current_levels:
            level_map[lvl].remove(self)
        self.start = self.start - Vec(0, 0, 1)
        self.end = self.end - Vec(0, 0, 1)
        if self.obj:
            self.obj.location = self.location()
        new_levels = self.levels()
        for lvl in new_levels:
            level_map[lvl].add(self)

    def location(self):
        return Vector(self.start) - Vector((0, 0, 1))

    def mesh_faces(self):
        start = self.start
        end = self.end
        dx = abs(end.x - start.x) + 1
        dy = abs(end.y - start.y) + 1
        dz = abs(end.z - start.z) + 1
        return [
            [(0, 0, 0), (0, dy, 0), (dx, dy, 0), (dx, 0, 0)],
            [(0, 0, 0), (0, 0, dz), (dx, 0, dz), (dx, 0, 0)],
            [(0, 0, dz), (0, dy, dz), (dx, dy, dz), (dx, 0, dz)],
            [(dx, 0, 0), (dx, dy, 0), (dx, dy, dz), (dx, 0, dz)],
            [(0, dy, 0), (dx, dy, 0), (dx, dy, dz), (0, dy, dz)],
            [(0, 0, 0), (0, dy, 0), (0, dy, dz), (0, 0, dz)],
        ]

    def spawn(self, scene):
        collection = get_collection(COLLECTION_NAME)
        bm = bmesh.new()
        # vertlist = [[(0, 0), (0, 1), (1, 1), (1, 0)], [(2, 2), (2, 3), (3, 3), (3, 2)]]
        faces = [self.start]
        for faceverts in self.mesh_faces():
            bm_verts = []
            for vert in faceverts:
                bm_verts.append(bm.verts.new((vert[0], vert[1], vert[2])))
            bm.faces.new(bm_verts)

        mesh = bpy.data.meshes.new(name="brickmesh")
        brick = bpy.data.objects.new(name=f"brick_{self.id_:>04}", object_data=mesh)
        bm.to_mesh(brick.data)
        collection.objects.link(brick)
        brick.location = self.location()
        return brick

    def clear_mat(self):
        if self.obj is None:
            return
        if self.obj.data.materials:
            self.obj.data.materials[0] = None

    def set_mat(self, mat):
        if self.obj is None:
            return
        if self.obj.data.materials:
            self.obj.data.materials[0] = mat
        else:
            self.obj.data.materials.append(mat)


def parse_line(line):
    start, end = line.split("~", 1)
    start = Vec(*(int(x) for x in start.split(",", 2)))
    end = Vec(*(int(x) for x in end.split(",", 2)))
    return start, end


all_bricks_spawned = [False]


def get_level_above(brick: Brick) -> int:
    return max(brick.levels()) + 1


def get_level_below(brick: Brick) -> int:
    return min(brick.levels()) - 1


def get_supported(brick: Brick, level_map) -> Iterator[Brick]:
    """Bricks above the current one, supported by input brick."""
    upper_lvl = get_level_above(brick)
    if upper_lvl >= len(level_map):
        return []
    upper_bricks = level_map[upper_lvl]
    collision_positions = set(p + Vec(0, 0, 1) for p in brick.top_positions())
    for upper in upper_bricks:
        bottoms = set(upper.bottom_positions())
        if bottoms & collision_positions:
            yield upper


def get_supporting(brick: Brick, level_map) -> Iterator[Brick]:
    """Bricks below the current one, supporting by input brick."""
    lower_lvl = get_level_below(brick)
    if lower_lvl < 0:
        return []
    lower_bricks = level_map[lower_lvl]
    collision_positions = set(p - Vec(0, 0, 1) for p in brick.bottom_positions())
    for lower in lower_bricks:
        tops = set(lower.top_positions())
        if tops & collision_positions:
            yield lower


def exclusively_supported(brick: Brick, level_map: list[set[Brick]]) -> Iterator[Brick]:
    supported = list(get_supported(brick, level_map))
    if not supported:
        return
    # print(f"brick {brick.id} supports {[b.id for b in supported]}")
    for upper in supported:
        upper_supporting = set(x.id_ for x in get_supporting(upper, level_map))
        supported_by_others = upper_supporting - {brick.id_}
        if not supported_by_others:
            yield upper


def count_disintegratable(bricks: list[Brick], level_map: list[set[Brick]]):
    disintegration_count = 0
    for b in bricks:
        supported = list(get_supported(b, level_map))
        if not supported:
            print(f"{b.id_} supports nothing")
            disintegration_count += 1
            continue
        print(f"{b.id_} supports {[x.id_ for x in supported]}")
        n_exclusive_support = len(supported)
        for upper in supported:
            upper_supporting = set(x.id_ for x in get_supporting(upper, level_map))
            if len(upper_supporting - {b.id_}):
                print(f"{upper.id_} supported by other bricks")
                n_exclusive_support -= 1
        if n_exclusive_support <= 0:
            disintegration_count += 1
    print(f"{disintegration_count=}")


_falling_if_destroyed_cache = {}


class Item(NamedTuple("Item", [("brick", Brick)])):
    def __lt__(self, other: tuple[Brick, ...]) -> bool:
        return self.brick.end.z < other[0].end.z


def falling_if_destroyed(brick: Brick, level_map: list[set[Brick]]):
    # print(f"{destroying=}")
    falling = {brick}
    to_check = []
    check_set = set()
    lvl_above = brick.level_above()
    for above in level_map[lvl_above]:
        heapq.heappush(to_check, Item(above))
        check_set.add(above)
    while to_check:
        item = heapq.heappop(to_check)
        b = item.brick
        check_set.remove(b)
        supporting = set(get_supporting(b, level_map))
        if b.start.z <= 1:
            continue
        if supporting - falling:
            continue  # other bricks holding this up
        falling.add(b)
        lvl_above = b.level_above()
        for above in level_map[lvl_above]:
            if above not in falling and above not in check_set:
                heapq.heappush(to_check, Item(above))
                check_set.add(above)
    # print(f"{falling=}")
    if not falling:
        return set()
    falling.remove(brick)
    return falling


def focus_brick(bricks: list[Brick], level_map: list[set[Brick]], idx: int):
    brick = bricks_dict[idx]
    print(f"focusing brick {brick}")
    for b in bricks:
        b.set_mat(DEFAULT_MAT)
    brick.set_mat(FOCUS_MAT_2)
    falling_set = falling_if_destroyed(brick, level_map)
    for b in falling_set:
        b.set_mat(FOCUS_MAT_1)


def count_falling(bricks: list[Brick], level_map: list[set[Brick]]):
    print("counting falling for each brick")
    counts = [0 for _ in bricks]

    bpy.context.preferences.edit.use_global_undo = True
    for b in bricks:
        falling_set = falling_if_destroyed(b, level_map)
        counts[b.id_] = len(falling_set)

    # pp(counts)
    print(f"{sum(counts)=}")


def settle_bricks(
    bricks: list[Brick], level_map: list[set[Brick]], next_stage=None, index=[0]
):
    BATCH_SIZE = 40
    if index[0] < len(bricks):
        i = index[0]
        batch = bricks[i : i + BATCH_SIZE]
        index[0] += BATCH_SIZE
        for b in batch:
            if b.start.z == 0 or b.end.z == 0:
                b.at_rest = True
                return 0.001
            while not b.at_rest:
                lower_lvl = min(b.levels()) - 1
                if lower_lvl < 0:
                    b.at_rest = True
                    break
                collision_positions = set(
                    p - Vec(0, 0, 1) for p in b.bottom_positions()
                )
                potential_collisions = set(
                    flatten(
                        lower_brick.top_positions()
                        for lower_brick in level_map[lower_lvl]
                        if lower_brick.at_rest
                    )
                )
                colliding = potential_collisions & collision_positions
                if colliding:
                    b.at_rest = True
                else:
                    b.move_down(level_map)
        return 0.01
    if next_stage is not None:
        return next_stage(bricks, level_map)


def add_bricks(scene, bricks: list[Brick], level_map, next_stage=None, index=[0]):
    BATCH_SIZE = 40
    if index[0] < len(bricks):
        i = index[0]
        batch = bricks[i : i + BATCH_SIZE]
        for brick in batch:
            brick.obj = brick.spawn(scene)
        index[0] += BATCH_SIZE
        return 0.01  # Wait x seconds before adding the next object
    bpy.app.timers.register(lambda: settle_bricks(bricks, level_map, next_stage))


def flatten(list_of_lists):
    "Flatten one level of nesting."
    return chain.from_iterable(list_of_lists)


def part_1(input):
    init()
    brick_pos = [parse_line(line) for line in input]
    bricks = [Brick(i, start, end) for i, (start, end) in enumerate(brick_pos)]

    max_z = max(end.z for _, end in brick_pos)
    level_map = [set() for _ in range(int(max_z))]
    for b in bricks:
        for lvl in b.levels():
            level_map[lvl].add(b)
    bricks.sort(key=lambda x: x.start.z)
    # spawn bricks
    bpy.context.preferences.edit.use_global_undo = False
    bpy.app.timers.register(
        lambda: add_bricks(
            bpy.context.scene, bricks, level_map, next_stage=count_falling
        )
    )


bricks_dict = {}
level_map = []


def part_2(input):
    global bricks, bricks_dict, level_map
    init()
    brick_pos = [parse_line(line) for line in input]
    bricks = [Brick(i, start, end) for i, (start, end) in enumerate(brick_pos)]
    bricks_dict = {b.id_: b for b in bricks}

    max_z = max(end.z for _, end in brick_pos)
    level_map = [set() for _ in range(int(max_z))]
    for b in bricks:
        for lvl in b.levels():
            level_map[lvl].add(b)
    bricks.sort(key=lambda x: x.start.z)
    # spawn bricks
    bpy.context.preferences.edit.use_global_undo = False
    bpy.app.timers.register(
        lambda: add_bricks(
            bpy.context.scene, bricks, level_map, next_stage=count_falling
        )
    )


# 43,080 is too low
print("-" * 60)
part_2(input_file)


class SimpleOperator(bpy.types.Operator):
    bl_idname = "object.simple_operator"
    bl_label = "Focus Brick"

    def execute(self, context):
        global bricks, level_map
        # brick_id = context.scene.brick_id
        # bpy.app.timers.register(lambda: focus_brick(bricks, level_map, brick_id))
        # print("Brick ID:", brick_id)
        return {"FINISHED"}


class SimplePanel(bpy.types.Panel):
    bl_label = "Simple Panel"
    bl_idname = "_PT_SimplePanel"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "AoC"

    def draw(self, context):
        layout = self.layout
        layout.prop(context.scene, "brick_id")
        layout.prop(context.scene, "support_id")
        layout.operator("object.simple_operator")


def show_support(bricks: list[Brick], level_map, brick_id: int):
    for b in bricks:
        b.set_mat(DEFAULT_MAT)
    brick = bricks_dict[brick_id]
    brick.set_mat(FOCUS_MAT_2)
    supporting = set(get_supporting(brick, level_map))
    supported = set(get_supported(brick, level_map))
    for b in supporting:
        b.set_mat(FOCUS_MAT_3)
    for b in supported:
        b.set_mat(FOCUS_MAT_4)
    print(f"{brick_id} supports {len(supported)} and is supported by {len(supporting)}")


def register():
    bpy.types.Scene.brick_id = bpy.props.IntProperty(
        name="Brick ID",
        description="Select brick",
        default=0,
        min=0,
        max=len(input_file),
        update=lambda _, ctx: focus_brick(bricks, level_map, ctx.scene.brick_id),
    )
    bpy.types.Scene.support_id = bpy.props.IntProperty(
        name="Support Brick ID",
        description="Select brick",
        default=0,
        min=0,
        max=len(input_file),
        update=lambda _, ctx: show_support(bricks, level_map, ctx.scene.support_id),
    )
    bpy.utils.register_class(SimpleOperator)
    bpy.utils.register_class(SimplePanel)


def unregister():
    bpy.utils.unregister_class(SimplePanel)
    bpy.utils.unregister_class(SimpleOperator)
    del bpy.types.Scene.brick_id
    del bpy.types.Scene.support_id


if __name__ == "__main__":
    register()
