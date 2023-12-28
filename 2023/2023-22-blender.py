from dataclasses import dataclass
from typing import Iterator, NamedTuple
from itertools import chain
import time

import bpy
from mathutils import Vector

OBJ_NAME = "brick"
COLLECTION_NAME = "bricks"

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


Vec = NamedTuple("Vec", [("x", int), ("y", int), ("z", int)])
Vec.__sub__ = lambda self, other: Vec(
    self.x - other.x, self.y - other.y, self.z - other.z
)
Vec.__add__ = lambda self, other: Vec(
    self.x + other.x, self.y + other.y, self.z + other.z
)


class Brick:
    id: int
    start: Vector
    end: Vector
    at_rest: bool
    obj = None

    def __init__(self, id: int, start: Vec, end: Vec):
        self.id = id
        self.start = start
        self.end = end
        self.at_rest = False
        self.can_destroy = False

    def __repr__(self):
        return (
            f"[{self.id}: {'rested' if self.at_rest else ''} {self.start} {self.end}]"
        )

    def __hash__(self):
        return hash(self.id)

    def levels(self) -> list[int]:
        return [x for x in range(int(self.start.z) - 1, int(self.end.z))]

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

    def spawn(self):
        bpy.ops.mesh.primitive_cube_add(size=2, location=(1, 1, 1))
        brick = bpy.context.active_object
        bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
        width = self.end.x - self.start.x + 1
        length = self.end.y - self.start.y + 1
        height = self.end.z - self.start.z + 1
        brick.scale = Vector((width / 2, length / 2, height / 2))
        brick.name = f"brick_{self.id:>04}"
        brick.location = self.location()
        return brick


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
    supported = set(get_supported(brick, level_map))
    if not supported:
        return
    for upper in supported:
        upper_supporting = set(x.id for x in get_supporting(upper, level_map))
        supported_by_others = upper_supporting - {brick.id}
        if not supported_by_others:
            yield upper


def settle_bricks(bricks: list[Brick], level_map, index=[0]):
    BATCH_SIZE = 1
    if index[0] < len(bricks):
        i = index[0]
        batch = bricks[i : i + BATCH_SIZE]
        index[0] += BATCH_SIZE
        for b in batch:
            if b.start.z == 1 or b.end.z == 1:
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
    disintegration_count = 0
    for b in bricks:
        supported = list(get_supported(b, level_map))
        if not supported:
            print(f"{b.id} supports nothing")
            disintegration_count += 1
            continue
        print(f"{b.id} supports {[x.id for x in supported]}")
        n_exclusive_support = len(supported)
        for upper in supported:
            upper_supporting = set(x.id for x in get_supporting(upper, level_map))
            # print(f"{upper.id} supported by {upper_supporting}")
            if len(upper_supporting - {b.id}):
                print(f"{upper.id} supported by other bricks")
                n_exclusive_support -= 1
        if n_exclusive_support <= 0:
            disintegration_count += 1
    print(f"{disintegration_count=}")


def add_bricks(scene, bricks: list[Brick], level_map, index=[0]):
    BATCH_SIZE = 10
    if index[0] < len(bricks):
        i = index[0]
        batch = bricks[i : i + BATCH_SIZE]
        for brick in batch:
            brick.obj = brick.spawn()
        index[0] += BATCH_SIZE
        return 0.1  # Wait x seconds before adding the next object
    # settle_bricks(bricks, level_map)
    bpy.app.timers.register(lambda: settle_bricks(bricks, level_map))


def flatten(list_of_lists):
    "Flatten one level of nesting."
    return chain.from_iterable(list_of_lists)


def part_1(input):
    init()
    brick_pos = [parse_line(line) for line in input]
    bricks = [Brick(i, start, end) for i, (start, end) in enumerate(brick_pos)]

    # while not all_bricks_spawned[0]:
    #     print("waiting for bricks to spawn")
    #     time.sleep(1)

    max_z = max(end.z for start, end in brick_pos)
    # bpy.ops.object.empty_add(type="SINGLE_ARROW", location=(0, 0, max_z))
    level_map = [set() for _ in range(int(max_z))]
    for b in bricks:
        for lvl in b.levels():
            level_map[lvl].add(b)
    bricks.sort(key=lambda x: x.start.z)
    # spawn bricks
    bpy.context.preferences.edit.use_global_undo = False
    bpy.app.timers.register(lambda: add_bricks(bpy.context.scene, bricks, level_map))


part_1(input_file)
