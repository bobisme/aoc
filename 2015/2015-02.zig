const std = @import("std");

const Box = struct {
    l: u64,
    w: u64,
    h: u64,

    fn surfaceArea(self: Box) u64 {
        return 2 * self.l * self.w + 2 * self.w * self.h + 2 * self.h * self.l;
    }

    fn sideAreas(self: Box) [3]u64 {
        return [_]u64{ self.l * self.w, self.w * self.h, self.h * self.l };
    }

    fn perims(self: Box) [3]u64 {
        return [_]u64{ 2 * self.l + 2 * self.w, 2 * self.w + 2 * self.h, 2 * self.h + 2 * self.l };
    }

    fn vol(self: Box) u64 {
        return self.l * self.w * self.h;
    }
};

fn parse(allocator: std.mem.Allocator, input: []const u8) !std.ArrayList(Box) {
    var boxes = try std.ArrayList(Box).initCapacity(allocator, 0);
    errdefer boxes.deinit(allocator);

    var lines = std.mem.tokenizeScalar(u8, input, '\n');

    while (lines.next()) |line| {
        var parts = std.mem.tokenizeScalar(u8, line, 'x');
        const box = Box{
            .l = try std.fmt.parseInt(u64, parts.next().?, 10),
            .w = try std.fmt.parseInt(u64, parts.next().?, 10),
            .h = try std.fmt.parseInt(u64, parts.next().?, 10),
        };
        try boxes.append(allocator, box);
    }
    return boxes;
}

fn part1(allocator: std.mem.Allocator, input: []const u8) !u64 {
    const boxes = try parse(allocator, input);

    var out: u64 = 0;
    for (boxes.items) |box| {
        const area = box.surfaceArea();
        const sides = box.sideAreas();
        const min = @min(sides[0], sides[1], sides[2]);
        out += area + min;
    }
    return out;
}

fn part2(allocator: std.mem.Allocator, input: []const u8) !u64 {
    const boxes = try parse(allocator, input);

    var out: u64 = 0;
    for (boxes.items) |box| {
        const perims = box.perims();
        const minPerim = @min(perims[0], perims[1], perims[2]);
        out += minPerim + box.vol();
    }
    return out;
}

pub fn main() !void {
    // 135-140 μs
    // var dbgalloc = std.heap.DebugAllocator(.{}){};
    // const allocator = dbgalloc.allocator();
    // defer {
    //     // Detect and report memory leaks
    //     const leaked = dbgalloc.deinit();
    //     if (leaked == .leak) {
    //         std.debug.print("Memory leak detected!\n", .{});
    //     } else {
    //         std.debug.print("No memory leaks detected.\n", .{});
    //     }
    // }

    // 58-62 μs
    // const allocator = std.heap.page_allocator;

    // 41-45 μs
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    const path = "2015-02.input";
    const size = (try std.fs.cwd().statFile(path)).size;
    const input = try std.fs.cwd().readFileAlloc(allocator, path, size);
    defer allocator.free(input);

    var stdoutBuffer: [0x100]u8 = undefined;
    var stdout = std.fs.File.stdout().writer(&stdoutBuffer);
    defer stdout.interface.flush() catch |err| {
        std.debug.panic("failed to flush: {}", .{err});
    };

    var timer = try std.time.Timer.start();
    const p1 = try part1(allocator, input);
    try stdout.interface.print("2015\t2\t1\t{}\t{}\n", .{ p1, timer.read() });
    timer.reset();
    const p2 = try part2(allocator, input);
    try stdout.interface.print("2015\t2\t2\t{}\t{}\n", .{ p2, timer.read() });
}

test "part 1" {
    try std.testing.expectEqual(part1(")())())"), -3);
}

test "part 2" {
    try std.testing.expectEqual(part2("()())"), 5);
}
