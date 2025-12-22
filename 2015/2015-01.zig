const std = @import("std");

fn part1(input: []const u8) i64 {
    var count: i64 = 0;
    for (input) |c| {
        if (c == '(') {
            count += 1;
        } else if (c == ')') {
            count -= 1;
        }
    }
    return count;
}

fn part2(input: []const u8) usize {
    var count: i64 = 0;
    for (input, 0..) |c, i| {
        if (c == '(') {
            count += 1;
        } else if (c == ')') {
            count -= 1;
        }
        if (count == -1) {
            return i + 1;
        }
    }
    return 0;
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    const size = (try std.fs.cwd().statFile("2015-01.input")).size;
    const input = try std.fs.cwd().readFileAlloc(allocator, "2015-01.input", size);

    var timer = try std.time.Timer.start();
    const p1 = part1(input);
    std.debug.print("part 1: {} in {} μs\n", .{ p1, timer.read() / 1000 });
    timer.reset();
    const p2 = part2(input);
    std.debug.print("part 2: {} in {} μs\n", .{ p2, timer.read() / 1000 });
}

test "part 1" {
    try std.testing.expectEqual(part1(")())())"), -3);
}

test "part 2" {
    try std.testing.expectEqual(part2("()())"), 5);
}
