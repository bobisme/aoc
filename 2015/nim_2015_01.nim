import std/math
# import std/sequtils
import std/sugar
import std/strutils
import std/strformat
import std/monotimes, std/times

proc part_1(input: seq[string]): int =
  # return sum(input[0].mapIt(if it == '(': 1 else: -1))
  # but this is a little faster:
  result = 0
  for c in input[0]:
    if c == '(':
      result += 1
    else:
      result -= 1

proc part_2(input: seq[string]): int =
  let line = input[0]
  var s = 0
  for i in 0..(line.len() - 1):
    s += (if line[i] == '(': 1 else: -1)
    if s == -1:
      return i + 1

proc run[T](fn: proc(): T, year = 2015, day = 1, part = 0): T =
  let start = getMonoTime()
  result = fn()
  let elapsed = (getMonoTime() - start).inNanoseconds
  echo &"{year}\t{day}\t{part}\t{result}\t{elapsed}"

when isMainModule:
  assert part_1(@[")())())"]) == -3
  assert part_2(@["()())"]) == 5

  let input = readFile("2015-01.input").splitLines()
  discard run(() => part_1(input), part=1)
  discard run(() => part_2(input), part=2)
