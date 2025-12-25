// clang -std=c23 -I../libb/src ../libb/src/*.c -o 2025-01 2025-01.c &&
// ./2025-01

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

#include "libb.h"

typedef Vec(i64) Turns;

void print_turns(Turns turns) {
  for (size_t i = 0; i < turns.n; i++) {
    printf("%ld\n", turns.p[i]);
  }
}

Turns parse(Arena *a, Str input) {
  Turns v = {};
  vec_init(&v, a, 200);
  auto it = str_split_lines(input);
  Str s;
  while (str_iter_split_next(&it, &s)) {
    if (s.n == 0)
      break;

    auto res = str_to_i64(str_slice(s, 1, s.n - 1));
    assert(res.ok);
    i64 n = res.val;
    if (s.p[0] == 'L') {
      n *= -1;
    }
    vec_push(&v, n);
  }
  return v;
}

i64 part_1(Arena *a, Str input) {
  auto turns = parse(a, input);
  i64 pos = 50;
  i64 cross_counts = 0;
  for (size_t i = 0; i < turns.n; i++) {
    i64 turn = turns.p[i];
    pos = (pos + turn) % 100;
    if (pos == 0) {
      cross_counts += 1;
    }
  }

  return cross_counts;
}

i64 part_2(Arena *a, Str input) {
  auto turns = parse(a, input);

  i64 pos = 50;
  i64 cross_counts = 0;

  for (size_t i = 0; i < turns.n; i++) {
    i64 turn = turns.p[i];

    auto q = div(pos + turn, 100);
    auto n_turns = q.quot;
    auto next_pos = q.rem;
    if (next_pos < 0) {
      next_pos = 100 + next_pos;
      n_turns -= 1;
    }

    if (turn >= 0) {
      cross_counts += n_turns;
      pos = next_pos;
    } else {
      cross_counts += abs(n_turns);
      if (next_pos == 0) {
        cross_counts += 1;
      }
      if (pos == 0) {
        cross_counts -= 1;
      }
      pos = next_pos;
    }
  }
  return cross_counts;
}

int main(void) {
  Arena a = {};
  arena_init(&a, 10 * 1024 * 1024);
  FReadResult res = fs_read_file(&a, "./2025-01.input");
  if (!res.ok) {
    printf("Failed to read file: %u\n", res.err);
    return 1;
  };
  assert(res.ok);
  auto content = res.val;

  auto start = now_ns();
  auto p1 = part_1(&a, content);
  printf("2025\t1\t1\t%ld\t%ld\n", p1, (now_ns() - start));

  start = now_ns();
  auto p2 = part_2(&a, content);
  printf("2025\t1\t2\t%ld\t%ld\n", p2, (now_ns() - start));
}
