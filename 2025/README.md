# 2025

## Notes Features

The extra challenge for myself this year was to solve everything in
python using only stdlib in as short a runtime as possible.

I've gone back in some cases and redone some problems in different
languages to learn/compare, but python is first.

- **Day 1**: `divmod` vs % in python.
- **Day 2**: Wrote a few different approaches. Ended up sitting down
  with pen and paper and mathing out an optimal solution. Computes
  [1, 10<sup>4000</sup>] in ~1.37s.
- **Day 8**: kd-trees and Prim's algorithm.
- **Day 10**: Ended up coding a full linear integer programming
  solution. Steps:
  1. Gaussian elimination to find independent variables and define the
     inequalities. If system is over-defined, return answer.
  1. Create a tableau from inequalities
     1. Run dual simplex to ensure tableau is feasible
     1. Run primal simplex to ensure tableau is optimal
     1. Loop until feasible _and_ optimal
     1. If all variables and solution are integral, return.
  1. Do branch and bound, adding constraints, to find an integral solution.

## Stats

| Year | Day | Part | Lang   |      Time |
| :--: | :-: | :--: | ------ | --------: |
| 2025 |  1  |  1   | python | 808.04 µs |
| 2025 |  1  |  2   | python |   1.17 ms |
| 2025 |  1  |  1   | c      | 111.27 µs |
| 2025 |  1  |  2   | c      | 103.73 µs |
| 2025 |  2  |  1   | python |  51.45 µs |
| 2025 |  2  |  2   | python | 233.28 µs |
| 2025 |  3  |  1   | python |   2.19 ms |
| 2025 |  3  |  2   | python |   3.77 ms |
| 2025 |  4  |  1   | python |  46.67 ms |
| 2025 |  4  |  2   | python |  89.73 ms |
| 2025 |  5  |  1   | python |   7.88 ms |
| 2025 |  5  |  2   | python | 319.15 µs |
| 2025 |  6  |  1   | python |   1.52 ms |
| 2025 |  6  |  2   | python |   2.43 ms |
| 2025 |  7  |  1   | python |   2.21 ms |
| 2025 |  7  |  2   | python |   5.37 ms |
| 2025 |  8  |  1   | python |  41.91 ms |
| 2025 |  8  |  2   | python |  58.19 ms |
| 2025 |  9  |  1   | python |  26.96 ms |
| 2025 |  9  |  2   | python |    2.04 s |
| 2025 | 10  |  1   | python |  53.76 ms |
| 2025 | 10  |  2   | python |  29.59 ms |
| 2025 | 11  |  1   | python | 359.75 µs |
| 2025 | 11  |  2   | python |   1.65 ms |
| 2025 | 12  |  1   | python | 161.34 ms |

**Overall Statistics:**

- Total time: 3.50 s
- Average: 139.95 ms
- Min: 51.45 µs
- Max: 2.93 s
- Count: 25 parts

**By Language:**

- **c**: Total 215.00 µs, Avg 107.50 µs, Min 103.73 µs, Max 111.27 µs, Count 2
- **python**: Total 3.50 s, Avg 152.11 ms, Min 51.45 µs, Max 2.93 s, Count 23
