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
- **Day 9**: Sweep line algorithm with and without an interval tree.
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
| 2025 |  1  |  1   | python | 784.62 µs |
| 2025 |  1  |  2   | python |   1.19 ms |
| 2025 |  1  |  1   | c      |  96.00 µs |
| 2025 |  1  |  2   | c      | 102.38 µs |
| 2025 |  2  |  1   | python |  51.48 µs |
| 2025 |  2  |  2   | python | 233.32 µs |
| 2025 |  3  |  1   | python |   2.18 ms |
| 2025 |  3  |  2   | python |   3.55 ms |
| 2025 |  4  |  1   | python |  46.24 ms |
| 2025 |  4  |  2   | python |  93.79 ms |
| 2025 |  5  |  1   | python |   8.16 ms |
| 2025 |  5  |  2   | python | 322.42 µs |
| 2025 |  6  |  1   | python |   1.51 ms |
| 2025 |  6  |  2   | python |   2.38 ms |
| 2025 |  7  |  1   | python |   2.17 ms |
| 2025 |  7  |  2   | python |   5.25 ms |
| 2025 |  8  |  1   | python |  41.85 ms |
| 2025 |  8  |  2   | python |  57.91 ms |
| 2025 |  9  |  1   | python |  27.14 ms |
| 2025 |  9  |  2   | python |  20.13 ms |
| 2025 | 10  |  1   | python |  57.27 ms |
| 2025 | 10  |  2   | python |  31.62 ms |
| 2025 | 11  |  1   | python | 366.63 µs |
| 2025 | 11  |  2   | python |   1.67 ms |
| 2025 | 12  |  1   | python | 144.15 ms |

**Overall Statistics:**

- Total time: 550.13 ms
- Average: 22.01 ms
- Min: 51.48 µs
- Max: 144.15 ms
- Count: 25 parts

**By Language:**

- **c**: Total 198.38 µs, Avg 99.19 µs, Min 96.00 µs, Max 102.38 µs, Count 2
- **python**: Total 549.93 ms, Avg 23.91 ms, Min 51.48 µs, Max 144.15 ms, Count 23
