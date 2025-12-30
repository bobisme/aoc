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
- **Day 9**: Sweep line algorithm with an interval tree -- massive speed up.
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
| 2025 |  1  |  1   | python | 778.35 µs |
| 2025 |  1  |  2   | python |   1.16 ms |
| 2025 |  1  |  1   | c      |  94.09 µs |
| 2025 |  1  |  2   | c      | 104.14 µs |
| 2025 |  2  |  1   | python |  51.76 µs |
| 2025 |  2  |  2   | python | 239.14 µs |
| 2025 |  3  |  1   | python |   2.33 ms |
| 2025 |  3  |  2   | python |   3.26 ms |
| 2025 |  4  |  1   | python |  47.03 ms |
| 2025 |  4  |  2   | python |  89.30 ms |
| 2025 |  5  |  1   | python |   7.83 ms |
| 2025 |  5  |  2   | python | 318.27 µs |
| 2025 |  6  |  1   | python |   1.53 ms |
| 2025 |  6  |  2   | python |   2.41 ms |
| 2025 |  7  |  1   | python |   2.21 ms |
| 2025 |  7  |  2   | python |   6.06 ms |
| 2025 |  8  |  1   | python |  41.94 ms |
| 2025 |  8  |  2   | python |  58.72 ms |
| 2025 |  9  |  1   | python |  27.20 ms |
| 2025 |  9  |  2   | python |  16.87 ms |
| 2025 | 10  |  1   | python |  56.98 ms |
| 2025 | 10  |  2   | python |  31.48 ms |
| 2025 | 11  |  1   | python | 373.62 µs |
| 2025 | 11  |  2   | python |   1.67 ms |
| 2025 | 12  |  1   | python | 973.18 µs |

**Overall Statistics:**

- Total time: 400.93 ms
- Average: 16.04 ms
- Min: 51.76 µs
- Max: 89.30 ms
- Count: 25 parts

**By Language:**

- **c**: Total 198.22 µs, Avg 99.11 µs, Min 94.09 µs, Max 104.14 µs, Count 2
- **python**: Total 400.73 ms, Avg 17.42 ms, Min 51.76 µs, Max 89.30 ms, Count 23
