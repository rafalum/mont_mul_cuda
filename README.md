# Accelerating Montgomery Multiplication

## Perfomance
Adding 20M points

### Era-Bellmann
- 1 SM/1 Thread: 16538.007812 ms
- 1 SM/40 Thread: 456.193115 ms
- 40 SM/1 Thread: 422.151672 ms
- 40 SM/40 Thread: 11.800256 ms
- 40 SM/384 Thread: 4.506816 ms

--> ~ 3 Billion Operations per second


## Refs
- http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf
- https://github.com/matter-labs/era-bellman-cuda