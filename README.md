# Accelerating Montgomery Multiplication

## Perfomance
Adding 20M points

### Era-Bellmann
- 1 SM/1 Thread: 14333.710938 ms
- 1 SM/40 Thread: 374.893951 ms
- 40 SM/1 Thread: 360.339844 ms
- 40 SM/40 Thread: 9.612320 ms
- 40 SM/384 Thread: 2.361152 ms

--> ~ 3 Billion Operations per second


## Refs
- http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf
- https://github.com/matter-labs/era-bellman-cuda