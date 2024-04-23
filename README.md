# Accelerating Montgomery Multiplication

## Perfomance
Multiplying 20M points

### Float
- 40 SM/32 Thread: 132.95 ms
- 40 SM/128 Thread: 71.47 ms
- 40 SM/384 Thread: 71.08 ms
- 40 SM/512 Thread: 71.10 ms
- 40 SM/768 Thread: 71.13 ms

### Supra
- 40 SM/32 Thread: 65.37 ms
- 40 SM/128 Thread: 16.68 ms
- 40 SM/384 Thread: 7.60 ms
- 40 SM/512 Thread: 6.82 ms
- 40 SM/768 Thread: 6.71 ms

--> ~ 1.5 Billion Operations per second


### Era-Bellmann
- 40 SM/32 Thread: 23.38 ms
- 40 SM/128 Thread: 8.33 ms
- 40 SM/384 Thread: 4.59 ms
- 40 SM/512 Thread: 4.38 ms
- 40 SM/768 Thread: 4.98 ms

--> ~ 2.3 Billion Operations per second


## Refs
- http://www.acsel-lab.com/arithmetic/arith23/data/1616a047.pdf
- https://github.com/matter-labs/era-bellman-cuda