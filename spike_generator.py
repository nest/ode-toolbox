import math
import numpy.random

# for the testing purpose fix the seed to 42 in order to make results reproducable
numpy.random.seed(42)

rate = 10.
t_len = 10.
slot_width = 0.2
mean_ISI = 1./rate
times = []

t_sum = 0
while t_sum < t_len:
    t_sum += numpy.random.exponential(mean_ISI, 1)[0]
    times.append(t_sum)

'''
Note that besides numpy.random, there is also the independent module
random. ISIs could also have been drawn using
random.expovariate(rate).
'''
n_spikes = numpy.random.poisson(rate * t_len)
times = numpy.random.uniform(0, t_len, n_spikes)
times = numpy.sort(times)


def count_spikes_in_slot(spikes, slot_width, time_span):
    time_slots = int(math.ceil(time_span/slot_width))
    spikes_per_slot = [0]*time_slots
    for slot in range(0, time_slots):
        t = list(filter(lambda x: slot * slot_width <= x < (slot + 1) * slot_width, spikes))
        spikes_per_slot[slot] = len(t)
    return spikes_per_slot


if __name__ == "__main__":
    print(len(count_spikes_in_slot(times, slot_width, t_len)))
    print(count_spikes_in_slot(times, slot_width, t_len))
