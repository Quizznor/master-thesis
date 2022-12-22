from Binaries import *

Events = EventGenerator("all", split = 1, real_background = False, ignore_particles = 4, ignore_low_vem = 1.0)
Events.unit_test(1000)