import history

imax = 160
hsize = 100

h = history.history(hsize)
for i in range(imax):
    h.append(i)

for i in h.sample(hsize):
    assert i >= imax - hsize and i < imax
    #print i

for idx, stored in enumerate(h.whole()):
    assert stored == idx + imax - hsize
    #print stored
