import image
import utime
import KPU as kpu
f=open('labels.txt','r')
labels=f.readlines()
f.close()
task = kpu.load(0x200000)
img = image.Image('muggcomp.jpg')
img.pix_to_ai()
start_time = utime.ticks_us()
for i in range(1000):
    fmap = kpu.forward(task, img)
    plist=fmap[:]
    pmax=max(plist)
    max_index=plist.index(pmax)
end_time = utime.ticks_us()
ellapsed = utime.ticks_diff(end_time,start_time)/1000
print(start_time)
print(end_time)
print("Az eltelt idő:", ellapsed)
print(pmax, labels[max_index])
