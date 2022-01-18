from scipy import io

v = io.loadmat('sounds.mat')
a = v['sounds']
print(a)
v = io.loadmat('icaTest.mat')
print(a)