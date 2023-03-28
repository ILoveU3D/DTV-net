import shutil
#path1 = "/home/nanovision/wyk/data/shepp/SheppLogan.raw"
path2 = "/home/nanovision/wyk/data/sheppInput/SheppLogan.raw"
for i in range(300):
    #shutil.copy(path1, "/home/nanovision/wyk/data/shepp/SheppLogan_{}.raw".format(i))
    shutil.copy(path2, "/home/nanovision/wyk/data/sheppInput/SheppLogan_{}.raw".format(i))
    print("finish {}".format(i));

