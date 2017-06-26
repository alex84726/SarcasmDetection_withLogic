import numpy as np

file1 = 'posproc_delOOV.npy'
file2 = 'negproc_delOOV.npy'
data1 = np.load(file1)
data2 = np.load(file2)

counter = 0
i = 0
save = [ ]
for data in data1:
    if counter == 3000:
        np.save('data_npy/posproc_delOOV_'+str(i*3000+1)+"-"+str((i+1)*3000)+'.npy',save)
        save = [ ]
        i += 1
        counter = 0
    save.append(data)
    counter+=1
np.save('data_npy/posproc_delOOV_'+str(i*3000+1)+"-end.npy",save)

counter = 0
i = 0
save = [ ]
for data in data2:
    if counter == 3000:
        np.save('data_npy/negproc_delOOV_'+str(i*3000+1)+"-"+str((i+1)*3000)+'.npy',save)
        save = [ ]
        i += 1
        counter = 0
    save.append(data)
    counter+=1
np.save('data_npy/negproc_delOOV_'+str(i*3000+1)+"-end.npy",save)
        
