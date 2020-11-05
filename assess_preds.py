import os
import numpy as np
from PIL import Image
    
sorted_preds = sorted(os.listdir('C:\\Users\\Guilherme\\Documents\\ZOOTR\\Testes'), key = lambda x: float(x.split('-')[0]))

inds = []
preds = []
for i in sorted_preds:
    tmp = i.split('-', 1)
    inds.append(int(tmp[0]))
    preds.append(float(tmp[1][0:(len(tmp[1])-4)]))



thr = 0.7
tmp_prob = thr
tmp_ind = -1
selected_imgs = []

for i in range(0, len(preds)):
    if preds[i] > tmp_prob:
        tmp_prob = preds[i]
        tmp_ind = i
    elif tmp_prob > thr:
        selected_imgs.append(tmp_ind)
        tmp_prob = thr
        tmp_ind = -1
        
np.array(inds)[selected_imgs]    



def check_name(filename, folder):
    if filename not in os.listdir(folder):
        return filename
    for i in range(2, 1000000):
        potential_name = filename.split('.')[0] + ' (' + str(i) + ')' + '.' + filename.split('.')[1] 
        if potential_name not in os.listdir(folder):
            return potential_name
        #return filename.split('.')[0] + ' ' +  str(time.time()) + '.' + filename.split('.')[1] 
#Create new name when old name is already in desired folder, tries to add (N) to the end of name until N = 1000000 if those are all taken just adds timestamp        
        

items_path = 'C:\\Users\\Guilherme\\Documents\\ZOOTR\\Test\\Items\\'
random_path = 'C:\\Users\\Guilherme\\Documents\\ZOOTR\\Random_frames\\'

counter = 0
for i in os.listdir('C:\\Users\\Guilherme\\Documents\\ZOOTR\\Testes'):
    for j in np.array(inds)[selected_imgs]:
        if i.startswith(str(j) + '-'):
            counter += 1
            img_path = os.path.join('C:\\Users\\Guilherme\\Documents\\ZOOTR\\Testes', i)
            print(img_path)
            img = Image.open(img_path)
            img.resize((200, 200)).show(title = str(counter) + '/' + str(len(selected_imgs)))
            is_item = input('Is item? (y/n)')
            if is_item not in ('y', 'n'):
                continue
            if is_item == 'y':
                file_name = input('File name :')
                file_name = check_name(file_name)
                command_copy = 'copy ' + img_path + ' ' + items_path + file_name + '.png'
            elif is_item == 'n':
                file_name = i
                file_name = check_name(file_name)
                command_copy = 'copy ' + img_path + ' ' + random_path + file_name
            os.system(command_copy)
                
            