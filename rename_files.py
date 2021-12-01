import os

path = 'csv_spo_final_take3/'


for file in os.listdir(path):
    print(file)
    os.rename(path+file, path+'new_'+file)
