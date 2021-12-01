# JK this is meant to generate condor 'in' files for the
#   original policy gradient version

import numpy as np
file_name = "run_hyperparams.in"
file=open(file_name, 'w')



lambd_list = ["[0.0]","[0.005]","[0.01]","[0.03]","[0.1]","[0.3]","[1.0]","[3.0]","[10.0]","[30.0]"]
epochs_list = [100]
lr_list = [1e-3,1e-4]
hidden_layer_list = [1,2,3]
optimizer_list = ['adam','sgd']
quad_reg_list = [0]
partial_train_data_list = [ "/home/jkotary/fultr/transformed_datasets/german/Train/partial_train_5k.pkl" ]
partial_val_data_list = ["/home/jkotary/fultr/transformed_datasets/german/Train/partial_valid_5k.pkl"]
partial_test_data_list = ["/home/jkotary/fultr/transformed_datasets/german/full/test.pkl"]  # JK note same as valid
log_dir_list = ["runs/default_JK"]
sample_size_list = [32]
batch_size_list = [4,64]
soft_train_list = [0]
allow_unfairness_list = [0]
fairness_gap_list = [0.0,0.001,0.005,0.01,0.05]
embed_groups_list = [0]
embed_quadscore_list = [0]


count = 0
for lambd in lambd_list:  # need to convert this to "[lambd]"
    for epochs in epochs_list:
        for lr in lr_list:
            for hidden_layer in hidden_layer_list:
                for optimizer in optimizer_list:
                    for quad_reg in quad_reg_list:
                        for partial_train_data in partial_train_data_list:
                            for partial_val_data in partial_val_data_list:
                                for partial_test_data in partial_test_data_list:
                                    for log_dir in log_dir_list:
                                        for sample_size in sample_size_list:
                                            for batch_size in batch_size_list:
                                                for soft_train in soft_train_list:
                                                    for allow_unfairness in allow_unfairness_list:
                                                        for fairness_gap in fairness_gap_list:
                                                            for embed_groups in embed_groups_list:
                                                                for embed_quadscore in embed_quadscore_list:
                                                                    print(  lambd,
                                                                            epochs,
                                                                            lr,
                                                                            hidden_layer,
                                                                            optimizer,
                                                                            quad_reg,
                                                                            partial_train_data,
                                                                            partial_val_data,
                                                                            partial_test_data,
                                                                            log_dir,
                                                                            sample_size,
                                                                            batch_size,
                                                                            soft_train,
                                                                            count,
                                                                            allow_unfairness,
                                                                            fairness_gap,
                                                                            embed_groups,
                                                                            embed_quadscore,
                                                                            sep=',',
                                                                            file=file  )

                                                                    count = count + 1

file.close()
print("Done")
