# JK this is meant to generate condor 'in' files for the
#   original policy gradient version

import numpy as np
from datetime import date
file_name = "run_hyperparams.in"
file=open(file_name, 'w')



lambd_list = [0.1]#["[0.0]","[0.005]","[0.01]","[0.03]","[0.1]","[0.3]","[1.0]","[3.0]","[10.0]","[30.0]"]
epochs_list = [1000]  #[2000]
lr_list = [1e-5,5e-6]   # removed on account of not as good:   2.5e-4,
hidden_layer_list = [5] #[2,4,5] #[1,2,3,4]
optimizer_list = ['adam']#,'sgd']
dropout_list = [0.0,0.1]#,0.25]

log_dir_list = [1]
sample_size_list = [128]#,128]
batch_size_list =  [64] #[4,64]
soft_train_list = [1]

allow_unfairness_gap_list = [ (1,0.0000001), (1,0.025), (1,0.05), (1,0.1), (1,0.2), (1,0.4), (1,0.8), (1,1.6) ]
allow_unfairness_gap_list = [ (1,0.0000001), (1,0.025), (1,0.05), (1,0.075), (1,0.1), (1,0.125), (1,0.150), (1,0.175), (1,0.2),  (1,0.3), (1,0.4), (1,0.6), (1,0.8) ]



output_tag = 'LP_tests_multi_disp1' + str(date.today()) + '_'
entreg_decay_list = [0.1]
evaluate_interval_list = [2000]
seeds_list = [0,1,2]
disparity_type_list = ['disp0']#,'disp1']
multi_groups_list = [2,3,4,5,6,7]

#partial_train_data_list = [ "/home/jkotary/fultr/transformed_datasets/german/Train/partial_train_5k.pkl" ]
#partial_val_data_list = ["/home/jkotary/fultr/transformed_datasets/german/Train/partial_valid_5k.pkl"]
#partial_test_data_list = ["/home/jkotary/fultr/transformed_datasets/german/full/test.pkl"]  # JK note same as valid

#partial_train_val_test_data_list = [ ("/home/jkotary/fultr/transformed_datasets/german/Train/partial_train_50k.pkl",
#                                      "/home/jkotary/fultr/transformed_datasets/german/Train/partial_valid_50k.pkl",
#                                      "/home/jkotary/fultr/transformed_datasets/german/full/test.pkl") ]

#partial_train_val_test_data_list = [ ("/home/jkotary/fultr/transformed_datasets/german/Train/partial_train_5k.pkl",
#                                      "/home/jkotary/fultr/transformed_datasets/german/Train/partial_valid_5k.pkl",
#                                      "/home/jkotary/fultr/transformed_datasets/german/full/test.pkl") ]




                                     #("/home/jkotary/fultr/transformed_datasets/german/Train/partial_train_5k.pkl",
                                     #"/home/jkotary/fultr/transformed_datasets/german/Train/partial_valid_5k.pkl",
                                     #"/home/jkotary/fultr/transformed_datasets/german/full/test.pkl") ]

partial_train_val_test_data_list = [
                                        ("/home/jkotary/fultr/transformed_datasets/german/Train/partial_train_50k.pkl",
                                         "/home/jkotary/fultr/transformed_datasets/german/Train/partial_valid_5k.pkl",
                                         "/home/jkotary/fultr/transformed_datasets/german/full/train_test_valid.pkl") ,

                                        ("/home/jkotary/fultr/transformed_datasets/german/Train/partial_train_JK_100k.pkl",
                                         "/home/jkotary/fultr/transformed_datasets/german/Train/partial_valid_5k.pkl",
                                         "/home/jkotary/fultr/transformed_datasets/german/full/train_test_valid.pkl") ,

                                        ("/home/jkotary/fultr/transformed_datasets/german/Train/partial_train_5k.pkl",
                                         "/home/jkotary/fultr/transformed_datasets/german/Train/partial_valid_5k.pkl",
                                         "/home/jkotary/fultr/transformed_datasets/german/full/train_test_valid.pkl") ,


                                            ("/home/jkotary/fultr/transformed_datasets/mslr/Train/partial_train_120k.pkl",
                                            "/home/jkotary/fultr/transformed_datasets/mslr/Train/partial_valid_4k.pkl",
                                            "/home/jkotary/fultr/transformed_datasets/mslr/full/test.pkl"),

                                            ('/home/jkotary/fultr/transformed_datasets/mslr/Train/partial_train_36k.pkl',
                                            "/home/jkotary/fultr/transformed_datasets/mslr/Train/partial_valid_4k.pkl",
                                            "/home/jkotary/fultr/transformed_datasets/mslr/full/test.pkl"),

                                            ("/home/jkotary/fultr/transformed_datasets/mslr/Train/partial_train_12k.pkl",
                                            "/home/jkotary/fultr/transformed_datasets/mslr/Train/partial_valid_4k.pkl",
                                            "/home/jkotary/fultr/transformed_datasets/mslr/full/test.pkl")  ]


partial_train_val_test_data_list = [ #("/home/jkotary/fultr/transformed_datasets/german/Train/partial_train_50k.pkl",
                                     # "/home/jkotary/fultr/transformed_datasets/german/Train/partial_valid_5k.pkl",
                                     # "/home/jkotary/fultr/transformed_datasets/german/full/train_test_valid.pkl") ,

                                     ("/home/jkotary/fultr/transformed_datasets/mslr/Train/partial_train_36k.pkl",
                                      "/home/jkotary/fultr/transformed_datasets/mslr/Train/partial_valid_4k.pkl",
                                      "/home/jkotary/fultr/transformed_datasets/mslr/full/test.pkl")  ]

gme_new_list = [0]

embed_groups_quadscore = (0,0)

if 'mslr' in partial_train_val_test_data_list[0][0]:
    hidden_layer_list.append(6)

count = 1
for lambd in lambd_list:  # need to convert this to "[lambd]"
    for epochs in epochs_list:
        for lr in lr_list:
            for hidden_layer in hidden_layer_list:
                for optimizer in optimizer_list:
                    for dropout in dropout_list:
                        for partial_train_val_test_data in partial_train_val_test_data_list:
                            for gme_new in gme_new_list:
                                for sample_size in sample_size_list:
                                    for batch_size in batch_size_list:
                                        for soft_train in soft_train_list:
                                            for allow_unfairness_gap in allow_unfairness_gap_list:
                                                #for fairness_gap in fairness_gap_list:
                                                #for embed_groups_quadscore in embed_groups_quadscore_list:
                                                for multi_groups in multi_groups_list:
                                                    for entreg_decay in entreg_decay_list:
                                                        for evaluate_interval in evaluate_interval_list:
                                                            for seed in seeds_list:
                                                                for disparity_type in disparity_type_list:

                                                                    (allow_unfairness, fairness_gap) = allow_unfairness_gap
                                                                    (partial_train_data, partial_val_data, full_test_data) = partial_train_val_test_data
                                                                    (embed_groups, embed_quadscore) = embed_groups_quadscore
                                                                    print(  lambd,
                                                                            epochs,
                                                                            lr,
                                                                            hidden_layer,
                                                                            optimizer,
                                                                            dropout,
                                                                            partial_train_data,
                                                                            partial_val_data,
                                                                            full_test_data,
                                                                            gme_new,
                                                                            sample_size,
                                                                            batch_size,
                                                                            soft_train,
                                                                            count,
                                                                            allow_unfairness,
                                                                            fairness_gap,
                                                                            embed_groups,
                                                                            multi_groups,
                                                                            entreg_decay,
                                                                            evaluate_interval,
                                                                            output_tag,
                                                                            disparity_type,
                                                                            #seed,
                                                                            sep=',',
                                                                            file=file  )

                                                                    count = count + 1




file.close()
print("Done")
