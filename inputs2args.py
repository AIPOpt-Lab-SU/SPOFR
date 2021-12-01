import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--inputs",    type=str,    default="",  help="List of comma separated inputs")
args = parser.parse_args()

#23
params  = [  " ", # lambda
             "epochs",
             "lr",
             "hidden_layer",
             "optimizer",
             "quad_reg",
             "partial_train_data",
             "partial_val_data",
             "full_test_data",
             "log_dir",
             "sample_size",
             "batch_size",
             "soft_train",
             "index",
             "allow_unfairness",
             "fairness_gap",
             "embed_groups",
             "embed_quadscore",
             "entreg_decay",
             "evaluate_interval",
             "output_tag",
             #"seed",
             "disparity_type"]


#str  = "[0.1],100,0.001,1,adam,0.1,/home/jkotary/fultr/transformed_datasets/german/Train/partial_train_5k.pkl,/home/jkotary/fultr/transformed_datasets/german/Train/partial_valid_5k.pkl,/home/jkotary/fultr/transformed_datasets/german/full/test.pkl,runs/default_JK,32,16,1,1,0,0.0,1,0,0.1,500,5ktests_altscoring,0"

tokens = args.inputs.split(',')
tokens[-2]

output = ""
if len(tokens) != len(params):
    print('Number of arguments and values not equal')
else:
    for i in range(len(tokens)):
        output += "--"+params[i] + " " +tokens[i] + " "


print(output)
