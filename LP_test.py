#JK 0803

import torch, numpy, os, sys
from torch import nn
#from evaluation import sample_double_stoch, multiple_sample_and_log_probability
from networksJK import PolicyLP, PolicyLP_Plus
#from evaluation import compute_dcg_rankings


"""
scores =  torch.Tensor([[ 0.7161,  0.8573,  0.2468,  0.7465, -0.2263,  0.0081,  0.3507,  0.2075,
                          0.0739,  0.2545,  0.0328,  0.4629,  0.3628,  0.1519,  0.3576, -0.3872,
                          0.1297,  0.3564,  0.0587, -0.3208],
                        [ 0.4916,  0.1974,  0.5085,  0.7465, -0.2026,  0.1018, -0.2020,  0.0155,
                          0.2705,  0.2143,  0.4667, -0.4390,  0.4057, -0.2159,  0.2518, -0.0217,
                          0.1297, -0.2160,  0.2303, -0.2090],
                        [ 0.2468, -0.1159,  0.7465,  0.3507,  0.2683,  0.3899,  0.4826,  0.2075,
                          0.2177, -0.4446,  0.0154,  0.3252, -0.2590, -0.2012,  0.4468, -0.2096,
                         -0.2160,  0.3564, -0.0594,  0.6037],
                        [-0.1092,  0.2369,  0.3056, -0.0403,  0.0081,  0.7804,  0.0856,  0.2800,
                          0.0154, -0.0152, -0.1959, -0.4018, -0.4649, -0.4390, -0.2012,  0.4173,
                          0.4468, -0.1444, -0.0217, -0.0098]])
"""

"These scores appeared right before PolicyLP_Plus leading to NaN"
scores = torch.Tensor([[ 8.7845e-02,  3.0729e-01,  1.3841e-01,  2.2779e-01, -2.3924e-01,
                          2.3625e-01, -1.7794e-01, -3.8440e-02,  1.9790e-01,  4.4483e-02,
                          6.6272e-01,  4.5028e-01,  3.1430e-01,  6.6546e-01,  1.1579e-01,
                          4.9145e-01,  2.8865e-01,  7.7143e-01,  8.4712e-02,  1.9706e-01],
                        #[ 1.9792e-01, -2.3924e-01,  4.1352e-01,  1.5531e-01,  1.0950e-02,
                        #  7.0820e-02,  9.9321e-02,  3.9382e-01, -3.8440e-02,  2.8947e-01,
                        # -1.0751e-01,  1.9790e-01,  1.6047e-01, -4.0301e-01,  3.7110e-01,
                        #  7.1352e-01,  2.6979e-01,  8.2892e-01,  5.7665e-01,  7.7763e-01],
                        [ 9.4937e-02, -2.3924e-01,  4.5744e-01,  4.5957e-01, -8.7312e-02,
                          3.2316e-01, -4.1939e-02,  3.9382e-01,  7.9789e-01, -8.5231e-02,
                          1.9790e-01,  2.6188e-01,  6.5783e-02, -4.3274e-02,  2.8851e-01,
                          6.6546e-01,  1.1579e-01,  5.4227e-01,  8.4712e-02,  4.5897e-01],
                        [ 1.9792e-01, -5.0770e-01,  1.3841e-01,  4.5957e-01, -1.4202e-01,
                         -1.0275e-01,  4.9831e-01,  2.3625e-01,  1.0950e-02, -2.0029e-01,
                          2.7416e-01,  1.1547e-04,  1.2828e-01,  1.9790e-01, -4.3274e-02,
                          2.8865e-01,  7.7143e-01,  7.1352e-01,  7.7763e-01,  4.5897e-01]])


scores.repeat(1,1,scores.shape[1])

scores.unsqueeze(0).view(scores.shape[0],-1,1)

scores.unsqueeze(0).view(scores.shape[0],1,-1)

torch.ones(scores.shape[0],1,scores.shape[1])

torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), torch.ones(scores.shape[0],1,scores.shape[1])  )

#torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1),


#scores = torch.nn.functional.relu(scores)

group_identities = torch.Tensor([[0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0.],
                                 #[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])

# Not related to the above; taken from some random output
test_rankings = torch.tensor(  [[[19,  2,  6, 15,  8, 14, 13, 12,  7, 18,  3, 11,  1,  9,  5, 17, 16, 0, 10,  4],
                                 [ 1, 10,  4, 17, 18,  9, 16,  2,  5, 19,  3, 12, 14,  0,  8,  6, 13, 7, 15, 11]],

                                [[13, 19, 14,  4,  5,  3,  7, 10,  1,  0, 15,  9, 12,  2, 16, 17,  6, 8, 11, 18],
                                 [ 6, 19,  7, 15,  5,  9,  2, 13,  1,  0, 18, 12,  3,  8, 16, 10,  4, 14, 17, 11]],

                                [[13, 18,  0, 17,  4, 15,  5,  9,  8, 11, 14,  6,  7, 10,  3,  1, 12, 16,  2, 19],
                                 [ 4, 15, 16,  1,  6, 14,  3, 13,  7,  9, 19,  0,  5,  8,  2, 12, 11, 17, 18, 10]],

                                [[11, 17,  8, 15, 16,  7, 13,  1,  2,  4, 12,  6,  5,  0, 19,  9, 18, 14,  3, 10],
                                 [17,  2,  6, 12,  0, 10, 14, 13,  5,  8, 19, 16, 11,  1,  7,  4,  9, 15, 18,  3]]]).long()


test_rels =       torch.Tensor([[0., 0., 0., 0., 5., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                #[0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])


#### temp ######
#policy_lp = PolicyLP_Plus(N=4)
#input("Finished inspecting row and column constraints")
################








#p_mat = 0.05*torch.ones(20,20).repeat(4,1,1)
batch = 3

policy_lp = PolicyLP_Plus(N=scores.shape[1])

#p_mat = policy_lp(scores.repeat(1,1,scores.shape[1]).squeeze(), None )#group_identities)
print("scores = ")
print( scores )
print("scores.repeat(1,1,scores.shape[1]).size() = ")
print( scores.repeat(1,1,scores.shape[1]).size()   )
print("torch.repeat_interleave(scores,4,1).size() = ")
print( torch.repeat_interleave(scores,scores.shape[1],1).size() )
input("Done printing scores")


fXv = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), scores.unsqueeze(0).view(scores.shape[0],-1,1).permute(0,2,1)  ).reshape(scores.shape[0],-1)
for k in range(batch):
    #fXv = torch.bmm( scores.unsqueeze(0).view(scores.shape[0],-1,1), torch.ones(scores.shape[0],1,scores.shape[1])  ).permute(0,2,1).reshape(scores.shape[0],-1)
    p_mat = policy_lp(  fXv[k].unsqueeze(0), None )
    #scores_rep = scores.repeat(1,1,scores.shape[1])
    #p_mat = policy_lp(  torch.repeat_interleave(scores,scores.shape[1],1)[k].unsqueeze(0), None ) #group_identities[k].unsqueeze(0))
    #p_mat = policy_lp( scores_rep.squeeze()[k].unsqueeze(0), None ) #group_identities[k].unsqueeze(0))
    #p_mat = policy_lp( torch.rand(scores_rep.shape).squeeze()[k].unsqueeze(0), None ) #group_identities[k].unsqueeze(0))
    print("Iteration {} complete".format(k))
    print("p_mat = ")
    print( p_mat )
    input("##############")
 #Jk note: Not every instance is infeasible, every instance show nan in verbose output



p_mat = policy_lp(  fXv, group_identities )



print("scores = ")
print(scores)
print("group_identities = ")
print(group_identities)



test_dscts = ( 1.0 / torch.log2(torch.arange(scores.shape[1]).float() + 2) ).repeat(batch,1,1)

print("p_mat = ")
print( p_mat )

print("test_dscts = ")
print( test_dscts )

print("test_rels = ")
print( test_rels )

print("p_mat.size() = ")
print( p_mat.size() )

print("test_dscts.size() = ")
print( test_dscts.size() )

print("test_rels.size() = ")
print( test_rels.size() )

a = torch.bmm( p_mat, test_dscts.view(batch,-1,1) )
b = torch.bmm( test_rels.view(batch,1,-1), a )

print("a = ")
print( a )
print("b = ")
print( b )

print("b.mean() = ")
print( b.mean() )
