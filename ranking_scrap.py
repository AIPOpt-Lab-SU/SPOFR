import numpy as np
import torch



def get_multiple_exposures(rankings, position_bias_vector):
    num_docs = rankings.shape[-1]

    """
    pb_matrix = position_bias_vector[:num_docs].expand_as(rankings)
    exposures = torch.zeros_like(rankings).float()
    exposures = exposures.scatter_(
        1,
        rankings,
        pb_matrix
    )
    """

    exposures = (position_bias_vector)[rankings.T].T


    return exposures

def compute_multiple_group_disparity(rankings,
                                     group_identities,
                                     position_biases):

    inds_g0 = (group_identities == 0).float()
    inds_g1 = (group_identities == 1).float()



    exposures = get_multiple_exposures(rankings, position_biases)


    exposures_g0 = exposures * inds_g0#.unsqueeze(1)
    exposures_g1 = exposures * inds_g1#.unsqueeze(1)

    """
    print("inds_g0 = ")
    print( inds_g0 )
    print("inds_g1 = ")
    print( inds_g1 )

    print("rankings = ")
    print( rankings )

    print("exposures = ")
    print( exposures )

    print("exposures_g0 = ")
    print( exposures_g0 )

    print("exposures_g1 = ")
    print( exposures_g1 )
    """

    ratio0 = torch.sum(exposures_g0, dim=-1) / inds_g0.sum()
    ratio1 = torch.sum(exposures_g1, dim=-1) / inds_g1.sum()
    group_disparities = ratio0 - ratio1
    return group_disparities

# batch-level test_fairness
def test_group_fairness(pmat_batch, group_identities, position_bias_vector):

    returns = []
    for k in range(len(pmat_batch)):
        pmat = pmat_batch[k]
        #group_identities = group_identities_batch[k]

        v = position_bias_vector[:pmat.shape[1]]
        f = ( group_identities / group_identities.sum() )   -   ( (1-group_identities) / (1-group_identities).sum() )


        print("pmat = ")
        print( pmat    )
        print("v = ")
        print( v    )
        print("f = ")
        print( f    )

        ret = torch.mv(pmat,v)

        print("pmat * v = ")
        print( ret    )

        ret = torch.dot(f,ret)

        print("f * pmat * v = ")
        print( ret    )

        returns.append(ret.item())


    return torch.Tensor(returns)




position_bias_vector = 1. / torch.arange(1.,100.)
I = torch.eye(10)

rankings = torch.argsort( torch.rand(10,10), 1 )
#rels = torch.rand(10,10)*10


group_identities =  torch.Tensor([0,0,1,0,0,0,0,1,0,0])


print("group_identities = ")
print( group_identities )

print("rankings = ")
print( rankings )

pmat_batch = []
for i in range(rankings.shape[0]):
    P = I[  rankings[i]  ]
    pmat_batch.append(P)
pmat_batch = torch.stack(pmat_batch)


print("pmat_batch = ")
print( pmat_batch )

print(   "test_group_fairness = "   )
print(    test_group_fairness(pmat_batch, group_identities, position_bias_vector)    )
print(   "compute_multiple_group_disparity = "   )
print(    compute_multiple_group_disparity(rankings,
                                           group_identities,
                                           position_bias_vector)     )
