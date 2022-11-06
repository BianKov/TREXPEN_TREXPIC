The implemented algorithms assign position vectors of *d* Cartesian coordinates to each node of a network. The embedding functions assume that the network to be embedded is connected (if it is undirected) or weakly connected (if it is directed).

# HOPE=*H*igh-*O*rder *P*roximity preserved *E*mbedding
In contrast to [the original HOPE method](https://dl.acm.org/doi/10.1145/2939672.2939751), this implementation performs usual SVD instead of JDGSVD. The implemented calculation of the matrix of Katz indexes is valid only for unweighted graphs! Two new variants introduced to resolve the restricting effect of the solely non-negative proximity values: HOPE-R and HOPE-S.

# TREXPEN=*TR*ansformation of *EX*ponential shortest *P*ath lengths to *E*uclidea*N* measures
Instead of the Katz indexes used by HOPE, TREXPEN builds on exponential proximities that consider only the length of the shortest paths instead of all the path lengths. This implementation can be used on weighted networks too; smaller weights should correspond to larger expected geometrical proximities (i.e., larger "similarities"). Two variants introduced to resolve the restricting effect of the solely non-negative proximity values: TREXPEN-R and TREXPEN-S.

# MIC=*M*odel-*I*ndependent Euclidean-hyperbolic *C*onversion
An algorithm for converting a Euclidean node arrangement to a hyperbolic one. It assumes that in the Euclidean embedding the large topological proximities are reflected by high inner products - these will be converted to small hyperbolic distances.

# TREXPIC=*TR*ansformation of *EX*ponential shortest *P*ath lengths to hyperbol*IC* distances
A dimension reduction technique for obtaining embeddings directly in the hyperbolic space instead of using a Euclidean-hyperbolic conversion. This implementation can be used on weighted networks too; smaller weights should correspond to larger expected geometrical proximities (i.e., larger "similarities").


# Reference
[Kovács, B. & Palla, G. *Model-independent methods for embedding directed networks into Euclidean and hyperbolic spaces*. arXiv:2207.07633 [physics.soc-ph] (2022)]( 	
https://doi.org/10.48550/arXiv.2207.07633)

For any problem, please contact Bianka Kovács: <bianka.kovacs@ttk.elte.hu>
