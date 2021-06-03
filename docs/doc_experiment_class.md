# `Experiment` class

Each experiment is described via an [Experiment object](../singularity/experiment.py), which possesses the following attributes:
- `exec` Prefix of the executable. It is set to `aurora` for all experiments.
- `algo` Name of the algorithm to consider. The main values for that variable are:
    - For AURORA:
        - `aurora_uniform` - AURORA with a uniform selector
        - `aurora_novelty` - AURORA with a novelty selector
        - `aurora_surprise` - AURORA with a surprise selector (the surprise score corresponds to the error of reconstruction by the encoder)
        - `aurora_nov_sur` - AURORA mixing a novelty (50%) and surprise (50%) selectors
        - `aurora_curiosity` - AURORA with a curiosity selector
    - For TAXONS:
        - `taxons` - TAXONS (using a selection procedure based on Novelty and Surprise) as described in [Unsupervised Learning and Exploration of Reachable Outcome Space](https://ieeexplore.ieee.org/document/9196819) (Paolo et al., 2020)
        - `taxo_n` - TAXO_N (using a selection only based on Novelty) as described in (Paolo et al., 2019)
        - `taxo_s` - TAXO_S (using a selection only based on Surprise) as described in (Paolo et al., 2019)
    - For the Hand-coded baselines:
        - `hand_coded_qd` - Hand-Coded QD algorithm using an unstructured archive with 2-dimensional hand-coded behavioural descriptors, and a uniform selector
        - `hand_coded_qd_no_sel` - Hand-Coded QD algorithm without any selector (new individuals are generated randomly). This is equivalent to random search.
        - `hand_coded_taxons` - Equivalent to Novelty Search, as described in [Novelty Search makes Evolvability Inevitable](https://dl.acm.org/doi/10.1145/3377930.3389840) (Doncieux et al., 2020)
- `env` Name of the environment/task to consider. The main values for that variable are:
    - `hard_maze` - Maze task
    - `hexa_cam_vertical` - Hexapod task
    - `air_hockey` - Air-Hockey task
- `latent_space` - Number of dimensions of the Behavioural Descriptor.
  In the case of AURORA and TAXONS, it corresponds to the number of dimensions of the latent space.
- `fixed_l` - If that value is not `None`, then the distance threshold
- `encoder_type` - Type of encoder used to encode the data.
  It is mostly equal to one of the following elements:
    - `EncoderType.cnn_ae` for the maze and hexapod tasks
    - `EncoderType.mlp_ae` for the air-hockey task
    - `EncoderType.none` for the hand-coded variants
- `lp_norm` - decides which L^p norm to use to compute distances (set to 2 in all our experiments)
- `has_fit` - If false, the fitness of individuals is not taken into account: the algorithm then becomes a pure divergent search procedure.
  It is set to True for all experiments (except for TAXONS and Novelty Search, which do have any mechanism to consider the fitness).
- `use_volume_adaptive_threshold` (set to `False` by default)
    - If `False`, the AURORA experiments use the Container Size Control technique (CSC).
    - If `True`, the AURORA experiments use the Volume Adaptive Threshold technique (VAT). 
- `taxons_elitism` - If true, TAXONS and Novelty Search select the best individuals from the set {parents + offspring}.
  If false, the parents are always discarded.
- `update_container_period` - Value of the container update period `T_{\mathcal{C}}` (10 by default)
- `coefficient_proportional_control_l` - Value of `K_{CSC}` (5e-6 by default)

Other possible values for those variables can be found in the [collections_experiments folder](../singularity/collections_experiments/) or in the [compilation_variable file](../cpp/compilation_variables.hpp).
