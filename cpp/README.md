# `cpp/` structure

## `algorithms/` 

Provides definitions for all the algorithms (AURORA, TAXONS, and their Hand-Coded variants).  

Each algorithm has a modifier type (modifier_t), and a stat type (stat_t), both following the [sferes2 framework](http://sferes2.github.io/sferes2/reference.html).
Moreover, each algorithm defines a selector type (selector_t), and a container type (container_t), following the [Quality Diversity framework]()
See for example [the definitions of AURORA variants](./algorithms/aurora/definitions_aurora.hpp).

## `environments/` 

Source code for all tasks under study (Maze, Hexapod and Air-Hockey).

Each environment has a fitness type (fit_t), genotype type (gen_t), and phenotype type (phen_t), all following the sferes2 framework.
  
## `modifier/` 

The modifier is executed after having mutated, copied and evaluated the population.

### Modifier for AURORA and TAXONS

In the case of AURORA and TAXONS, we use the modifier [DimensionalityReduction](./modifier/dimensionality_reduction.hpp), which has several roles:
- Using the Encoder to encode the sensory data from new individuals into Behavioural Descriptors (BDs).
- *Encoder Update*: Training the Encoder based on all the sensory data from the container, and then use it to recompute all the BDs.
- *Container Size Control (CSC)*: Updating the value of the distance threshold (d_{min}).
- *Container Update*: Performing the container update by removing all individuals from the container, and re-adding them with their recomputed BDs, and with the up-to-date distance threshold d_{min}.

This modifier uses a [NetworkLoader](./modifier/network_loader_pytorch.hpp) as an interface with the torch models.
For example, the `NetworkLoader` can be called to evaluate the latent projection of a model, and to train the torch model on a dataset.

The torch models are stored in the [autoencoder folder](./modifier/autoencoder/).


### Modifier for hand-coded variants

Hand-coded variants use a different modifier [ContainerUpdateHandCoded](./modifier/container_update_hand_coded.hpp), which does not consider any dimensionality reduction algorithm.
In other words, the hand-coded variants only perform the last two items described above: 
- *Container Size Control (CSC)*: Updating the value of the distance threshold (d_{min}).
- *Container Update*: Performing the container update by removing all individuals from the container, and re-adding them with their recomputed BDs, and with the up-to-date distance threshold d_{min}.

## `stat/` 

The Statistics classes are used to save the relevant data for further analysis. 


The most useful Statistics class is [`Projection`](./stat/stat_projection.hpp) which periodically saves the following attributes of all individuals from the archive: 
- index of the individual
- reconstruction error of its sensory data by the AE
- Unsupervised behavioural descriptor
- Hand-coded behavioural descriptor (ground-truth state)
- Fitness value (equal to -1 if the fitness function is not considered)
- "Implicit" Fitness value (present the fitness value that would be obtained if the fitness function was considered)
- Novelty score

## Source Files

- `aurora.cpp` - for defining main experiments.

- `save_video.cpp` - used to easily save videos for the Hexapod and Air-Hockey tasks.
  
- `train_ae_from_scratch.cpp` - used to re-train the AE from scratch given a pre-defined dataset (has been mostly used for the Maze task).
  
## Others

- `compilation_variables.hpp` - making the link between the variables defined in the `Experiment` object, and the C++ compilation variables.
