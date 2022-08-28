import sidechainnet as scn
import numpy as np
import wandb
from pytorch_lightning.loggers import WandbLogger

# –––––––––––––––––––––––––– WANDB ––––––––––––––––––––––––––#
# wandb.init(project="SummerResearch2022", entity="jbkasper5")
# wandb_logger = WandbLogger()
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#


# –––––––––––––––––––––––––– LOAD THE DATA ––––––––––––––––––––––––––#
np.set_printoptions(suppress=True)
np.random.seed(0)
# use this line to load a small portion of the sidechainnet dataset
d = scn.load("debug", scn_dataset=True)
# use this line for the full dataset
# d = scn.load(casp_version=7, thinning=30, scn_dataset=True)
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#


# –––––––––––––––––––––––––– DATA ATTRIBUTES ––––––––––––––––––––––––––#
print(d.splits)

# get the first protein in the dataset
print(d[0])

# view the first protein in the dataset as a 3D molecule
d[0].to_3Dmol() # only works in a jupyter esque notebook

# get the chain mask of the 31st protein in the sequence - Note that '-' means the protein is missing it's full structure
print(d[0].mask)

# get the angles from the first protein
print(d[0].angles) # note that since some chains are shorter than others or have different angle requirements, some values are autofilled as NAN
# also note that the angles vector is actually the sine of the angles, so you need to take sin^(-1) of the model output

# get the 3D coordinates of the first protein chain
print(d[0].coords) # again, with shorter sequences, some rows (which would be atoms) are filled in as NAN

# get the number of sequences in this [portion] of the dataset
print(len(d))
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#


# –––––––––––––––––––––––––– USING ATTRIBUTES ––––––––––––––––––––––––––#
# ––––– SEP ––––– #
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('darkgrid')

lengths = [len(p) for p in d] # get the length of each protein in the dataset

print("Mean/median length =", int(np.mean(lengths)), np.median(lengths)) # print mean and median

# create and decorate histogram of lengths
sns.histplot(lengths)
plt.axvline(np.mean(lengths), color='black', linestyle="--")
plt.xlabel("Sequence Length")
plt.show()

MAX_LEN = np.max(lengths) # the longest sequence in the dataset is 890 amino acids
print(MAX_LEN)
exit()
# ––––––––––––––– #

# ––––– SEP ––––– #
# Experiment quality proteins have a resolution around 3 or less...
train_res = [p.resolution for p in d if p.split == 'train']
valid_res = [p.resolution for p in d if p.split == 'valid-10']

print("Maximum Resolution =", np.max([r for r in train_res if r is not None ]))
print("Minimum Resolution =", np.min([r for r in train_res if r is not None ]))

sns.histplot(train_res, label="train")
sns.histplot(valid_res, label="valid-10", color='orange')
plt.xlabel("Resolution")
plt.legend()
plt.show()
# ––––––––––––––– #

# ––––– SEP ––––– #
p = d[40]
p.trim_edges()
print("Sequence =", p.seq)
print("\nAngles =\n", p.angles[:3], "\n\t...")
print("\nCoordinates =\n", p.coords[:26], "\n\t...")

sb = scn.StructureBuilder(p.seq, p.angles) # structure builder builds a protein based only on angles and the sequence, meaning it is used to generate 3D structures from model predictions
sb.to_3Dmol() # render the object

p.to_pdb("example.pdb") # you can save structures into a file and view them later
# ––––––––––––––– #
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#


# –––––––––––––––––––––––––– PYTORCH SPECIFICS ––––––––––––––––––––––––––#
# dynamic batching basically says that we can group more smaller sequences in a batch and less large sequences in a batch
d = scn.load("debug", with_pytorch = 'dataloaders', batch_size = 12, dynamic_batching = True) # load in example dataset
print(d.keys()) # print the keys

# weird behavior
if __name__ == '__main__':
    for batch in d['train']:
        print(batch)
    print(batch[0])
    print(batch.seqs.shape) # shape of [batch_size, max_len, one_hot_representation]
    print(batch.angles.shape) # shape of [batch_size, max_len, angle_vector]
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––#