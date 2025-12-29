# Linear Autoencoders with DRO losses and FairPCA

This repository accompanies the study: Yeh _et al_. **Robust self-supervised machine learning for single cell embeddings and annotations.** _bioRxiv_ (2025). This repository contains our exploratory implementation of Linear Autoencoders with DRO losses and FairPCA as other distributionally robust/fairness-aware alternatives to ([DR-GEM](https://github.com/Jerby-Lab/drgem/)).

# **Quick start**
1. Download this repository from Github:
   ```
	git clone https://github.com/Jerby-Lab/fair-lin-ae
	cd fair-lin-ae
   ```
2. (Recommended) Create conda environment (tested on version 24.7.1):
	```
	conda create -n fair-lin-ae python=3.11 pip
	conda activate fair-lin-ae
	```
4. Install dependencies:
	pip install -r requirements.txt
5. Run code for fitting Linear Autoencoders with DRO losses
	cd src
	python linae.py
6. Run code for fitting FairPCA
	cd src
	python fairpca.py

# **Software Requirements**
* macOS: Tahoe 26.1
* Python (tested on 3.11.9)
* Python package dependencies: numpy (1.26.4), pandas (2.2.2), scipy (1.14.0), scikit-learn (1.5.1), matplotlib (3.9.1), seaborn (0.13.2), tqdm (4.66.4), umap-learn (0.5.6), scanpy (1.10.2), torch (2.4.0)

# **Citation**
Please consider citing this work: Yeh _et al_. **Robust self-supervised machine learning for single cell embeddings and annotations.** _bioRxiv_ (2025)

# **License** 

BSD 3-Clause License provided ([here](https://github.com/Jerby-Lab/fair-lin-ae/blob/main/LICENSE)).

# **Contact**
For any inquiries on this repository please feel free to post these under "issues" or reach out to Christine Yiwen Yeh ([cyyeh@stanford.edu](cyyeh@stanford.edu)) and Livnat Jerby ([ljerby@stanford.edu](ljerby@stanford.edu)).
