# Transfer-learning-for-BNER-Bioinformatics-2018

This repository contains supplementary data, and links to the model and corpora used for the paper _Transfer learning for biomedical named entity recognition with neural networks_ (pre-print available [here](https://www.biorxiv.org/content/early/2018/02/12/262790).)

### Code

Corpora pre-processing steps were collected in a single script with a jupyter notebook for ease-of-use. Script and notebook can be found in `code`.

#### Model

The model used in this study is __NeuroNER__ [[1](#citations)], a domain-independent named entity recognizer (NER) based on a bi-directional long short term memory network-conditional random field (LSTM-CRF). A repository for the model can be found [here](https://github.com/Franck-Dernoncourt/NeuroNER).

NeurNER uses standard python config files to specify hyperparameters. We provide three of these config files for reproducibility (see `code/configs`):

1. `baseline.ini`: config used while training on the target data sets (i.e., the baseline.)
2. `source.ini`: config used while training on the source data sets.
3. `transfer.ini`: config used while transferring a model trained on the source data set for training on a target data set.

#### Word Embeddings

The word embeddings used in this study were obtained from [here](http://bio.nlplab.org/#word-vectors) [[2](#citations)]. Code for converting the word vectors to the `.txt` format necessary for use with NeuroNER can be found in the __jupyter notebook__ in `code`, under __data cleaning__.

### Corpora

All corpora used in this study (which can be re-distributed) are in the `corpora` folder (given in Brat-standoff format).

> Data can be uncompressed with the following command: `tar -zxvf <name_of_corpora>`.

Alternatively, the corpora can be publicly accessed at the following links:

| Corpora | Text Genre | Standard | Entities | Publication |
| --- | --- | --- | --- | --- |
| [AZDC](http://diego.asu.edu/downloads/AZDC_6-26-2009.txt) | Scientific Article | Gold | diseases | [link](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=FLnUx4cAAAAJ&citation_for_view=FLnUx4cAAAAJ:ufrVoPGSRksC) |
| [BioInfer](http://mars.cs.utu.fi/BioInfer/?q=download) | Scientific Article | Gold | genes/proteins | [link](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-50) |
| [BioSemantics](https://biosemantics.org/index.php/resources/chemical-patent-corpus) | Patent | Gold | chemicals, diseases | [link](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0107477) |
| [CDR](http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/) | Scientific Article | Gold | chemicals, diseases | [link](academic.oup.com/database/article/doi/10.1093/database/baw068/2630414) |
| [CellFinder](https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder) | Scientific Article | Gold | species, gene/proteins, cells, anatomy | [link](https://www.informatik.hu-berlin.de/de/forschung/gebiete/sar/wbi/research/publications/2012/lrec2012_corpus.pdf)|
|[CEMP](http://www.biocreative.org/tasks/biocreative-v/track-2-chemdner/)| Patent | Gold | chemicals|link|
|[DECA](http://www.nactem.ac.uk/deca/)| Scientific Article | Gold | gene/proteins |[link](http://bioinformatics.oxfordjournals.org/content/26/5/661.abstract?keytype=ref&ijkey=6nc2iFEN0sYYYz1)|
|[FSU-PRGE](http://pubannotation.org/projects/FSU-PRGE)| Scientific Article | Gold | genes/proteins|[link](http://aclweb.org/anthology/W/W10/W10-1838.pdf)|
|[Linneaus](http://linnaeus.sourceforge.net/)| Scientific Article | Gold | species | [link](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-85)|
| [IEPA](http://corpora.informatik.hu-berlin.de/corpora/brat2bioc/iepa_bioc.xml.zip) | Scientific Article | Gold | genes/proteins | [link](http://psb.stanford.edu/psb-online/proceedings/psb02/abstracts/p326.html) |
|[miRNA](http://www.scai.fraunhofer.de/mirna-corpora.html)| Scientific Article | Gold | diseases, species, genes/proteins | [link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4602280/) |
|[NCBI disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/)| Scientific Article | Gold | diseases|[link](http://www.sciencedirect.com/science/article/pii/S1532046413001974)|
|[S800](http://species.jensenlab.org/)| Scientific Article | Gold | species|[link](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0065390)|

### Supplementary Information

The supplementary data can be found in the file `supplementary/additional_file_1.pdf`. Additionally, blacklists used for the silver-standard corpora (SSCs) can be found in `supplementary/additional_file_2.zip`.

### Citations

1. Dernoncourt, F., Lee, J. Y., & Szolovits, P. (2017). NeuroNER: an easy-to-use program for named-entity recognition based on neural networks. arXiv preprint arXiv:1705.05487.
2. Moen, S. P. F. G. H., & Ananiadou, T. S. S. (2013). Distributional semantics resources for biomedical text processing. In Proceedings of the 5th International Symposium on Languages in Biology and Medicine, Tokyo, Japan (pp. 39-43).
