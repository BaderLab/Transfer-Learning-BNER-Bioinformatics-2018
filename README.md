# Transfer-learning-for-BNER-Bioinformatics-2018

This repository contains supplementary data, and links to the model and corpora used for the paper _[Transfer learning for biomedical named entity recognition with neural networks](https://academic.oup.com/bioinformatics/advance-article/doi/10.1093/bioinformatics/bty449/5026661)_.

### Code

Corpora pre-processing steps were collected in a single script with a jupyter notebook for ease-of-use. Script and notebook can be found in `code`.

#### Model

The model used in this study is __NeuroNER__ [[1](#citations)], a domain-independent named entity recognizer (NER) based on a bi-directional long short term memory network-conditional random field (LSTM-CRF). A repository for the model can be found [here](https://github.com/Franck-Dernoncourt/NeuroNER).

NeuroNER uses standard python config files to specify hyperparameters. We provide three of these config files for reproducibility (see `code/configs`):

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
| [AZDC](http://diego.asu.edu/downloads/AZDC_6-26-2009.txt) | Scientific Article | Gold | disease | [link](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=FLnUx4cAAAAJ&citation_for_view=FLnUx4cAAAAJ:ufrVoPGSRksC) |
| [BioCreative II GM](https://sourceforge.net/projects/biocreative/files/biocreative2entitytagging/1.1/) | Scientific Article | Gold | genes/proteins | [link](https://doi.org/10.1186/gb-2008-9-s2-s2) |
| [BioInfer](http://mars.cs.utu.fi/BioInfer/?q=download) | Scientific Article | Gold | genes/proteins | [link](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-50) |
| [BioSemantics](https://biosemantics.org/index.php/resources/chemical-patent-corpus) | Patent | Gold | chemicals, disease | [link](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0107477) |
| [CALBC-III-Small](http://ftp.ebi.ac.uk/pub/databases/CALBC/) | Scientific Article | Silver | chemicals, diseases, species, genes/proteins | [link](https://s3.amazonaws.com/academia.edu.documents/45849509/CALBC_silver_standard_corpus20160522-3059-1j189nl.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1537536482&Signature=hyYEo5%2BVtlPYeaNQwO5KP4o2HMY%3D&response-content-disposition=inline%3B%20filename%3DCalbc_Silver_Standard_Corpus.pdf) |
| [CDR](http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/) | Scientific Article | Gold | chemicals, diseases | [link](academic.oup.com/database/article/doi/10.1093/database/baw068/2630414) |
| [CellFinder](https://www.informatik.hu-berlin.de/de/forschung/gebiete/wbi/resources/cellfinder) | Scientific Article | Gold | species, gene/proteins, cells, anatomy | [link](https://www.informatik.hu-berlin.de/de/forschung/gebiete/sar/wbi/research/publications/2012/lrec2012_corpus.pdf)|
|[CHEMDNER Patent](http://www.biocreative.org/tasks/biocreative-v/track-2-chemdner/)| Patent | Gold | chemicals|[link](https://jcheminf.springeropen.com/articles/10.1186/1758-2946-7-S1-S2)|
|[DECA](http://www.nactem.ac.uk/deca/)| Scientific Article | Gold | gene/proteins |[link](http://bioinformatics.oxfordjournals.org/content/26/5/661.abstract?keytype=ref&ijkey=6nc2iFEN0sYYYz1)|
|[FSU-PRGE](http://pubannotation.org/projects/FSU-PRGE)| Scientific Article | Gold | genes/proteins|[link](http://aclweb.org/anthology/W/W10/W10-1838.pdf)|
|[Linneaus](http://linnaeus.sourceforge.net/)| Scientific Article | Gold | species | [link](http://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-11-85)|
|[LocText](https://www.tagtog.net/-corpora/loctext)| Scientific Article | Gold | species, genes/proteins | [link](http://bmcproc.biomedcentral.com/articles/10.1186/1753-6561-9-S5-A4)|
|[IEPA](http://corpora.informatik.hu-berlin.de/corpora/brat2bioc/iepa_bioc.xml.zip) | Scientific Article | Gold | genes/proteins | [link](http://psb.stanford.edu/psb-online/proceedings/psb02/abstracts/p326.html) |
|[miRNA](http://www.scai.fraunhofer.de/mirna-corpora.html)| Scientific Article | Gold | diseases, species, genes/proteins | [link](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4602280/) |
|[NCBI disease](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/)| Scientific Article | Gold | diseases|[link](http://www.sciencedirect.com/science/article/pii/S1532046413001974)|
|[S800](http://species.jensenlab.org/)| Scientific Article | Gold | species|[link](http://journals.plos.org/plosone/article?id=10.1371/journal.pone.0065390)|
|[Variome](http://www.opennicta.com.au/home/health/variome)| Scientific Article | Gold | diseases, species, genes/proteins|[link](http://database.oxfordjournals.org/content/2013/bat019.abstract)|

Many of these corpora can also be accessed and visualized in the browser [here](http://corpora.informatik.hu-berlin.de) [[3](#citations)].

### Supplementary Information

The supplementary data can be found in the file `supplementary/additional_file_1.pdf`. Additionally, blacklists used for the silver-standard corpora (SSCs) can be found in `supplementary/blacklists`.

### Citations

1. Dernoncourt, F., Lee, J. Y., & Szolovits, P. (2017). NeuroNER: an easy-to-use program for named-entity recognition based on neural networks. arXiv preprint arXiv:1705.05487.
2. Moen, S. P. F. G. H., & Ananiadou, T. S. S. (2013). Distributional semantics resources for biomedical text processing. In Proceedings of the 5th International Symposium on Languages in Biology and Medicine, Tokyo, Japan (pp. 39-43).
3. Stenetorp, P., Topić, G., Pyysalo, S., Ohta, T., Kim, J. D., & Tsujii, J. I. (2011, June). BioNLP shared task 2011: Supporting resources. In Proceedings of the BioNLP Shared Task 2011 Workshop (pp. 112-120). Association for Computational Linguistics.
