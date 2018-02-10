#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import shutil
import codecs

from math import ceil
from collections import Counter
from nltk import word_tokenize, sent_tokenize
from gensim.models.keyedvectors import KeyedVectors
from sklearn.model_selection import train_test_split

# TODO (John): Write tests
# TODO (John): Change find_entity_in_corpra so it walks the directory path

## DATA CLEANING

def change_ann_labels(corpus_dir, labels_to_replace, new_label, drop=False):
    """
    Changes the label of each annotation <label_to_replace> with <new_label> for
    a given corpus.

    Args:
        corpus_dir: path to corpus
        label_to_replace: list, the textual annotation label(s) to replace
        new_label: str, the new textual annotation label to replace label_to_replace
        drop: bool, True, drops any annotations with entity lables not one of labels_to_replace
    """
    print('[INFO] Changing annotations...', end = '')
    for filename in get_filenames(corpus_dir):
        if filename.endswith(".ann"):
            # filepath of current file
            filepath = os.path.join(corpus_dir, filename)
            # read in annotation file at filepath
            with codecs.open(filepath, 'r', encoding = 'utf8') as f:
                # get all non-empty lines
                file_lines = [line for line in f.readlines() if len(line.strip()) != 0]
            # change all annotations in labels_to_replace to new_label
            with codecs.open(filepath, 'w+', encoding = 'utf8') as f:
                for line in file_lines:
                    tab_split = line.split('\t')
                    space_split = tab_split[1].split(' ')

                    annotation = space_split[0]
                    start = space_split[1]
                    end = space_split[-1]

                    if annotation in labels_to_replace:
                        tab_split[1] = '{} {} {}'.format(new_label, start, end)
                        f.write('\t'.join(tab_split))
                    else:
                        if not drop:
                            f.write(line)
    print('DONE')

# I believe these hang around after a file has been deleted...
def rm_hidden_files(corpus_dir):
    """
    Removes files with filenames beginning with '._'

    Args:
        corpus_dir: path to corpus
    """
    print('[INFO] Removing hidden files...', end = ' ')
    for filename in get_filenames(corpus_dir):
        filepath = os.path.join(filename, corpus_dir)
        # remove any hidden files
        if filename.startswith("._"):
            os.system('rm {}'.format(filepath))
            print('{}'.format(filename), end = '\r')
    print()
    print('[INFO] DONE.')

def rm_lone_pairs(corpus_dir):
    """
    Removes any lone .ann or .txt in corpus at input_dir

    Args:
        corpus_dir: path to corpus
    """
    print('[INFO] Removing lone pairs...')
    for filename in get_filenames(corpus_dir):
        filepath = os.path.join(corpus_dir, filename)
        # for each file in the corpus, if the corresponding
        # .txt or .ann file does not exist, remove its pair.
        if filename.endswith(".txt"):
            txt_filepath = filepath
            ann_filepath = filepath.replace('.txt', '.ann')
            ann_filename = filename.replace('.txt', '.ann')
            if not os.path.isfile(ann_filepath):
                print('{}'.format(ann_filename), end = '\r')
                os.system('rm {}'.format(txt_filepath))
        elif filename.endswith(".ann"):
            ann_filepath = filepath
            txt_filepath = filepath.replace('.ann', '.txt')
            txt_filename = filename.replace('.ann', '.txt')
            if not os.path.isfile(txt_filepath):
                print('{}'.format(txt_filename), end = '\r')
                os.system('rm {}'.format(ann_filepath))
    print()
    print('[INFO] DONE.')

def rm_invalid_ann(corpus_dir, remove=True):
    """
    Removes any annotation file which contains one or more invalid annotations.

    Args:
        corpus_dir: path to corpus
        remove: True if the invalid annotation file (and corresponding text file) should be removed.

    Returns:
        set of invalid filenames
    """
    # accumulator for invalid file names
    invalid_filenames = set()

    print('[INFO] Removing invalid .txt .ann pairs...', end = ' ')
    for filename in get_filenames(corpus_dir):
        if filename.endswith('.ann') and not filename.startswith('._'):
            filepath = os.path.join(corpus_dir, filename)
            # open <filename>.ann and <filename>.text
            ann_file = codecs.open(filepath, 'r', encoding = 'utf8')
            txt_file = codecs.open(filepath.replace('.ann', '.txt'), 'r', encoding = 'utf8')

            # read in contents of <filename>.ann and <filename>.text
            annotations = ann_file.readlines()
            text = txt_file.read()

            # if the annotated entity text in the annotation file does not exactly match
            # the labeled text in the text file, remove both <filename>.ann and <filename>.text
            for ann in annotations:
                split_line = ann.split('\t')
                start_idx = int(split_line[1].split(' ')[1])
                end_idx = int(split_line[1].split(' ')[2])
                entity = split_line[2].strip()

                if text[start_idx:end_idx] != entity:
                    ann_file.close()
                    txt_file.close()

                    invalid_filenames.add(filename)

                    print('{}'.format(filename.replace('.ann', '')), end = '\r')
                    if remove:
                        os.system('rm {}'.format(filename))
                        os.system('rm {}'.format(filename.replace('.ann', '.txt')))
                    # Early exit the loop. Only looking for 1 or more invalid annotations.
                    break

            # close files on each loop
            ann_file.close()
            txt_file.close()
    print()
    print('[INFO] DONE.')
    return invalid_filenames

def convert_bin_to_glove(input_file, output_dir=os.getcwd()):
    """
    Converts word embeddings given in the binary C format (w2v)
    to a text format that can be used with NeuroNER.

    Args:
        input_file: path to the word vectors file in binary C (w2v) format
        output_dir: path to save converted word vectors file (defaults to 
                    current working directory)
    """
    assert input_file.endswith('.bin'), "You need to provide a .bin file!"
    
    # load word vectors provided in C binary format
    word_vectors = KeyedVectors.load_word2vec_format(binary_w2v_file_path, binary=True)
    vocab = word_vectors.vocab

    # write contents of input_file to new file at output_file_path in glove format
    output_file_path = output_dir + '/converted_word_vectors.txt'
    with open(output_file_path, 'w+') as f:
        for word in vocab:
            vector = word_vectors[word]
            f.write("%s %s\n" %(word, " ".join(str(v) for v in vector)))
            
    print('[INFO] Converted C binary file saved to {}'.format(output_file_path))

## CORPUS PROCESSING

def print_top_bottom_n(counter, n = 10):
    """
    Prints the top n most common items in a counter

    Args:
        counter: counter object
        n: n most common items in counter to print
    """
    print('MOST COMMON')
    most_common = counter.most_common(n)
    for entity, count in most_common:
        print('{}: {}'.format(entity, count))
    print('LEAST COMMON')
    least_common = counter.most_common()[:-n-1:-1]
    for entity, count in least_common:
        print('{}: {}'.format(entity, count))
    return most_common, least_common

def find_entity_in_corpra(corpra_dir, entity_to_find, search_text = False):
    """Return all filenames in corpra_dir that contain entity_to_find

    Args:
        corpra_dir: path to corpus
        entity_to_find: string, textual entity to search for
        search_text: bool, set to True to search corpus text files as well as annotation files
    """
    for filename in os.listdir(corpra_dir):
        if filename.endswith(".ann") and not filename.startswith('.'):
            with open(os.path.join(corpra_dir, filename), 'r') as ann_file:
                ann_file_lines = ann_file.readlines()
                for line in ann_file_lines:
                    try:
                        entity = line.split('\t')[2].strip()
                        if entity == entity_to_find:
                            print("Found: '{}' at <{}>".format(entity_to_find, filename))
                    except IndexError:
                        print('Issue with: ', line)
        if search_text:
            if filename.endswith(".txt"):
                with open(os.path.join(corpra_dir, filename), 'r') as txt_file:
                    if entity_to_find in txt_file.read():
                        print("Found: '{}' at <{}>".format(entity_to_find, filename))

def split_brat_standoff(corpra_dir, train_size, test_size, valid_size, random_seed=42):
    """
    Randomly splits the corpus into train, test and validation sets.

    Args:
        corpus_dir: path to corpus
        train_size: float, train set parition size
        test_size: float, test set parition size
        valid_size: float, validation set parition size
        random_seed: optional, seed for random parition

    """
    assert train_size < 1.0 and train_size > 0.0, "TRAIN_SIZE must be between 0.0 and 1.0"
    assert test_size < 1.0 and test_size > 0.0, "TEST_SIZE must be between 0.0 and 1.0"
    assert valid_size < 1.0 and valid_size > 0.0, "VALID_SIZE must be between 0.0 and 1.0"
    assert ceil(train_size + test_size + valid_size) == 1, "TRAIN_SIZE, TEST_SIZE, and VALID_SIZE must sum to 1"

    print('[INFO] Moving to directory: {}'.format(corpra_dir))
    with cd(corpra_dir):
        print('[INFO] Getting all filenames in dataset...', end = ' ')
        # accumulators
        text_filenames = []
        ann_filenames =[]

        # get filenames
        for file in os.listdir():
            filename = os.fsdecode(file)
            if filename.endswith(".txt") and not filename.startswith('.'):
                text_filenames.append(filename)
            elif filename.endswith('.ann') and not filename.startswith('.'):
                ann_filenames.append(filename)

        assert len(text_filenames) == len(ann_filenames), '''Must be equal
            number of .txt and .ann files in corpus_dir'''

        # hackish way of making sure .txt and .ann files line up across the two lists
        text_filenames.sort()
        ann_filenames.sort()


        print('DONE')
        print('[INFO] Splitting corpus into {}% train, {}% test, {}% valid...'.format(train_size*100, test_size*100, valid_size*100),
        	end=' ')
        # split into train and all other, then split all other into test and valid
        X_train, X_test_and_valid = train_test_split(text_filenames, train_size=train_size, random_state = random_seed)
        y_train, y_test_and_valid = train_test_split(ann_filenames, train_size=train_size, random_state = random_seed)
        X_test, X_valid = train_test_split(X_test_and_valid, train_size=test_size/(1-train_size), random_state = random_seed)
        y_test, y_valid = train_test_split(y_test_and_valid, train_size=test_size/(1-train_size), random_state = random_seed)

        # leads to less for loops
        X_train.extend(y_train)
        X_test.extend(y_test)
        X_valid.extend(y_valid)

        print('Done.')
        print('[INFO] Creating train/test/valid directories at {} if they do not already exist...'.format(corpra_dir))
        # if they do not already exist
        os.makedirs('train', exist_ok=True)
        os.makedirs('test', exist_ok=True)
        os.makedirs('valid', exist_ok=True)

        for x in X_train:
            shutil.move(x, 'train/' + x)
        for x in X_test:
            shutil.move(x, 'test/' + x)
        for x in X_valid:
            shutil.move(x, 'valid/' + x)

## CORPUS STATS

def get_corpus_stats(corpus_dir):
    """
    Get corpus stats, including sentence and token counts
    for given corpus. Corpus must be in the brat standoff
    format, but can be split across multiple subdirectories
    in corpus_dir.

    Args:
        corpus_dir: path to corpus
    """
    # accumulators

    # counter object, holds all corpus stats
    c = Counter()
    # accumulators for unique tokens/annotations
    unique_tokens = set()
    unique_annotations = set()
    # accumulates paths to all files of a corpus
    filepaths = set()

    # get list of all filenames in the corpus
    for roots, _, files in os.walk(corpus_dir):
        root = os.fsdecode(roots)
        for file in files:
            filename = os.fsdecode(file)
            if filename.endswith('.txt') or filename.endswith('.ann'):
                filepath = os.path.join(root, filename)
                filepaths.add(filepath)

    for filepath in filepaths:
    	# loop over .txt files in directory, compute counts
        if filepath.endswith('.txt'):
            with open(os.path.join(corpus_dir, filepath), 'r') as tf:
                text_document = tf.read().strip()

            # generate list of tokens for document
            tokens = word_tokenize(text_document)
            # get list of sentences for document
            sentences = sent_tokenize(text_document)

            # update total counts
            c['sentence_count'] += len(sentences)
            c['total_token_count'] += len(tokens)
            # accumulate unique tokens
            unique_tokens = (unique_tokens | set(tokens))

        # loop over .ann files in directory, compute counts
        if filepath.endswith('.ann'):
            with open(os.path.join(corpus_dir, filepath), 'r') as af:
            	ann_document = [line for line in af.readlines() if line != '\n' and line != ' ' and line != '']

            # get list of annotations for document
            annotations = [line.split('\t')[2].strip() for line in ann_document]

            # update total count
            c['total_annotation_count'] += len(annotations)
            # accumulate unique tokens
            unique_annotations = (unique_annotations | set(annotations))

    # get unique total count
    c['unique_token_count'] = len(unique_tokens)
    c['unique_annotation_count'] = len(unique_annotations)

    print('TOTAL NUMBER OF DOCUMENTS: ', len(filepaths))
    print('Sentence count: ', c['sentence_count'])
    print('Total token count: ', c['total_token_count'])
    print('Total unique token count: ', c['unique_token_count'])
    print('Total annotation count: ', c['total_annotation_count'])
    print('Total unique annotation count: ', c['unique_annotation_count'])

## HARMONIZATION

def harmonize_ssc_gsc(GSC, SSC):
    pmid_in_GSC = set() # accumulator for all unique pmids

    # get all unique PMIDs in GSC
    for corpus in GSC:
        for _, _, files in os.walk(corpus):
            for file in files:
                filename = os.fsdecode(file)
                if filename.endswith(".ann"):
                    pmid_in_GSC.add(filename.replace('.ann', ''))

    # remove any PMIDs that appeard in the GSC from the SSC
    for pmid in pmid_in_GSC:
        ann_filepath = os.path.join(SSC, pmid + '.ann')
        txt_filepath = os.path.join(SSC, pmid + '.txt')

        if os.path.isfile(ann_filepath) or os.path.isfile(txt_filepath):
            os.system('rm {}'.format(ann_filepath))
            os.system('rm {}'.format(txt_filepath) )
            print('[INFO] Removing {} and {}...'.format(pmid + '.ann', pmid + '.txt'), end = '\r')

def read_blacklist(blacklist):
    """Returns all lines in a newline seperated text files as entities of a list.
    Removes any empty line (None, ' ', or '\n').
    """
    with open(blacklist, 'r') as rf:
        blacklist_entities = rf.readlines()
    # is there a better way to preform this check?
    return [line.strip() for line in blacklist_entities if line not in [None, '', '\n']]

def extract_text(corpra_dir):
    """Return all text (as elements of a set) from a given corpra.

    Corpra is returned as a str, where each document in the corpra
    is sperated by a newline.
    """
    corpra_text = set()
    for filename in os.listdir(corpra_dir):
        if filename.endswith(".txt") and not filename.startswith('.'):
            with open(os.path.join(corpra_dir, filename), 'r') as txt_file:
                print(filename, end = '\r')
                corpra_text.add(txt_file.read())
    return '\n\n'.join(corpra_text)

def extract_word_types(corpra_dir):
    """Return a sorted set of all word types from a coropra"""
    extracted_text = extract_text(corpra_dir)
    return sorted(set(word_tokenize(extracted_text)))

def extract_tokens(corpra_dir):
    """Return a sorted set of all tokens from a corpra"""
    extracted_text = extract_text(corpra_dir)
    return sorted(word_tokenize(extracted_text))

def disjoint_annotations(SSC, GSC):
    """Return entities which are present and annotated in the SSC,
    and present but not annotated in one more of the GSC"""
    # get the union of the GSC_word_types sets
    GSC_word_types = set().union(*[extract_word_types(corpus) for corpus in GSC])

    # get all annotated entities in the GSC
    GSC_ann = Counter()
    for corpus in GSC:
        GSC_ann += extract_ann(corpus)
    # get all annotated entities in the SSC
    SSC_ann = extract_ann(SSC)

    # accumulator of disjoint annotations
    entities_SSC_not_GSC = set()

    for entity in SSC_ann:
        # if entity is present in any of the GSC but is not annotated, add it to our set
        if entity in GSC_word_types and GSC_ann[entity] == 0:
            entities_SSC_not_GSC.add(entity)

    return entities_SSC_not_GSC

def rm_disjoint_annotations(SSC, blacklist_path):

    # get blacklisted entities
    blacklist = read_blacklist(blacklist_path)
    for filename in get_filenames(SSC):
        print('[INFO] Processing file: {}'.format(filename), end = '\r')
        if filename.endswith('.ann') and not filename.startswith('._'):
            filepath = os.path.join(SSC, filename)
            with open(filepath, 'r') as rf:
                file_lines = rf.readlines()
            with open(filepath, 'w') as wf:
                for line in file_lines:
                    entity_label = line.split('\t')[2].strip().lower()
                    # write the line back to filename.ann iff it does not contain a blacklisted entity
                    if entity_label not in blacklist: wf.write(line)
                    else: print('[INFO] Removing annotation: {} from file: {}'.format(entity_label, filename))

## EVALUATION

def get_labels(path_to_labels):
	"""
	Returns a list of strings containing the annotations from file at
	path_to_labels with TX counter removed.

	Args:
		path_to_labels: path to annotation file in brat-standoff format

	Returns:
		list of strings containing annotations from file at path_to_labels
	"""
	global_labels = []

	for file in os.listdir(path_to_labels):
		filename = os.fsdecode(file)
		if filename.endswith(".ann") or filename.endswith('.a1'):
			with codecs.open(os.path.join(path_to_labels, filename), 'r',
				encoding='utf-8') as test:
				labels = test.readlines()

			# remove TX counter, add predictions\labels to global list
			global_labels.extend(['\t'.join(x.split('\t')[1:]).strip() for
				x in labels])

	return global_labels

def get_FN_FP_TP(predictions, labels):
	"""
	Returns tuple of lists containing false-negatives, false-positives and
	true-positives.

	Args:
		predictions: list of entity predictions in brat-standoff format
		labels: list of entity gold standard labels in brat-standoff format

	Returns:
		3-tuple of sets, containing FN, FP and TP.
	"""
	# accumulators
	FN = set()
	FP = set()
	TP = set()

	# FN
	for label in labels:
		if not label in predictions:
			FN.add(label)

	# FP
	for pred in predictions:
		if not pred in labels:
			FP.add(pred)

	# TP
	for pred in predictions:
		if pred in labels:
			TP.add(pred)

	return FN, FP, TP

def get_top_n_intersection(A, B, n = 10):
	"""
	Returns a list containing the n most common elements in A intersection B.

	Args:
		A: first set
		B: second set

	Returns:
		list of tuples containing the n most common elements in A intersection B
	"""
	A_intersection_B = Counter([x.split('\t')[1] for x in
		(A.intersection(B))]).most_common(n)
	return A_intersection_B

def get_top_n_difference(A, B, n = 10):
	"""
	Returns a tuple of lists, where the first list contains the n most common elements
	in A \ B and the second list contains the n most common elements in B \ A.

	Args:
		A: first set
		B: second set

	Returns:
		2-tuple of sets containing relative complements, A \ B and B \ A
	"""
	A_minus_B = Counter([x.split('\t')[1] for x in (A - B)]).most_common(n)
	B_minus_A = Counter([x.split('\t')[1] for x in (B - A)]).most_common(n)
	return A_minus_B, B_minus_A


def write_FP_FN_to_disk(errors):
	"""
	Writes FP and FN to disk

	Args:
		errors: output from get_FP_and_FN()
	"""
	pass

def get_FN_FP_TP_single_corpus(path_to_pred, path_to_gold):
	"""
	Returns sets of FN, FP and FP for predictions at path_to_pred
	for ground truth labels at path_to_gold.

	Args:
		path_to_pred: path to annotation predictions in brat-standoff format
		path_to_gold: path to annotation ground-truth in brat-standoff format

	Returns:
		3-tuple of FN, FP and TP for predictions based on ground truth.
	"""
	# get predictions and labels
	predictions, labels = get_labels(path_to_pred), get_labels(path_to_gold)
	# get FP, FN, TP
	FN, FP, TP = get_FN_FP_TP(predictions, labels)

	return FN, FP, TP

def print_corpus_eval(path_to_pred_baseline, path_to_pred_tl, path_to_gold):
	"""
	Prints total counts and contents of sets of FN, FP and FP for predictions at
	path_to_pred_baseline, path_to_pred_tl compared to ground truth labels
	at path_to_gold.

	Args:
		path_to_pred_baseline: path to annotation predictions in brat-standoff
		format for baseline method
		path_to_pred_tl: path to annotation predictions in brat-standoff
		format for transfer learning method
		path_to_gold: path to annotation ground-truth in brat-standoff format
	"""
	# get FP, FN, TP for baseline
	FN_baseline, FP_baseline, TP_baseline = get_FN_FP_TP_single_corpus(path_to_pred_baseline, path_to_gold)
	# get FP, FN, TP for transfer learning
	FN_tl, FP_tl, TP_tl = get_FN_FP_TP_single_corpus(path_to_pred_tl, path_to_gold)

	print('BASELINE\n' + '-'*50)
	print('FNs: {}'.format(len(FN_baseline)))
	print('FPs: {}'.format(len(FP_baseline)))
	print('TPs: {}'.format(len(TP_baseline)))
	print()

	print('TRANSFER LEARNING\n' + '-'*50)
	print('FNs: {}'.format(len(FN_tl)))
	print('FPs: {}'.format(len(FP_tl)))
	print('TPs: {}'.format(len(TP_tl)))
	print()

	print('INTERSECTION\n' + '-'*50)
	print('FNs: {}'.format(len(FN_baseline.intersection(FN_tl))))
	print('FPs: {}'.format(len(FP_baseline.intersection(FP_tl))))
	print('TPs: {}'.format(len(TP_baseline.intersection(TP_tl))))


## UTILITY METHODS/CLASSES

class cd:
    """Context manager for changing the current working directory."""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def get_filenames(input_dir):
    """
    Returns list of filenames in input_dir.

    Args:
        input_dir: path to input directory
    """
    return [os.fsdecode(file) for file in os.listdir(input_dir)]

def extract_ann(corpra_dir):
    """Returns a Counter object, where keys are
    unique textual annotations in copra at copra_dir,
    and values are their respective counts.

    Args:
        corpus_dir: path to corpus
    """
    unique_entities_in_corpra = Counter()
    for filename in os.listdir(corpra_dir):
        if filename.endswith(".ann") or filename.endswith('.a1'):
            try:
                with open(os.path.join(corpra_dir, filename), 'r') as ann_file:
                    ann_file_lines = ann_file.readlines()
                    for line in ann_file_lines:
                        try:
                            entity = line.split('\t')[2].strip()
                            unique_entities_in_corpra[entity] += 1
                        except IndexError:
                            print('Issue with: {}'.format(line))
            except UnicodeDecodeError:
                print('UnicodeDecodeError raised when trying to read file {}'.format(filename))
    return unique_entities_in_corpra

def average_len_of_iterable(iterable):
    """ Returns the average length of string elements in interable. Returns 0
    if the iterable is empty.

    Args:
        iterable: iterable with str elements

    Returns:
        average length of string elements in iterable

    Preconditions:
        elements of the iterable are str.
    """
    lengths = [len(i) for i in iterable]
    return 0 if len(lengths) == 0 else (float(sum(lengths)) / len(lengths))
