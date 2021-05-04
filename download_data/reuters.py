# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/usr/bin/env python3

import argparse
import logging
import os
import xml.etree.ElementTree as ET

INDICES = "C:/Users/Lodewijk/Desktop/scriptie/ReutersCorpus/mldoc-indices/english.train.10000"
OUTPUT = "C:/Users/Lodewijk/Desktop/scriptie/GNN-document-classification/data/reuters.train.10000.en"

DO_ENCODING = False

RAW_DIR = "C:/Users/Lodewijk/Desktop/scriptie/ReutersCorpus/RCV2_Multilingual_Corpus/english"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rcv-dir',
        dest='rcv_dir',
        help='Directory of rcv1/rcv2 corpus with sub-directory structure '
             'indicated by indices, e.g. index FDCH5-39373 corresponds to '
             'document <rcv_dir>/FDCH5/39373.xml.',
        default=RAW_DIR
    )
    parser.add_argument(
        '--output-filename',
        dest='output_filename',
        help='Path to store documents being indexed',
        default=OUTPUT
    )
    parser.add_argument(
        '--indices-file',
        dest='indices_file',
        help='Path to indices file.',
        default=INDICES
    )
    args = parser.parse_args()

    delim_str = '\t'
    sentence_delim = ' '
    code_class = 'bip:topics:1.0'
    labels = ['C', 'E', 'G', 'M']
    target_topics = ['{}CAT'.format(label) for label in labels]
    with open(args.indices_file, 'r', encoding="utf8") as indices_f, \
            open(args.output_filename, 'w', encoding="utf8") as output_f:
        for line in indices_f:
            sub_corpus, file_name = line.strip().split('-')
            sub_corpus_path = os.sep.join([args.rcv_dir, sub_corpus])
            doc_path = os.sep.join(
                [sub_corpus_path, '{}.xml'.format(file_name)]
            )
            data_str = open(doc_path, encoding="utf8").read()
            try:
                xml_parsed = ET.fromstring(data_str)
                topics = [
                    topic.attrib['code'] for topic in xml_parsed.findall(
                        ".//codes[@class='{}']/code".format(code_class)
                    ) if topic.attrib['code'] in target_topics
                ]
                assert len(topics) == 1, 'More than one class label found.'
                doc = sentence_delim.join(
                    [p.text for p in xml_parsed.findall(".//p")]
                )

                if DO_ENCODING:
                    output_f.write(
                        '{}{}{}\n'.format(topics[0], delim_str, doc.encode('utf-8'))
                    )
                else:
                    output_f.write(
                        '{}{}{}\n'.format(topics[0], delim_str, doc)
                    )
            except Exception as e:
                logging.error('Failed to parse xml file: {}.'.format(doc_path))

if __name__ == '__main__':
    main()
