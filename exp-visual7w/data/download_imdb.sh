#!/bin/bash
cd ./exp-visual7w/data
wget http://people.eecs.berkeley.edu/~ronghang/projects/cmn/data/exp-visual7w/imdb.tgz
wget http://people.eecs.berkeley.edu/~ronghang/projects/cmn/data/exp-visual7w/all_qa_pairs_relationship.json
tar xvf imdb.tgz
cd ../..
