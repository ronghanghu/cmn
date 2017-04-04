#!/bin/bash
cd ./word_embedding
wget http://people.eecs.berkeley.edu/~ronghang/projects/cmn/word_embedding/embed_matrix.tgz
tar xvf embed_matrix.tgz
rm embed_matrix.tgz
cd ..
