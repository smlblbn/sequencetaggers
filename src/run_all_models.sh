#!/usr/bin/env bash

mkdir -p ../logs

date "+%Y-%m-%d %H:%M:%S"

python tagger_lr_pos.py &> ../logs/tagger_lr_pos.log
date "+%Y-%m-%d %H:%M:%S"

python tagger_lr_chunk.py &> ../logs/tagger_lr_chunk.log
date "+%Y-%m-%d %H:%M:%S"

python tagger_lr_ner.py &> ../logs/tagger_lr_ner.log
date "+%Y-%m-%d %H:%M:%S"

python tagger_crf_pos.py &> ../logs/tagger_crf_pos.log
date "+%Y-%m-%d %H:%M:%S"

python tagger_crf_chunk.py &> ../logs/tagger_crf_chunk.log
date "+%Y-%m-%d %H:%M:%S"

python tagger_crf_ner.py &> ../logs/tagger_crf_ner.log
date "+%Y-%m-%d %H:%M:%S"