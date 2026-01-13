#!/bin/bash

MOSES_GIT_URL=${MOSES_GIT_URL:-https://github.com/moses-smt/mosesdecoder.git}
SUBWORD_GIT_URL=${SUBWORD_GIT_URL:-https://github.com/rsennrich/subword-nmt.git}
SEQ2SEQ_GIT_URL=${SEQ2SEQ_GIT_URL:-https://github.com/google/seq2seq.git}

MOSES_TOKENIZER=${MOSES_SCRIPT:-mosesdecoder/scripts/tokenizer/tokenizer.perl}
MOSES_CLEAN=${MOSES_CLEAN:-mosesdecoder/scripts/training/clean-corpus-n.perl}
VOCAB_SKRIPT=${VOCAB_SKRIPT:-seq2seq/bin/tools/generate_vocab.py}

LEARN_BPE_SKRIPT=${LEARN_BPE_SKRIPT:-subword-nmt/learn_bpe.py}
APPLY_BPE_SKRIPT=${APPLY_BPE_SKRIPT:-subword-nmt/apply_bpe.py}
GET_VOCAB_SKRIP=${GET_VOCAB_SKRIP:-subword-nmt/get_vocab.py}
NUM_THREADS=${NUM_THREADS:-8}
PERL=${PERL:-perl}
PYTHON=${PYTHON:-python}

SOURCE=corpus.sa
TARGET=corpus.hi

cp $1 $SOURCE
cp $2 $TARGET

if [ "$#" -ne 2 ]; then
  echo "Source and target files must be passed"
  exit 1
fi

if [ ! -d "mosesdecoder" ]; then
  git clone ${MOSES_GIT_URL}
fi

if [ ! -d "subword-nmt" ]; then
  git clone ${SUBWORD_GIT_URL}
fi

if [ ! -d "seq2seq" ]; then
  git clone ${SEQ2SEQ_GIT_URL}
fi

cat $SOURCE | ${PERL} ${MOSES_TOKENIZER} -threads ${NUM_THREADS} -l sa > $SOURCE.tok
cat $TARGET | ${PERL} ${MOSES_TOKENIZER} -threads ${NUM_THREADS} -l hi > $TARGET.tok

mv $SOURCE.tok $SOURCE
mv $TARGET.tok $TARGET

${PERL} ${MOSES_CLEAN} corpus sa hi "corpus.clean" 1 500

${PYTHON} ${VOCAB_SKRIPT} --max_vocab_size 50000 < corpus.clean.hi > vocab.50k.hi
${PYTHON} ${VOCAB_SKRIPT} --max_vocab_size 50000 < corpus.clean.sa > vocab.50k.sa

# Learn Shared BPE
for merge_ops in 32000; do
  echo "Learning BPE with merge_ops=${merge_ops}. This may take a while..."
  cat "corpus.clean.hi" "corpus.clean.sa" | \
    ${PYTHON} ${LEARN_BPE_SKRIPT} -s $merge_ops > "bpe.${merge_ops}"

  echo "Apply BPE with merge_ops=${merge_ops} to tokenized files..."
  for lang in hi sa; do
    for f in "corpus.${lang}" "corpus.clean.${lang}"; do
      outfile="${f%.*}.bpe.${merge_ops}.${lang}"
      ${PYTHON} ${APPLY_BPE_SKRIPT} -c "bpe.${merge_ops}" < $f > "${outfile}"
      echo ${outfile}
    done
  done

  # Create vocabulary file for BPE
  cat "corpus.clean.bpe.32000.hi" "corpus.clean.bpe.32000.sa" | \
    ${PYTHON} ${GET_VOCAB_SKRIP} | cut -f1 -d ' ' > "vocab.bpe.${merge_ops}"
done
