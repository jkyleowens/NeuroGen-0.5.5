import sentencepiece as spm
spm.SentencePieceTrainer.train(
    input='data.txt',
    model_prefix='mymodel',
    vocab_size=8000,
    model_type='bpe',
    character_coverage=1.0
)