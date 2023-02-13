# environment names
VPN = 'vpn'
QUAN_ZHOU = 'quanzhou'
NEI_MENG = 'neimeng'
BEI_JING = 'beijing'

# aug names
KEYWORD_ENHANCE = 'keyword_enhance'
RANDOM_WORD_DROPOUT = 'random_word_dropout'
TFIDF_WORD_DROPOUT = 'tfidf_word_dropout'
FEATURE_DATA_AUGMENTATIONS = [KEYWORD_ENHANCE, RANDOM_WORD_DROPOUT, TFIDF_WORD_DROPOUT]

RANDOM_DELETION = 'random_deletion'
RANDOM_SWAP = 'random_swap'
RANDOM_INSERTION = 'random_insertion'
SYNONYM_REPLACEMENT = 'synonym_replacement'

# model names
DISTILBERT = 'distilbert'
CNN = 'cnn'
LSTM = 'lstm'
RNN = 'rnn'


# parameter_names
TFIDFS = 'tfidfs'
KEYWORD_IDS = 'keyword_ids'
DROPOUT_PROB = 'dropout_prob'
EX_FEATURE_NAMES = [TFIDFS, KEYWORD_IDS, DROPOUT_PROB]