# environment names
LOCAL = 'local'
VPN = 'vpn'
QUAN_ZHOU = 'quanzhou'
NEI_MENG = 'neimeng'
BEI_JING = 'beijing'
NAN_JING = 'nanjing'

# aug names
KEYWORD_ENHANCE = 'keyword_enhance'
RANDOM_WORD_DROPOUT = 'random_word_dropout'
TFIDF_WORD_DROPOUT = 'tfidf_word_dropout'
HIDDEN_STATE_POOLING = 'hidden_state_pooling'
HIDDEN_STATE_CNN = 'hidden_state_cnn'
LOSS_BASED_REPLACEMENT = 'loss_based_replacement'
PRED_BASED_REPLACEMENT = 'pred_based_replacement'
PRED_LOSS_REPLACEMENT = 'pred_loss_replacement'
ONLINE_RANDOM_REPLACEMENT = 'online_random_replacement'
ONLINE_DATA_AUGMENTATIONS = [KEYWORD_ENHANCE, RANDOM_WORD_DROPOUT, TFIDF_WORD_DROPOUT, HIDDEN_STATE_POOLING, HIDDEN_STATE_CNN,
                             LOSS_BASED_REPLACEMENT, PRED_BASED_REPLACEMENT, ONLINE_RANDOM_REPLACEMENT, PRED_LOSS_REPLACEMENT]

RANDOM_DELETION = 'random_deletion'
RANDOM_SWAP = 'random_swap'
RANDOM_INSERTION = 'random_insertion'
SYNONYM_REPLACEMENT = 'synonym_replacement'

# model names
DISTILBERT = 'distilbert'
CNN = 'cnn'
LSTM = 'lstm'
RNN = 'rnn'
ROBERTA = 'roberta'
ALBERT = 'albert'
XLNET = 'xlnet'
ELECTRA = 'electra'

# parameter_names
TFIDFS = 'tfidfs'
KEYWORD_IDS = 'keyword_ids'
DROPOUT_PROB = 'dropout_prob'
REPLACEMENT_PROB = 'replacement_prob'
EX_FEATURE_NAMES = [TFIDFS, KEYWORD_IDS, DROPOUT_PROB, REPLACEMENT_PROB]