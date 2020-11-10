from ncn.model import *
from ncn.training import *

SEED = 34

random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

data = get_bucketized_iterators("./data/arxiv_data.csv",
                                batch_size = 64,
                                len_context_vocab = 20000,
                                len_title_vocab = 20000,
                                len_aut_vocab = 20000)
PAD_IDX = data.ttl.vocab.stoi['<pad>']
cntxt_vocab_len = len(data.cntxt.vocab)
aut_vocab_len = len(data.aut.vocab)
ttl_vocab_len = len(data.ttl.vocab)

net = NeuralCitationNetwork(context_filters=[4,4,5,6,7],
                            author_filters=[1,2],
                            context_vocab_size=cntxt_vocab_len,
                            title_vocab_size=ttl_vocab_len,
                            author_vocab_size=aut_vocab_len,
                            pad_idx=PAD_IDX,
                            num_filters=128,
                            authors=True, 
                            embed_size=128,
                            num_layers=2,
                            hidden_size=128,
                            dropout_p=0.2,
                            show_attention=False)
net.to("cuda")

train_losses, valid_losses = train_model(model = net, 
                                         train_iterator = data.train_iter, 
                                         valid_iterator = data.valid_iter,
                                         lr = 0.001,
                                         pad = PAD_IDX,
                                         model_name = "want_to_reproduce_best",
                                         n_epochs=30)