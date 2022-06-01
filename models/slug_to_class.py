from models import ggnn
from models import gru
from models import transformer
from models import gcn
from models import gat
from models import gat_gru

SLUG_TO_CLASS = {
    "ggnn_bop": ggnn.GGNNForBOP,
    "ggnn_nap": ggnn.GGNNForNAP,
    "ggnn_ntp": ggnn.GGNNForNTP,
    "gcn_bop": gcn.GCNForBOP,
    "gcn_nap": gcn.GCNForNAP,
    "gcn_ntp": gcn.GCNForNTP,
    "gat_bop": gat.GATForBOP,
    "gat_nap": gat.GATForNAP,
    "gat_ntp": gat.GATForNTP,
    "gru_nap": gru.GRUForNAP,
    "gru_bop": gru.GRUForBOP,
    "gru_ntp": gru.GRUForNTP,
    "gat_gru_nap": gat_gru.GATGRUForNAP,
    "transformer_bop": transformer.TransformerForBOP,
    "transformer_nap": transformer.TransformerForNAP,
    "transformer_ntp": transformer.TransformerForNTP,
}
