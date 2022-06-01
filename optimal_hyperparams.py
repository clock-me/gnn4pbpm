OPT_HYPERPARAMS = {
    "ggnn": {
        "num_timestamps": 5,
        "num_node_types": 50,
        "hidden_size": 64,
        "num_activities": 14,
        "learning_rate": 1e-5,
        "add_freq_features": True,
    },
    "gat_gru": {
        "learning_rate": 5e-4,
        "num_activities": 18,
        "hidden_size": 64,
        "num_gru_layers": 2, 
        "num_node_types": 50,
        "num_gat_heads": 4,
        "num_gat_layers": 2,
    },
    "gru": {
        "num_activities": 14,
        "num_layers": 2,
        "hidden_size": 64,
        "learning_rate": 0.0005
    },
    "transformer": {
        "num_activities": 19,
        "num_layers": 2,
        "num_heads": 4,
        "hidden_size": 64,
        "dropout": 0.1,
        "learning_rate": 0.0002
    },
    "gat": {
        "hidden_size": 64,
        "learning_rate": 3e-3,
        "num_activities": 14,
        "num_node_types": 50,
        "num_layers": 2,
        "num_heads": 4,
        "add_freq_features": True,
        "add_time_features": True,
        "add_type_features": True,
    },
    "gcn": {
        "hidden_size": 64,
        "num_node_types": 50,
        "learning_rate": 1e-4,
        "num_activities": 14,
        "num_layers": 3,
        "add_freq_features": True,
    }
}