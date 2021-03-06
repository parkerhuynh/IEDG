data_config = {
    "data_dir": "./text-to-CDQL/data/data.csv",
    "batch_size": 64,
    "max_vocab_size": 5000,
    "test_sample": 256
}
 
model_config = {
    "embedding_dim":256,
    "hidden_units": 1024,
    "epoch":100,
    "model_name":"Bahdanau",
    "save_dir":"./text-to-CDQL/saved_model/translator",
    "learning_rate": 0.0001,
    "attention": "Bahdanau",
}