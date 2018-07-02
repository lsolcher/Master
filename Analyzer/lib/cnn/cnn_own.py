import tensorflow as tf

# Define Inputs
inputs_ = tf.placeholder(tf.int32, [None, seq_len], name='inputs')
labels_ = tf.placeholder(tf.float32, [None, 1], name='labels—)
training_ = tf.placeholder(tf.bool, name='training')

# Define Embeddings
embedding = tf.Variable(tf.random_uniform((vocab_size, embed_size), -1, 1))
embed = tf.nn.embedding_lookup(embedding, inputs_)