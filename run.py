import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import re
import time
import logging
tf.__version__

FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
logging.basicConfig(filename="assistant.log",
                    filemode='a',
                    format=FORMAT,
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Loading movie lines.")
lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')
logging.info("Loaded movie lines.")

# Dictionary to map each line's id with text
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Create a list of all conversation's lines' ids.
convs = [ ]
for line in conv_lines[:-1]:
    _line = line.split(' +++$+++ ')[-1][1:-1].replace("'","").replace(" ","")
    convs.append(_line.split(','))

# Sort sentences into questions (inputs) and answers (targets)
questions = []
answers = []

for conv in convs:
    for i in range(len(conv)-1):
        questions.append(id2line[conv[i]])
        answers.append(id2line[conv[i+1]])

# Top of loaded dat
limit = 0
for i in range(limit, limit+5):
    logging.debug("Q - {}".format(questions[i]))
    logging.debug("A - {}".format(answers[i]))
    logging.debug("")

assert(len(questions) == len(answers))

def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)
    
    return text

# Clean given data
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

# Look at some data to ensure it was cleaned properly
for i in range(limit, limit + 5):
    logging.debug("Q - {}".format(clean_questions[i]))
    logging.debug("A - {}".format(clean_answers[i]))
    logging.debug("")

# Find sentence lengths
lengths = []
for question in clean_questions:
    lengths.append(len(question.split()))
for answer in clean_answers:
    lengths.append(len(answer.split()))

# Make dataframe
lengths = pd.DataFrame(lengths, columns=['counts'])

logging.info(lengths.describe())

logging.info("80th Percentile Length: {}".format(np.percentile(lengths, 80)))
logging.info("85th Percentile Length: {}".format(np.percentile(lengths, 85)))
logging.info("90th Percentile Length: {}".format(np.percentile(lengths, 90)))
logging.info("95th Percentile Length: {}".format(np.percentile(lengths, 95)))
logging.info("99th Percentile Length: {}".format(np.percentile(lengths, 99)))

# Cull q + a's shorter than 2 words and longer than 20 words
min_line_length = 2
max_line_length = 20

# Filter out questions too short or long
short_questions_temp = []
short_answers_temp = []

i=0
# TODO: Do better
for question in clean_questions:
    if len(question.split()) >= min_line_length and len(question.split()) <= max_line_length:
        short_questions_temp.append(question)
        short_answers_temp.append(clean_answers[i])
    i+=1

# Filter answers too short/long
short_questions = []
short_answers = []

i = 0
for answer in short_answers_temp:
    if len(answer.split()) >= min_line_length and len(answer.split()) <= max_line_length:
        short_answers.append(answer)
        short_questions.append(short_questions_temp[i])
    i += 1

logging.info("# of questions:\t{}".format(len(short_questions)))
logging.info("# of answers:\t{}".format(len(short_answers)))
logging.info("% of data used: {}%".format(round(len(short_questions)/len(questions), 4) * 100))

vocab = {}
for question in short_questions:
    for word in question.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

for answer in short_answers:
    for word in answer.split():
        if word not in vocab:
            vocab[word] = 1
        else:
            vocab[word] += 1

# Remove rare words from vocabulary.
threshold = 10
count = 0
for k,v in vocab.items():
    if v >= threshold:
        count += 1

logging.info("Size of total vocab: {}".format(len(vocab)))
logging.info("Size of vocab used: {}".format(count))

# To use different vocab sizes for source and target text,
# set different threshold value.
# Create dictionaries to provide unique int for each word
questions_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        questions_vocab_to_int[word] = word_num
        word_num += 1

answers_vocab_to_int = {}

word_num = 0
for word, count in vocab.items():
    if count >= threshold:
        answers_vocab_to_int[word] = word_num
        word_num += 1

codes = ['<PAD>', '<EOS>', '<UNK>', '<GO>']

for code in codes:
    questions_vocab_to_int[code] = len(questions_vocab_to_int) + 1
    answers_vocab_to_int[code] = len(answers_vocab_to_int) + 1

# Create dictionaries to map unique ints to respective word
questions_int_to_vocab = {v_i: v for v, v_i in questions_vocab_to_int.items()}
answers_int_to_vocab = {v_i: v for v, v_i in answers_vocab_to_int.items()}

logging.debug("len(questions_vocab_to_int)={}".format(len(questions_vocab_to_int)))
logging.debug("len(questions_int_to_vocab)={}".format(len(questions_int_to_vocab)))
logging.debug("len(answers_vocab_to_int)={}".format(len(answers_vocab_to_int)))
logging.debug("len(answers_int_to_vocab)={}".format(len(answers_int_to_vocab)))

# Add end of sentence token to every answer
for i in range(len(short_answers)):
    short_answers[i] += ' <EOS>'

# Convert text to ints
# Replace words not in vocabulary with <UNK>
questions_int = []
for question in short_questions:
    ints = []
    for word in question.split():
        if word not in questions_vocab_to_int:
            ints.append(questions_vocab_to_int['<UNK>'])
        else:
            ints.append(questions_vocab_to_int[word])
    questions_int.append(ints)

answers_int = []
for answer in short_answers:
    ints = []
    for word in answer.split():
        if word not in answers_vocab_to_int:
            ints.append(answers_vocab_to_int['<UNK>'])
        else:
            ints.append(answers_vocab_to_int[word])
    answers_int.append(ints)

logging.info("Length of questions: {}".format(len(questions_int)))
logging.info("Length of answers: {}".format(len(answers_int)))

# Calc percentage of words replaced by <UNK>
word_count = 0
unk_count = 0

for question in questions_int:
    for word in question:
        if word == questions_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1

for answer in answers_int:
    for word in answer:
        if word == answers_vocab_to_int["<UNK>"]:
            unk_count += 1
        word_count += 1

unk_ratio = round(unk_count/word_count, 4)*100

logging.info("Total word count: {}".format(word_count))
logging.info("Number of times <UNK> is used: {}".format(unk_count))
logging.info("Percent of words that are <UNK>: {}%".format(round(unk_ratio,3)))

# Sort questions and answers by the question length.
# This serves to reduce padding during training
# Which should speed up training and help reduce loss

sorted_questions = []
sorted_answers = []

for length in range(1, max_line_length+1):
    for i in enumerate(questions_int):
        if len(i[1]) == length:
            sorted_questions.append(questions_int[i[0]])
            sorted_answers.append(answers_int[i[0]])

logging.debug("Sorted questions length is {}".format(len(sorted_questions)))
logging.debug("Sorted answers length is {}".format(len(sorted_answers)))
logging.debug("")
for i in range(3):
    logging.debug("Q - {}".format(sorted_questions[i]))
    logging.debug("A - {}".format(sorted_answers[i]))


tf.compat.v1.disable_eager_execution()
def model_inputs():
    input_data = tf.compat.v1.placeholder(tf.int32,
                                [None, None],
                                name='input')

    targets = tf.compat.v1.placeholder(tf.int32,
                             [None, None],
                             name='targets')

    lr = tf.compat.v1.placeholder(tf.float32, name='learning_rate')

    keep_prob = tf.compat.v1.placeholder(tf.float32, name='keep_prob')

    return input_data, targets, lr, keep_prob

def process_encoding_input(target_data, vocab_to_int, batch_size):
    ending = tf.strided_slice(target_data,
                              [0,0],
                              [batch_size, -1],
                              [1,1])

    dec_input = tf.concat([tf.fill([batch_size, 1],
                                    vocab_to_int['<GO>']),
                           ending], 1)

    return dec_input

def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob,
                   sequence_length, attn_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)

    drop = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

    _, enc_state = tf.nn.bidirectional_dynamic_rnn(
        cell_fw = enc_cell,
        cell_bw = enc_cell,
        sequence_length = sequence_length,
        inputs = rnn_inputs,
        dtype=tf.float32
    )

    return enc_state

def decoding_layer_train(encoder_state, dec_cell, dec_embed_input,
                         sequence_length, decoding_scope,
                         output_fn, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,
                                1,
                                dec_cell.output_size])

    att_keys, att_vals, att_score_fn, att_construct_fn = \
        tf.contrib.seq2seq.prepare_attention(
            attention_states,
            attention_option="bahdanau",
            num_units=dec_cell.output_size
        )

    train_decoder_fn = \
        tf.contrib.seq2seq.attention_decoder_fn_train(
            encoder_state[0],
            att_keys,
            att_vals,
            att_score_fn,
            att_construct_fn,
            name = "attn_dec_train"
        )

    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        dec_cell,
        train_embed_input,
        sequence_length,
        scope=decoding_scope)
    
    train_pred_drop = tf.nn.dropout(train_pred, keep_prob)

    return output_fn(train_pred_drop)

def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings,
                         start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope,
                         output_fn, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size,
                                1,
                                dec_cell.output_size])

    att_keys, att_vals, att_score_fn, att_construct_fn = \
        tf.contrib.seq2seq.prepare_attention(
            attention_states,
            attention_option="bahdanau",
            num_units=dec_cell.output_size
        )

    infer_decoder_fn = \
        tf.contrib.seq2seq.attention_decoder_fn_inference(
            output_fn,
            encoder_state[0],
            att_keys,
            att_vals,
            att_score_fn,
            att_construct_fn,
            dec_embeddings,
            start_of_sequence_id,
            end_of_sequence_id,
            maximum_length,
            vocab_size,
            name="attn_dec_inf"
        )

    infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
        dec_cell,
        infer_decoder_fn,
        scope=decoding_scope
    )

    return infer_logits

def decoding_layer(dec_embed_input, dec_embeddings, encoder_state,
                   vocab_size, sequence_length, rnn_size,
                   num_layers, vocab_to_int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(
            lstm,
            input_keep_prob = keep_prob
        )
        dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_fn = lambda x: tf.contrib.layers.fully_connected(
            x,
            vocab_size,
            None,
            scope=decoding_scope,
            weights_initializer = weights,
            biases_initializer = biases
        )

    train_logits = decoding_layer_train(encoder_state,
                                        dec_cell,
                                        dec_embed_input,
                                        sequence_length,
                                        decoding_scope,
                                        output_fn,
                                        keep_prob,
                                        batch_size)
    decoding_scope.reuse_variables()

    infer_logits = decoding_layer_infer(encoder_state,
                                        dec_cell,
                                        dec_embeddings,
                                        vocab_to_int['<GO>'],
                                        vocab_to_int['<EOS>'],
                                        sequence_length - 1,
                                        vocab_size,
                                        decoding_scope,
                                        output_fn,
                                        keep_prob,
                                        batch_size)

    return train_logits, infer_logits

def seq2seq_model(input_data, target_data, keep_prob, batch_size,
                  sequence_length, answers_vocab_size,
                  questions_vocab_size, enc_embedding_size,
                  dec_embedding_size, rnn_size, num_layers,
                  questions_vocab_to_int):
    enc_embed_input = tf.contrib.layers.embed_sequence(
        input_data,
        answers_vocab_size+1,
        enc_embedding_size,
        initializer = tf.random_uniform_initializer(-1,1)
    )

    enc_state = encoding_layer(enc_embed_input,
                               rnn_size,
                               num_layers,
                               keep_prob,
                               sequence_length)

    dec_input = process_encoding_input(target_data,
                                       questions_vocab_to_int,
                                       batch_size)

    dec_embeddings = tf.Variable(
        tf.random_uniform([questions_vocab_size+1,
                           dec_embedding_size],
                           -1, 1)
    )

    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings,
                                             dec_input)

    train_logits, infer_logits = decoding_layer(
        dec_embed_input,
        dec_embeddings,
        enc_state,
        questions_vocab_size,
        sequence_length,
        rnn_size,
        num_layers,
        questions_vocab_to_int,
        keep_prob,
        batch_size
    )

    return train_logits, infer_logits

# Hyperparams
epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75

# Reset graph ensuring its ready for training
ops.reset_default_graph()
# Start session
sess = tf.compat.v1.InteractiveSession()

# Load model inputs
input_data, targets, lr, keep_prob = model_inputs()
# Sequence length will be max line length for each batch
sequence_length = tf.compat.v1.placeholder_with_default(max_line_length, None, name='sequence_length')
# Find shape of the input data for sequence_loss
input_shape = tf.shape(input_data)

# Create training and inference logits
train_logits, inference_logits = seq2seq_model(
    tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(answers_vocab_to_int),
    len(questions_vocab_to_int), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers,
    questions_vocab_to_int
)

# Create a tensor for the inference logits, needed if loading a checkpoint version of the model
tf.identity(inference_logits, 'logits')

with tf.name_scope("optimization"):
    # Loss function
    cost = tf.contrib.seq2seq.sequence_loss(
        train_logits,
        targets,
        tf.ones([input_shape[0], sequence_length])
    )

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)

def pad_sentence_batch(sentence_batch, vocabF_to_int):
    """Pad sentences with <PAD> so that each sentence of a batch has the same length"""
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def batch_data(questions, answers, batch_size):
    """Batch questions and answers together"""
    for batch_i in range(0, len(questions)//batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i: start_i + batch_size]
        pad_questions_batch = np.array(pad_sentence_batch(questions_batch, questions_vocab_to_int))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, answers_vocab_to_int))
        yield pad_questions_batch, pad_answers_batch

# Validate the training with 10% of the data
train_valid_split = int(len(sorted_questions)*0.15)

# Split the questions and answers into training and validating data
train_questions = sorted_questions[train_valid_split:]
train_answers = sorted_questions[train_valid_split:]

valid_questions = sorted_questions[:train_valid_split]
valid_answers = sorted_questions[:train_valid_split]

logging.debug("Training size: {}".format(len(train_questions)))
logging.debug("Validation size: {}".format(len(valid_questions)))

display_step = 100 # Check training loss after every 100 batches
stop_early = 0
stop = 5 # If the validation loss does decrease in 5 consecutive checks, stop training
validation_check = ((len(train_questions))//batch_size//2)-1 # Modulus for checking validation loss
total_train_loss = 0 # Record the training loss for each display step
summary_valid_loss = [] # Record the validation loss for saving improvements in the model

checkpoint = "best_model.ckpt"

sess.run(tf.global_variables_initializer())

for epoch_i in range(1, epochs+1):
    for batch_i, (questions_batch, answers_batch) in enumerate(
            batch_data(train_questions, train_answers, batch_size)):
        start_time = time.time()
        _, loss = sess.run(
            [train_op, cost],
            {input_data: questions_batch,
             targets: answers_batch,
             lr: learning_rate,
             sequence_length: answers_batch.shape[1],
             keep_prob: keep_probability}
        )

        total_train_loss += loss
        end_time = time.time()
        batch_time = end_time - start_time

        if batch_i % display_step == 0:
            logging.info('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                  .format(epoch_i,
                          epochs, 
                          batch_i, 
                          len(train_questions) // batch_size, 
                          total_train_loss / display_step, 
                          batch_time*display_step))
            total_train_loss = 0

        if batch_i % validation_check == 0 and batch_i > 0:
            total_valid_loss = 0
            start_time = time.time()
            for batch_ii, (questions_batch, answers_batch) in enumerate(batch_data(valid_questions, valid_answers, batch_size)):
                valid_loss = sess.run(
                    cost, {input_data: questions_batch,
                           targets: answers_batch,
                           lr: learning_rate,
                           sequence_length: answers_batch.shape[1],
                           keep_prob: 1}
                )
            total_vaid_loss += valid_loss
        end_time = time.time()
        batch_time = end_time - start_time
        avg_valid_loss = total_valid_loss / (len(valid_questions) / batch_size)
        logging.info('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time))

        # Reduce learning rate, but not below its minimum value
        learning_rate *= learning_rate_decay
        if learning_rate < min_learning_rate:
            learning_rate = min_learning_rate
        
        summary_valid_loss.append(avg_valid_loss)
        if avg_valid_loss <= min(summary_valid_loss):
            logging.debug('New Record!')
            stop_early = 0
            saver = tf.train.Saver()
            saver.save(sess, checkpoint)

        else:
            logging.debug('No Improvement.')
            stop_early += 1
            if stop_early == stop:
                break

    if stop_early == stop:
        logging.info("Stopping Training.")
        break

def question_to_seq(question, vocab_to_int):
    '''Prepare the question for the model'''

    question = clean_text(question)
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in question.split()]