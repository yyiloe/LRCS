import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import argparse
import logging
import tensorflow_addons as tfa
import opennmt as onmt
from model import DualSourceTransformer as DST

tf.get_logger().setLevel(logging.INFO)

root_path=os.path.dirname(__file__)

src_path=os.path.join(root_path,'datasets/python/train/src_path.txt')
src_code=os.path.join(root_path,'datasets/python/train/src_code.txt')
tgt_path=os.path.join(root_path,'datasets/python/train/tgt_title.txt')

ast_vocab=os.path.join(root_path,'datasets/vocab/path-vocab.txt')
src_vocab=os.path.join(root_path,'datasets/vocab/src-vocab.txt')
tgt_vocab=os.path.join(root_path,'datasets/vocab/tgt-vocab.txt')
model_dir=os.path.join(root_path,'run')

ttt=0

model = DST()

learning_rate = onmt.schedules.NoamDecay(scale=2.0, model_dim=512, warmup_steps=8000)
optimizer = tfa.optimizers.LazyAdam(learning_rate)
opt_copy = tfa.optimizers.LazyAdam(learning_rate)
checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)

def compute_loss(model, source, target):
    source_inputs = model.features_inputter(source, training=True)
    encoder_outputs, _, _ = model.encoder(
        source_inputs, [source["inputter_0_length"],source["inputter_1_length"]], training=True
    )

    # Run the decoder.
    target_inputs = model.labels_inputter(target, training=True)
    decoder_state = model.decoder.initial_state(
        memory=encoder_outputs, memory_sequence_length=[source["inputter_0_length"],source["inputter_1_length"]]
    )
    logits, _, _ = model.decoder(
        target_inputs, target["length"], state=decoder_state, training=True
    )

    # Compute the cross entropy loss.
    loss_num, loss_den, _ = onmt.utils.cross_entropy_sequence_loss(
        logits,
        target["ids_out"],
        target["length"],
        label_smoothing=0.1,
        average_in_time=True,
        training=True,
    )
    loss = loss_num / loss_den

    return loss

def train(
    src1,
    src2,
    target_file,
    checkpoint_manager,
    maximum_length=100,
    shuffle_buffer_size=-1,  # Uniform shuffle.
    train_steps=100000,
    save_every=2000,
    report_every=1,
):
    """Runs the training loop.
    Args:
      source_file: The source training file.
      target_file: The target training file.
      checkpoint_manager: The checkpoint manager.
      maximum_length: Filter sequences longer than this.
      shuffle_buffer_size: How many examples to load for shuffling.
      train_steps: Train for this many iterations.
      save_every: Save a checkpoint every this many iterations.
      report_every: Report training progress every this many iterations.
    """

    # Create the training dataset.
    dataset = model.examples_inputter.make_training_dataset(
        [src1,src2],
        target_file,
        batch_size=3072,
        batch_type="tokens",
        shuffle_buffer_size=shuffle_buffer_size,
        length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
        maximum_features_length=maximum_length,
        maximum_labels_length=maximum_length,
    )

    @tf.function(input_signature=dataset.element_spec)
    def training_step(source, target):
        loss = compute_loss(model, source, target)

        # Compute and apply the gradients.
        model_copy = model
        variables = model.trainable_variables
        variables_copy = model_copy.trainable_variables
        gradients = optimizer.get_gradients(loss, variables_copy)
        opt_copy.apply_gradients(list(zip(gradients, variables_copy)))
        query_loss = compute_loss(model_copy, source, target) 
        gradients = optimizer.get_gradients(query_loss, variables)

        optimizer.apply_gradients(list(zip(gradients, variables)))
    
        return query_loss

    for source, target in dataset:
        loss = training_step(source, target)
        step = optimizer.iterations.numpy()
        if step % report_every == 0:
            tf.get_logger().info(
                "Step = %d ; Learning rate = %f ; Loss = %f",
                step,
                learning_rate(step),
                loss,
            )
        if step % save_every == 0:
            tf.get_logger().info("Saving checkpoint for step %d", step)
            checkpoint_manager.save(checkpoint_number=step)
        if step == train_steps:
            break



def translate(src1, src2, batch_size=32, beam_size=4):
    """Runs translation.
    Args:
      source_file: The source file.
      batch_size: The batch size to use.
      beam_size: The beam size to use. Set to 1 for greedy search.
    """

    # Create the inference dataset.
    dataset = model.examples_inputter.make_inference_dataset([src1,src2], batch_size)

    @tf.function(input_signature=(dataset.element_spec,))
    def predict(source):
        # Run the encoder.
        source_length = [source["inputter_0_length"],source["inputter_1_length"]]

        batch_size = tf.shape(source_length)[0]
        # print(batch_size.shape)
        source_inputs = model.features_inputter(source)
        encoder_outputs, _, _ = model.encoder(source_inputs, source_length)

        # Prepare the decoding strategy.
        if beam_size > 1:
            encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
            source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
            decoding_strategy = onmt.utils.BeamSearch(beam_size)
        else:
            decoding_strategy = onmt.utils.GreedySearch()
        # Run dynamic decoding.
        decoder_state = model.decoder.initial_state(
            memory=encoder_outputs, memory_sequence_length=source_length
        )
        
        decoded = model.decoder.dynamic_decode(
            model.labels_inputter,
            tf.fill([batch_size], onmt.START_OF_SENTENCE_ID),
            end_id=onmt.END_OF_SENTENCE_ID,
            initial_state=decoder_state,
            decoding_strategy=decoding_strategy,
            maximum_iterations=200,
        )
        # target_lengths = decoded.lengths
        # target_tokens = model.labels_inputter.ids_to_tokens.lookup(
        #     tf.cast(decoded.ids, tf.int64)
        # )
        # return target_tokens, target_lengths
    i=0
    for source in dataset:
        # print(source)
        # predict(source)
        # batch_tokens, batch_length = predict(source)
        if(i==2):
            i=1
            continue

        source_length = [source["inputter_0_length"],source["inputter_1_length"]]

        batch_size = tf.shape(source_length)[0]
        # print(batch_size.shape)
        source_inputs = model.features_inputter(source)
        encoder_outputs, _, _ = model.encoder(source_inputs, source_length)

        # Prepare the decoding strategy.
        if beam_size > 1:
            encoder_outputs = tfa.seq2seq.tile_batch(encoder_outputs, beam_size)
            source_length = tfa.seq2seq.tile_batch(source_length, beam_size)
            decoding_strategy = onmt.utils.BeamSearch(beam_size)
        else:
            decoding_strategy = onmt.utils.GreedySearch()
        
        # Run dynamic decoding.
        decoder_state = model.decoder.initial_state(
            memory=encoder_outputs, memory_sequence_length=source_length
        )
        #decoded = model.decoder.dynamic_decode(
        #    model.labels_inputter,
        #     tf.fill([batch_size], onmt.START_OF_SENTENCE_ID),
        #     end_id=onmt.END_OF_SENTENCE_ID,
        #     initial_state=decoder_state,
        #     decoding_strategy=decoding_strategy,
        #     maximum_iterations=200,
        # )


        break
        for tokens, length in zip(batch_tokens.numpy(), batch_length.numpy()):
            sentence = b" ".join(tokens[0][: length[0]])
            print(sentence.decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # parser.add_argument("run", choices=["train", "translate"], help="Run type.")
    # parser.add_argument("--src", required=True, help="Path to the source file.")
    # parser.add_argument("--tgt", help="Path to the target file.")
    # parser.add_argument(
    #     "--src_vocab", required=True, help="Path to the source vocabulary."
    # )
    # parser.add_argument(
    #     "--tgt_vocab", required=True, help="Path to the target vocabulary."
    # )
    # parser.add_argument(
    #     "--model_dir",
    #     default="checkpoint",
    #     help="Directory where checkpoint are written.",
    # )
    # args = parser.parse_args()

    data_config = {
        "source_1_vocabulary": ast_vocab,
        "source_2_vocabulary": src_vocab,
        "target_vocabulary": tgt_vocab,
    }

    model.initialize(data_config)
    
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, model_dir, max_to_keep=5
    )
    if checkpoint_manager.latest_checkpoint is not None:
        tf.get_logger().info(
            "Restoring parameters from %s", checkpoint_manager.latest_checkpoint
        )
        checkpoint.restore(checkpoint_manager.latest_checkpoint)

    run = "translate"


    if run == "train":
        train(src_path, src_code, tgt_path, checkpoint_manager)
    elif run == "translate":
        translate(src_path, src_code)

        # config = {
        # "model_dir": model_dir,
        #     "data": {
        #     "source_1_vocabulary": ast_vocab,
        #     "source_2_vocabulary": src_vocab,
        #     "target_vocabulary": tgt_vocab
        #     }
        # }
        # runner = onmt.Runner(model, config, auto_config=True)
        # runner.infer([src_path, src_code])


if __name__ == "__main__":
    main()