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

py_path=os.path.join(root_path,'datasets/python/train/src_path.txt')
py_code=os.path.join(root_path,'datasets/python/train/src_code.txt')
py_tgt=os.path.join(root_path,'datasets/python/train/tgt_title.txt')

java_path=os.path.join(root_path,'datasets/java/train/src_path.txt')
java_code=os.path.join(root_path,'datasets/java/train/src_code.txt')
java_tgt=os.path.join(root_path,'datasets/java/train/tgt_title.txt')

paths = [py_path, java_path]
codes = [py_code, java_code]
tgts = [py_tgt, java_tgt]

ast_vocab=os.path.join(root_path,'datasets/vocab/path-vocab.txt')
src_vocab=os.path.join(root_path,'datasets/vocab/src-vocab.txt')
tgt_vocab=os.path.join(root_path,'datasets/vocab/tgt-vocab.txt')
model_dir=os.path.join(root_path,'run_new')


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
    paths,
    codes,
    tgts,
    checkpoint_manager,
    maximum_length=100,
    shuffle_buffer_size=-1,  # Uniform shuffle.
    train_steps=100000,
    save_every=2000, #2000
    report_every=1,
):
    # Create the training dataset.
    model_init = model
    for path, code, tgt in zip(paths, codes, tgts):

        dataset = model.examples_inputter.make_training_dataset(
            [path,code],
            tgt,
            batch_size=3072,
            batch_type="tokens",
            shuffle_buffer_size=shuffle_buffer_size,
            length_bucket_width=1,  # Bucketize sequences by the same length for efficiency.
            maximum_features_length=maximum_length,
            maximum_labels_length=maximum_length,
        )

        @tf.function(input_signature=dataset.element_spec)
        def training_step(source, target):
            loss=compute_loss(model, source, target)

            # Compute and apply the gradients.
            model_copy = model_init
            variables = model.trainable_variables
            variables_copy= model_copy.trainable_variables
            gradients = optimizer.get_gradients(loss, variables_copy)
            opt_copy.apply_gradients(list(zip(gradients, variables_copy)))
            query_loss=compute_loss(model_copy, source, target)
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
            # print(step,train_steps)
            if (step == train_steps or step == 2*train_steps):
                break


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )


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

    run = "train"


    if run == "train":
        train(paths, codes, tgts, checkpoint_manager)

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