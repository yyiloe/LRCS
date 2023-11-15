from model import Dual_Source_Transformer as DST
import tensorflow as tf

def copy_model(model, source):
    '''Copy model weights to a new model.
    
    Args:
        model: model to be copied.
        x: An input example. This is used to run
            a forward pass in order to add the weights of the graph
            as variables.
    Returns:
        A copy of the model.
    '''
    copied_model = DST()
    
    # If we don't run this step the weights are not "initialized"
    # and the gradients will not be computed.
    
    source_inputs = model.features_inputter(source, training=True)
    model.encoder(source_inputs, [source["inputter_0_length"],source["inputter_1_length"]], training=True)

    copied_model.set_weights(model.get_weights())
    return copied_model

