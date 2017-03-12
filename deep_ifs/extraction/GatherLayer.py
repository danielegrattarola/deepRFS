
from keras.engine.topology import Layer
from keras.optimizers import *


class GatherLayer(Layer):
    def __init__(self, output_dim, nb_actions, **kwargs):
        self.output_dim = output_dim
        self.nb_actions = nb_actions
        super(GatherLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(GatherLayer, self).build(input_shape)

    def call(self, args, mask=None):
        return self.gather_layer(args, self.output_dim, self.nb_actions)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim

    @staticmethod
    def gather_layer(args, output_size, nb_actions):
        full_output, indices = args

        # Build the one-hot mask for each sample in the batch.
        # For example given:
        #   full output: [[1, 2, 3, 4, 5, 6], [21, 22, 23, 24, 25, 26]]
        #   nb_actions: 3
        #   output_size: 2
        #   indices: [[2], [0]]
        #   desired output: [[5, 6], [21, 22]]
        # we want the couple of elements [5, 6] representing the output
        # for the third action for the first sample and [21, 22] representing
        # the output for the first actions for the second sample;
        # so we need the absolute indices [[4, 5], [0, 1]].
        # To build these, we compute the first absolute indices (4 and 0) by
        # multiplying the action indices for the output size:
        # [[2], [0]] * 2 = [[4], [0]]
        base_absolute_indices = tf.multiply(indices, output_size)
        # We then build an array containing the first absolute indices repeated
        # output_size times: [[4, 4], [0, 0]]
        bai_repeated = tf.tile(base_absolute_indices, [1, output_size])
        # Finally, we add range(output_size) to these tensors to get the full
        # absolute indices tensors:
        # [4, 4] + [0, 1] = [4, 5]
        # [0, 0] + [0, 1] = [0, 1]
        # so we get: [[4, 5], [0, 1]]
        absolute_indices = tf.add(bai_repeated, tf.range(output_size))
        # We now need to flatten the full tensor in order to later compute the
        # one hot encoding for each absolute index: [4, 5, 0, 1]
        ai_1d = tf.reshape(absolute_indices, [-1])

        # Compute the one-hot encoding for the absolute indices.
        # Continuing the last example, from [4, 5, 0, 1] we now get:
        # [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1],
        #   [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
        ai_onehot = tf.one_hot(ai_1d, output_size * nb_actions)

        # Build the masks for params from the one-hot encoded absolute indices.
        # We now need to group the one-hot absolute indices tensor into
        # output_size-dimensional sub-tensors, in order to reduce-sum along
        # axis 1 and get the correct masks.
        # We therefore get:
        # [
        #   [[0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1]],
        #   [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]
        # ]
        group = tf.reshape(ai_onehot, [-1, output_size, output_size * nb_actions])
        # And with the reduce_sum along axis 1 we get:
        # [[0, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 0]].
        masks = tf.reduce_sum(group, axis=1)

        # Convert mask to boolean.
        # We get: [[False, False, False, False, True, True],
        #           [True, True, False, False, False, False]]
        zero = tf.constant(0, dtype=tf.float32)
        bool_masks = tf.not_equal(masks, zero)

        # Convert the boolean mask to absolute indices for the full_output
        # tensor.
        # We get: [[0, 4], [0, 5], [1, 0], [1, 1]] ([sample index, value index])
        ai_mask = tf.where(bool_masks)

        # Apply the mask to the full output.
        # We get a mono-dimensional tensor: [5, 6, 21, 22]
        reduced_output = tf.gather_nd(full_output, ai_mask)

        # Reshape the reduction to match the output shape.
        # We get: [[5, 6], [21, 22]]
        return tf.reshape(reduced_output, [-1, output_size])
