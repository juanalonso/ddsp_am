# Copyright 2019 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Tests for ddsp.synths."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ddsp import synths
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class AdditiveTest(tf.test.TestCase):

  def test_output_shape_is_correct(self):
    synthesizer = synths.Additive(
        n_samples=64000,
        sample_rate=16000,
        amp_scale_fn=None,
        normalize_below_nyquist=True)
    batch_size = 3
    num_frames = 1000
    amp = tf.zeros((batch_size, num_frames, 1), dtype=tf.float32) + 1.0
    harmonic_distribution = tf.zeros(
        (batch_size, num_frames, 16), dtype=tf.float32) + 1.0 / 16
    f0_hz = tf.zeros((batch_size, num_frames, 1), dtype=tf.float32) + 16000

    output = synthesizer(amp, harmonic_distribution, f0_hz)

    self.assertAllEqual([batch_size, 64000], output.shape.as_list())


if __name__ == '__main__':
  tf.test.main()