# Copyright 2021 The DDSP Authors.
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
"""Library of synthesizer functions."""

from ddsp import core
from ddsp import processors
import gin
import tensorflow.compat.v2 as tf


@gin.register
class TensorToAudio(processors.Processor):
  """Identity "synth" returning input samples with channel dimension removed."""

  def __init__(self, name='tensor_to_audio'):
    super().__init__(name=name)

  def get_controls(self, samples):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      samples: 3-D Tensor of "controls" (really just samples), of shape
        [batch, time, 1].

    Returns:
      Dictionary of tensors of synthesizer controls.
    """
    return {'samples': samples}

  def get_signal(self, samples):
    """"Synthesize" audio by removing channel dimension from input samples.

    Args:
      samples: 3-D Tensor of "controls" (really just samples), of shape
        [batch, time, 1].

    Returns:
      A tensor of audio with shape [batch, time].
    """
    return tf.squeeze(samples, 2)


@gin.register
class Harmonic(processors.Processor):
  """Synthesize audio with a bank of harmonic sinusoidal oscillators."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               scale_fn=core.exp_sigmoid,
               normalize_below_nyquist=True,
               amp_resample_method='window',
               name='harmonic'):
    """Constructor.

    Args:
      n_samples: Fixed length of output audio.
      sample_rate: Samples per a second.
      scale_fn: Scale function for amplitude and harmonic distribution inputs.
      normalize_below_nyquist: Remove harmonics above the nyquist frequency
        and normalize the remaining harmonic distribution to sum to 1.0.
      amp_resample_method: Mode with which to resample amplitude envelopes.
        Must be in ['nearest', 'linear', 'cubic', 'window']. 'window' uses
        overlapping windows (only for upsampling) which is smoother
        for amplitude envelopes with large frame sizes.
      name: Synth name.
    """
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.scale_fn = scale_fn
    self.normalize_below_nyquist = normalize_below_nyquist
    self.amp_resample_method = amp_resample_method

  def get_controls(self,
                   amplitudes,
                   harmonic_distribution,
                   f0_hz):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      amplitudes: 3-D Tensor of synthesizer controls, of shape
        [batch, time, 1].
      harmonic_distribution: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_harmonics].
      f0_hz: Fundamental frequencies in hertz. Shape [batch, time, 1].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the amplitudes.
    if self.scale_fn is not None:
      amplitudes = self.scale_fn(amplitudes)
      harmonic_distribution = self.scale_fn(harmonic_distribution)

    # Bandlimit the harmonic distribution.
    if self.normalize_below_nyquist:
      n_harmonics = int(harmonic_distribution.shape[-1])
      harmonic_frequencies = core.get_harmonic_frequencies(f0_hz,
                                                           n_harmonics)
      harmonic_distribution = core.remove_above_nyquist(harmonic_frequencies,
                                                        harmonic_distribution,
                                                        self.sample_rate)

    # Normalize
    harmonic_distribution /= tf.reduce_sum(harmonic_distribution,
                                           axis=-1,
                                           keepdims=True)

    return {'amplitudes': amplitudes,
            'harmonic_distribution': harmonic_distribution,
            'f0_hz': f0_hz}

  def get_signal(self, amplitudes, harmonic_distribution, f0_hz):
    """Synthesize audio with harmonic synthesizer from controls.

    Args:
      amplitudes: Amplitude tensor of shape [batch, n_frames, 1]. Expects
        float32 that is strictly positive.
      harmonic_distribution: Tensor of shape [batch, n_frames, n_harmonics].
        Expects float32 that is strictly positive and normalized in the last
        dimension.
      f0_hz: The fundamental frequency in Hertz. Tensor of shape [batch,
        n_frames, 1].

    Returns:
      signal: A tensor of harmonic waves of shape [batch, n_samples].
    """
    signal = core.harmonic_synthesis(
        frequencies=f0_hz,
        amplitudes=amplitudes,
        harmonic_distribution=harmonic_distribution,
        n_samples=self.n_samples,
        sample_rate=self.sample_rate,
        amp_resample_method=self.amp_resample_method)
    return signal


@gin.register
class FilteredNoise(processors.Processor):
  """Synthesize audio by filtering white noise."""

  def __init__(self,
               n_samples=64000,
               window_size=257,
               scale_fn=core.exp_sigmoid,
               initial_bias=-5.0,
               name='filtered_noise'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.window_size = window_size
    self.scale_fn = scale_fn
    self.initial_bias = initial_bias

  def get_controls(self, magnitudes):
    """Convert network outputs into a dictionary of synthesizer controls.

    Args:
      magnitudes: 3-D Tensor of synthesizer parameters, of shape [batch, time,
        n_filter_banks].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the magnitudes.
    if self.scale_fn is not None:
      magnitudes = self.scale_fn(magnitudes + self.initial_bias)

    return {'magnitudes': magnitudes}

  def get_signal(self, magnitudes):
    """Synthesize audio with filtered white noise.

    Args:
      magnitudes: Magnitudes tensor of shape [batch, n_frames, n_filter_banks].
        Expects float32 that is strictly positive.

    Returns:
      signal: A tensor of harmonic waves of shape [batch, n_samples, 1].
    """
    batch_size = int(magnitudes.shape[0])
    signal = tf.random.uniform(
        [batch_size, self.n_samples], minval=-1.0, maxval=1.0)
    return core.frequency_filter(signal,
                                 magnitudes,
                                 window_size=self.window_size)


@gin.register
class Wavetable(processors.Processor):
  """Synthesize audio from a series of wavetables."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               scale_fn=core.exp_sigmoid,
               name='wavetable'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.scale_fn = scale_fn

  def get_controls(self,
                   amplitudes,
                   wavetables,
                   f0_hz):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      amplitudes: 3-D Tensor of synthesizer controls, of shape
        [batch, time, 1].
      wavetables: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_harmonics].
      f0_hz: Fundamental frequencies in hertz. Shape [batch, time, 1].

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the amplitudes.
    if self.scale_fn is not None:
      amplitudes = self.scale_fn(amplitudes)
      wavetables = self.scale_fn(wavetables)

    return  {'amplitudes': amplitudes,
             'wavetables': wavetables,
             'f0_hz': f0_hz}

  def get_signal(self, amplitudes, wavetables, f0_hz):
    """Synthesize audio with wavetable synthesizer from controls.

    Args:
      amplitudes: Amplitude tensor of shape [batch, n_frames, 1]. Expects
        float32 that is strictly positive.
      wavetables: Tensor of shape [batch, n_frames, n_wavetable].
      f0_hz: The fundamental frequency in Hertz. Tensor of shape [batch,
        n_frames, 1].

    Returns:
      signal: A tensor of of shape [batch, n_samples].
    """
    wavetables = core.resample(wavetables, self.n_samples)
    signal = core.wavetable_synthesis(amplitudes=amplitudes,
                                      wavetables=wavetables,
                                      frequencies=f0_hz,
                                      n_samples=self.n_samples,
                                      sample_rate=self.sample_rate)
    return signal


@gin.register
class Sinusoidal(processors.Processor):
  """Synthesize audio with a bank of arbitrary sinusoidal oscillators."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               amp_scale_fn=core.exp_sigmoid,
               amp_resample_method='window',
               freq_scale_fn=core.frequencies_sigmoid,
               name='sinusoidal'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.amp_scale_fn = amp_scale_fn
    self.amp_resample_method = amp_resample_method
    self.freq_scale_fn = freq_scale_fn

  def get_controls(self, amplitudes, frequencies):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      amplitudes: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_sinusoids].
      frequencies: 3-D Tensor of synthesizer controls, of shape
        [batch, time, n_sinusoids]. Expects strictly positive in Hertz.

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the inputs.
    if self.amp_scale_fn is not None:
      amplitudes = self.amp_scale_fn(amplitudes)

    if self.freq_scale_fn is not None:
      frequencies = self.freq_scale_fn(frequencies)
      amplitudes = core.remove_above_nyquist(frequencies,
                                             amplitudes,
                                             self.sample_rate)

    return {'amplitudes': amplitudes,
            'frequencies': frequencies}

  def get_signal(self, amplitudes, frequencies):
    """Synthesize audio with sinusoidal synthesizer from controls.

    Args:
      amplitudes: Amplitude tensor of shape [batch, n_frames, n_sinusoids].
        Expects float32 that is strictly positive.
      frequencies: Tensor of shape [batch, n_frames, n_sinusoids].
        Expects float32 in Hertz that is strictly positive.

    Returns:
      signal: A tensor of harmonic waves of shape [batch, n_samples].
    """
    # Create sample-wise envelopes.
    amplitude_envelopes = core.resample(amplitudes, self.n_samples,
                                        method=self.amp_resample_method)
    frequency_envelopes = core.resample(frequencies, self.n_samples)

    signal = core.oscillator_bank(frequency_envelopes=frequency_envelopes,
                                  amplitude_envelopes=amplitude_envelopes,
                                  sample_rate=self.sample_rate)
    return signal


@gin.register
class AmplitudeModulation(processors.Processor):
  """Synthesize audio with amplitude modulation."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               amp_scale_fn=core.exp_sigmoid,
               amp_resample_method='window',
               freq_scale_fn=core.frequencies_sigmoid,
               name='ampmod'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.amp_scale_fn = amp_scale_fn
    self.amp_resample_method = amp_resample_method
    self.freq_scale_fn = freq_scale_fn

  def get_controls(self,
                   amps,
                   f0_hz,
                   mod_amps,
                   mod_f0_hz):
    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      amps: 3-D Tensor of synthesizer controls, of shape
        [batch, time, 1].
      f0_hz: Fundamental frequencies in hertz. Shape [batch, time, 1].
      mod_amps: 3-D Tensor of synthesizer controls, of shape
        [batch, time, 1].
      mod_f0_hz: 3-D Tensor of synthesizer controls, of shape
        [batch, time, 1]. Expects strictly positive in Hertz.

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """
    # Scale the inputs.
    if self.amp_scale_fn is not None:
      amps = self.amp_scale_fn(amps)
      mod_amps = self.amp_scale_fn(mod_amps)

    if self.freq_scale_fn is not None:
      mod_f0_hz = self.freq_scale_fn(mod_f0_hz, hz_max=self.sample_rate/2.0)
      # mod_amps = core.remove_above_nyquist(mod_f0_hz,
      #                                        mod_amps,
      #                                        self.sample_rate)

    return {'amps': amps,
            'f0_hz': f0_hz,
            'mod_amps': mod_amps,
            'mod_f0_hz': mod_f0_hz}

  def get_signal(self,  amps, f0_hz, mod_amps, mod_f0_hz):
    """Synthesize audio with am synthesizer from controls.

    Args:
      amps: Amplitude tensor of shape [batch, n_frames, 1]. Expects
        float32 that is strictly positive.
      f0_hz: The fundamental frequency in Hertz. Tensor of shape [batch,
        n_frames, 1].
      mod_amps: Amplitude tensor of shape [batch, n_frames, 1].
        Expects float32 that is strictly positive.
      mod_f0_hz: Tensor of shape [batch, n_frames, 1].
        Expects float32 in Hertz that is strictly positive.

    Returns:
      signal: A tensor of shape [batch, n_samples].
    """
    # Create sample-wise envelopes.
    amps_envelopes = core.resample(amps, self.n_samples,
                                        method=self.amp_resample_method)
    f0_hz_envelopes = core.resample(f0_hz, self.n_samples)
    mod_amps_envelopes = core.resample(mod_amps, self.n_samples,
                                        method=self.amp_resample_method)
    mod_f0_hz_envelopes = core.resample(mod_f0_hz, self.n_samples)

    signal = core.modulate_amplitude(amps=amps_envelopes,
                                     f0_hz=f0_hz_envelopes,
                                     mod_amps=mod_amps_envelopes,
                                     mod_f0_hz=mod_f0_hz_envelopes,
                                     sample_rate=self.sample_rate)
    return signal


@gin.register
class FrequencyModulation(processors.Processor):
  """Synthesize audio with frequency modulation."""

  def __init__(self,
               n_samples=64000,
               sample_rate=16000,
               amp_scale_fn=core.exp_sigmoid,
               amp_resample_method='window',
               name='freqmod'):
    super().__init__(name=name)
    self.n_samples = n_samples
    self.sample_rate = sample_rate
    self.amp_scale_fn = amp_scale_fn
    self.amp_resample_method = amp_resample_method

  def get_controls(self, f0,
                   op1, op2, op3, op4,
                   modulators
                   ):

    """Convert network output tensors into a dictionary of synthesizer controls.

    Args:
      f0_hz: Fundamental frequencies in hertz. Shape [batch, time , 1]
      op1-4: Amp, idx and ADSR of each operator. Shape [batch, time , 10]
      modulators: Modulation between operators. Shape [batch, time , 6]

    Returns:
      controls: Dictionary of tensors of synthesizer controls.
    """

    return {'f0': f0,
            'op1': op1, 'op2': op2, 'op3': op3, 'op4': op4,
            'modulators': modulators,
           }

  def get_signal(self, f0,
                 op1, op2, op3, op4,
                 modulators
                 ):
    """Synthesize audio with fm synthesizer from controls.

    Args:
      f0_hz: Fundamental frequencies in hertz. Shape [batch, n_frames , 1]
      op1-4: Amp, idx and ADSR of each operator. Shape [batch, n_frames , 10]
      modulators: Modulation between operators. Shape [batch, n_frames , 6]

    Returns:
      signal: A tensor of shape [batch, n_samples].
    """
    # Create sample-wise envelopes.
    f0_env = core.resample(f0, self.n_samples)

    op1_env, op1_adsr = tf.split(op1, [2,-1], axis=2)
    op1_env = core.resample(op1_env, self.n_samples)
    op1_adsr = core.resampleADSR(op1_adsr, self.n_samples)

    op2_env, op2_adsr = tf.split(op2, [2,-1], axis=2)
    op2_env = core.resample(op2_env, self.n_samples)
    op2_adsr = core.resampleADSR(op2_adsr, self.n_samples)

    op3_env, op3_adsr = tf.split(op3, [2,-1], axis=2)
    op3_env = core.resample(op3_env, self.n_samples)
    op3_adsr = core.resampleADSR(op3_adsr, self.n_samples)

    op4_env, op4_adsr = tf.split(op4, [2,-1], axis=2)
    op4_env = core.resample(op4_env, self.n_samples)
    op4_adsr = core.resampleADSR(op4_adsr, self.n_samples)

    modulators_env = core.resample(modulators, self.n_samples)

    signal = core.modulate_frequency(f0=f0_env,
                                     op1=op1_env, op2=op2_env, op3=op3_env, op4=op4_env,
                                     op1_adsr=op1_adsr, op2_adsr=op2_adsr, op3_adsr=op3_adsr, op4_adsr=op4_adsr,
                                     modulators = modulators_env,
                                     sample_rate=self.sample_rate)
    return signal
