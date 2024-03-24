import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf


class Noise:
    """
    Noise Class generates white, brown or pink noise with specified parameters

    Attributes:
        amplitude (int): Maximum amplitude for 16-bit PCM audio.
        colour (str): The colour of the noise to generate ('White', 'Brown', 'Pink').
        sample_rate (int): Sampling rate in HZ.
        duration (int): Length of the noise in seconds.
        num_samples (int): The total number of noise samples generated.
        noise_sample (np.ndarray): The generated noise/frequency array
    """

    amplitude = 32767  # max amplitude for 16-bit PCM

    def __init__(self, colour, sample_rate, duration):
        self.colour = colour
        self.sample_rate = sample_rate
        self.duration = duration
        self.num_samples = self.sample_rate * self.duration
        self.noise_sample = None

        if self.colour == 'White':
            self.noise_sample = self.generate_white_noise
        elif self.colour == 'Brown':
            self.noise_sample = self.generate_brown_noise
        elif self.colour == 'Pink':
            self.noise_sample = self.generate_pink_noise
        else:
            raise ValueError('Unsupported noise colour')

    @property
    def generate_white_noise(self) -> np.ndarray:
        return np.random.uniform(low=-self.amplitude, high=self.amplitude, size=self.num_samples)

    @property
    def generate_brown_noise(self) -> np.ndarray:
        white_noise = self.generate_white_noise

        # calculate cumulative sum of rach elements - it leads to the fact that lower tones are more powerful
        brown_noise = np.cumsum(white_noise)

        # normalise brown noise to the range (-1, 1) using linear interpolation
        brown_noise = np.interp(brown_noise, (brown_noise.min(), brown_noise.max()), (-1, 1))

        return brown_noise

    @property
    def generate_pink_noise(self) -> np.ndarray:
        white_noise = self.generate_white_noise

        # Estimation of 1/f filter:
        b = np.array([0.04992203, 0.0506127, 0.0506127, 0.04992203])
        a = np.array([1, -2.494956002, 2.017265875, -0.522189400])

        pink_noise = np.convolve(white_noise, b / a[0], 'same')
        pink_noise = np.interp(pink_noise, (pink_noise.min(), pink_noise.max()), (-1, 1))

        return pink_noise

    def draw_noise(self) -> None:
        plt.plot(self.noise_sample)
        plt.show()

    def save_sound(self) -> None:
        normalised_noise = self.sound_normalisation(self.noise_sample)
        filename = f'{self.colour} noise.wav'
        sf.write(filename, normalised_noise, self.sample_rate)

    def sound_normalisation(self, noise) -> np.ndarray:
        desired_amplitude = 0.1
        max_amp = np.max(np.abs(noise))
        normalised_noise = noise / max_amp * desired_amplitude
        return normalised_noise

    def __str__(self):
        return f'---------------\n' \
               f'Sample rate: {self.sample_rate},\n' \
               f'Duration: {self.duration},\n' \
               f'Amplitude: {self.amplitude}\n' \
               f'---------------'


if __name__ == '__main__':
    white_noise = Noise('White', 48000, 5)
    brown_noise = Noise('Brown', 48000, 5)
    pink_noise = Noise('Pink', 48000, 5)

    white_noise.save_sound()
    brown_noise.save_sound()
    pink_noise.save_sound()

    white_noise.draw_noise()
    brown_noise.draw_noise()
    pink_noise.draw_noise()