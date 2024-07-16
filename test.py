import clean
import utils.data_plots

clean.clean_audio('./dataset/audio')
paths = [
    './dataset/audio/AmericanCrow/114552.wav',
    './clean/AmericanCrow/114552.wav'
]
utils.data_plots.plot_signals(paths)