import numpy as np
import soundfile as sf

from ic_model import params as yamnet_params
from ic_model import yamnet as yamnet_model
import tensorflow as tf
import librosa
import io
from six.moves.urllib.request import urlopen


instrument_list = ['Accordion', 'AcousticBass', 'AcousticGuitar', 'Agogo', 'Alto', 'AltoSaxophone', 'Bagpipes', 'Banjo',
                   'Baritone', 'BaritoneSaxophone', 'Bass', 'BassClarinet', 'BassDrum', 'BassTrombone', 'Bassoon',
                   'BongoDrums', 'BrassInstrument', 'Castanets', 'Celesta', 'ChurchBells', 'Clarinet', 'Clavichord',
                   'CongaDrum', 'Contrabass', 'Cowbell', 'CrashCymbals', 'Cymbals', 'Dulcimer', 'ElectricBass',
                   'ElectricGuitar', 'ElectricOrgan', 'EnglishHorn', 'FingerCymbals', 'Flute', 'FretlessBass',
                   'Glockenspiel', 'Gong', 'Guitar', 'Handbells', 'Harmonica', 'Harp', 'Harpsichord', 'HiHatCymbal',
                   'Horn', 'Instrument', 'InstrumentException', 'Kalimba', 'KeyboardInstrument', 'Koto', 'Lute',
                   'Mandolin', 'Maracas', 'Marimba', 'MezzoSoprano', 'Oboe', 'Ocarina', 'OrderedDict', 'Organ',
                   'PanFlute', 'Percussion', 'Piano', 'Piccolo', 'PipeOrgan', 'PitchedPercussion', 'Ratchet',
                   'Recorder', 'ReedOrgan', 'RideCymbals', 'SandpaperBlocks', 'Saxophone', 'Shakuhachi', 'Shamisen',
                   'Shehnai', 'Siren', 'Sitar', 'SizzleCymbal', 'SleighBells', 'SnareDrum', 'Soprano',
                   'SopranoSaxophone', 'SplashCymbals', 'SteelDrum', 'StringInstrument', 'SuspendedCymbal', 'Taiko',
                   'TamTam', 'Tambourine', 'TempleBlock', 'Tenor', 'TenorDrum', 'TenorSaxophone', 'Test',
                   'TestExternal', 'Timbales', 'Timpani', 'TomTom', 'Triangle', 'Trombone', 'Trumpet', 'Tuba',
                   'TubularBells', 'Ukulele', 'UnpitchedPercussion', 'Vibraphone', 'Vibraslap', 'Viola', 'Violin',
                   'Violoncello', 'Vocalist', 'Whip', 'Whistle', 'WindMachine', 'Woodblock', 'WoodwindInstrument',
                   'Xylophone', '_DOC_ORDER', '_MOD', '__builtins__', '__cached__', '__doc__', '__file__', '__loader__',
                   '__name__', '__package__', '__spec__', '_combinations', 'base', 'bundleInstruments', 'common',
                   'copy', 'ensembleNameBySize', 'ensembleNamesBySize', 'environLocal', 'environment', 'fromString',
                   'instrumentFromMidiProgram', 'interval', 'partitionByInstrument', 'pitch', 'sys',
                   'unbundleInstruments', 'unittest', 'Violin, fiddle']


def inst_classifier(url, cutting_start=0, cutting_end=-1):
    # Read in the audio.
    wav_data_raw, sr = sf.read(io.BytesIO(urlopen(url).read()), dtype=np.int16, always_2d=False)
    '''
    wav_data_raw, sr = librosa.load(io.BytesIO(urlopen(url).read()))
    wav_file_name = filename
    wav_data_raw, sr = sf.read(wav_file_name, dtype=np.int16, always_2d=False)'''
    wav_data_raw = wav_data_raw[cutting_start:cutting_end]

    try:
        if wav_data_raw.shape[1] == 1:
            wav_data = wav_data_raw
        elif wav_data_raw.shape[1] == 2:
            wav_data = (wav_data_raw[:, 0] + wav_data_raw[:, 1]) / 2
    except:
        wav_data = wav_data_raw

    # print(max(wav_data))
    waveform = wav_data / 32768.0
    # print(max(waveform))

    # The graph is designed for a sampling rate of 16 kHz, but higher rates should work too.
    # We also generate scores at a 10 Hz frame rate.
    params = yamnet_params.Params(sample_rate=sr, patch_hop_seconds=0.1)

    # Set up the YAMNet model.
    class_names = yamnet_model.class_names('ic_model/yamnet_class_map.csv')
    yamnet = yamnet_model.yamnet_frames_model(params)
    yamnet.load_weights('ic_model/yamnet.h5')

    # Run the model.
    scores, embeddings, spectrogram = yamnet(waveform)
    scores = scores.numpy()

    # Plot and label the model output scores for the top-scoring classes.
    mean_scores = np.mean(scores, axis=0)
    top_N = 10
    top_class_indices = np.argsort(mean_scores)[::-1][:top_N]
    # Compensate for the patch_window_seconds (0.96s) context window to align with spectrogram.
    # Label the top_N classes.

    # plt.savefig('pages/Data/temp.png')
    # png_abs_path = os.path.abspath('pages/Data/temp.png')
    top_ = [class_names[top_class_indices[x]] for x in range(0, 5, 1)]

    return top_


def predict_inst(top):
    instrument = 'None'
    if list(top):
        pass
    for i in top:
        if i in instrument_list:
            instrument = i
            break
    return instrument
