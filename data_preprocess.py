import pandas as pd
import os, glob
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile


class DataPreprocess():
    def __init__(self, data_path, csv_path):

        self.csvfile = pd.read_csv(csv_path)
        self.data_path = data_path

        self.tr = 16000
        self.categories = ['africa', 'australia', 'canada', 'england', 'hongkong', 'us']
        self.colors = ['gray', 'yellow', 'orange', 'navy', 'red', 'violet']
    
    
    def hist_with_number(self, x):
        setx = set(x)
        plt.hist( x
                , bins=6
                , alpha=0.4
                , rwidth=0.9)
    
        for c in setx:
            n_class = len(x[x==c])
            plt.text( c, n_class, n_class,
                    color='slateblue',
                    fontweight='bold',
                    horizontalalignment='center',
                    verticalalignment='bottom')
            
        plt.show()
        
    def hist_time(self, N=-1):
        t_list = [[] for i in range(len(self.categories))]
        for ci, category in enumerate(self.categories):
            print(ci, category)
            file_list = self.get_file_list(category)
            for f in file_list[:N]:
                x, t = self.get_data_from_wav(f)
                t_list[ci].append(t[-1])
    
        plt.hist(t_list, stacked=True, bins=20, color=self.colors)
        plt.legend(self.categories)
        plt.show()
        
    def get_file_list(self, cidx):
        if type(cidx)==int:
            file_list = glob.glob('%s/%s/*wav'%(self.data_path, self.categories[cidx]))
        elif type(cidx)==str:
            file_list = glob.glob('%s/%s/*wav'%(self.data_path, cidx))
        return file_list
    
    def get_data_from_wav(self, wav_file):
        x, sr = librosa.load(wav_file, sr=self.tr)
        t = np.linspace(0, len(x)/sr, len(x))
        return x, t
    
    def draw_audio(self, x, t, title=None, color='blue'):
        plt.figure(figsize=(20,4))
        plt.plot(t, x, marker='.', alpha=0.05, color=color)
        if title: plt.title(title)
        plt.show()
        
    def draw_classcount_hist(self):
        val_hist = []
        for c in self.categories:
            file_list = self.get_file_list(c)
            n_files = len(file_list)
            val_hist.append(n_files)
        print(val_hist)
        plt.hist()
        
    def hist_data_dist(self, data_dict):
    
        for i, (category, color) in enumerate(zip(self.categories, self.colors)):
            d = np.array(data_dict[category], dtype=object)
            plt.hist(d, density=True, histtype='bar', stacked=True, 
                     color=[color for _ in range(len(d))],
                     alpha=1-(i*0.1), bins=50)
        plt.show()
        
    def softmax_quantize(self, x, mu=255):
        assert abs(x.all())<1
        y = np.sign(x)*(np.log(1+(mu*np.abs(x))))/(np.log(1+mu))
        return y 
    
    def data_process(self, N):
        x_data=[]
        y_data=[]
        for ci, category in enumerate(self.categories):
            print(ci, category)
            for idx in range(N):
                file_list = self.get_file_list(category)
                f = file_list[idx]
                x, t = self.get_data_from_wav(f)
                x_quant = self.softmax_quantize(x)
    
                x_data.append(x_quant)
                y_data.append(ci)
    
        return np.array(x_data, dtype=object), np.array(y_data)
