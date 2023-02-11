from scipy.signal import butter, lfilter
from scipy.signal import freqz


def print_scores(label, scores):
    print(f"{label} Score Time: %.2f" % (scores['score_time'].mean() * 100))
    print(f"{label} Test Accuracy: %.2f%%" % (scores['test_accuracy'].mean() * 100))
    print(f"{label} Train Accuracy: %.2f%%" % (scores['train_accuracy'].mean() * 100))
    print(f"{label} Test Precision Weighted Accuracy: %.2f%%" % (scores['test_precision_weighted'].mean() * 100))
    print(f"{label} Train Precision Weighted Accuracy: %.2f%%" % (scores['train_precision_weighted'].mean() * 100))
    print(f"{label} Test Recall Weighted Accuracy: %.2f%%" % (scores['test_recall_weighted'].mean() * 100))
    print(f"{label} Train Recall Weighted Accuracy: %.2f%%" % (scores['train_recall_weighted'].mean() * 100))
    print(f"{label} Test F1 Accuracy: %.2f%%" % (scores['test_f1_weighted'].mean() * 100))
    print(f"{label} Train F1 Accuracy: %.2f%%\n" % (scores['train_f1_weighted'].mean() * 100))


def butter_bandpass(low_cut: object, high_cut: object, fs: object, order: object = 5) -> object:
    return butter(order, [low_cut, high_cut], fs=fs, btype='band')


def butter_bandpass_filter(data, low_cut, high_cut, fs, order=5):
    b, a = butter_bandpass(low_cut, high_cut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def readTemperatures(csv_file):
    with open(csv_file) as file:
        file.readline()
        csv_file = csv.reader(file)
        time = []
        temperature = []
        for row in csv_file:
            e = row[0].split()
            time.append(e[0])
            temperature.append(e[1])

        return time, temperature


def readFilteredBioSignals(csv_file):
    with open(csv_file) as file:
        file.readline()
        csv_file = csv.reader(file)
        time = []
        gsr = []
        ecg = []
        emg_trapezius = []
        emg_corrugator = []
        emg_zygomaticus = []
        for row in csv_file:
            e = row[0].split()
            time.append(e[0])
            gsr.append(e[1])
            ecg.append(e[2])
            emg_trapezius.append(e[3])
            emg_corrugator.append(e[4])
            emg_zygomaticus.append(e[5])

        return time, gsr, ecg, emg_trapezius, emg_corrugator, emg_zygomaticus


def plotTimeTemperature(path):
    df = pd.read_csv(path, delimiter="\t")
    df = pd.DataFrame(df)
    time_data = df[df.columns[0]]
    temperature_data = df[df.columns[1]]
    plt.plot(time_data, temperature_data)
    plt.xlabel("Time")
    plt.ylabel("Temperature")
    plt.title("Thermode temperature over time diagram")
    plt.show()


def plotBiosignals(path):
    df = pd.read_csv(path, delimiter="\t")
    df = pd.DataFrame(df)
    time = df[df.columns[0]]
    gsr = df[df.columns[1]]
    ecg = df[df.columns[2]]
    emg_trapezius = df[df.columns[3]]
    emg_corrugator = df[df.columns[4]]
    emg_zygomaticus = df[df.columns[5]]

    plt.plot(time, gsr, label="GSR")
    plt.plot(time, ecg, label="ECG")
    plt.plot(time, emg_trapezius, label="EMG Trapezius")
    plt.plot(time, emg_corrugator, label="EMG Corrugator")
    plt.plot(time, emg_zygomaticus, label="EMG Zygomaticus")

    plt.xlabel("Time")
    plt.title("Biophysical signals over time diagram")
    plt.legend()

    plt.show()

    return time, gsr, ecg, emg_trapezius, emg_corrugator, emg_zygomaticus
