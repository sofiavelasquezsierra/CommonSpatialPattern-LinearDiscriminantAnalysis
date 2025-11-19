# %% imports
from scipy.io import loadmat
import numpy as np
from scipy import signal, stats
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from CSP import csp_fit, csp_transform, plot_csp_feature_extraction
from LDA import lda_fit, lda_predict


# Please refer to requirements.txt for Python environment requirements
# You could easily set up the environment using pip by running the following command:
# python -m pip install -U pip
# pip install -r requirements.txt


# %% load data

session_file = 'AssignmentData01.mat' # load data
task = 'LR' # 1D Left/Right cursor control task
mat = loadmat(session_file, squeeze_me=True, struct_as_record=False)
BCI = mat['BCI']

srate = int(np.array(BCI.srate).item()) # sampling rate

data_cells = list(BCI.data) # segmented eeg data 
TrialData = BCI.TrialData 

# Extract trials with a fix window length
trials = []                     
labels = []
online_preds = []
for i, data in enumerate(data_cells): 
    if str(TrialData[i].taskParadigm) != task:
        continue
    if np.shape(data)[-1] < 5000: # remove trials that are shorter than 3 seconds
        continue
    data = data[:, 2500:5000] # trial length varies; keep the first 0.5 - 3 s of the trial 
    label = int(TrialData[i].targetNum) # 1 is right/up in the LR/UD task, and 2 is left/down
    labels.append(label)
    trials.append(data)

    online_pred = label if TrialData[i].result == 'hit' else 0 # online decoding results
    online_preds.append(online_pred) 

trials = np.asarray(trials) # n_trials, n_chans, n_samples
labels = np.asarray(labels) # n_trials
online_preds = np.asarray(online_preds) # n_trials

# %% preprocess

# common average referencing
# We want the average of n_chans
trials = trials-trials.mean(axis=1, keepdims=True) # along which dimension we want to reference

# downsample
downsrate = 100
n_trials, n_chans, n_samples = np.shape(trials)
newSamples = int(n_samples/srate*downsrate)
trials = resample(trials, newSamples, t=None, axis=-1, window=None, domain='time')
		
# bandpass filtering [4, 40] Hz
padding_length = 100  # Number of zeros to pad; padding before and after to preserve information in the start and end of the signal
padded_sig = np.pad(trials, ((0,0),(0,0),(padding_length,padding_length)), 'constant', constant_values=0)

b, a = signal.butter(4, [4, 40], btype='bandpass', fs=downsrate)
padded_sig = signal.lfilter(b, a, padded_sig, axis=-1)
trials = padded_sig[:,:,padding_length:-padding_length]

# (optional) channel selection: feel free to select only task-relevent channels to see how the performance changes
trials = trials[:, :n_chans, :] # all the channels are included here
# %% CSP
# train/test set split
X_train, X_test, y_train, y_test, online_pred_train, online_pred_test = train_test_split(
    trials, labels, online_preds, 
    test_size=1/5, random_state=13, shuffle=True)

# prepare class splits (on training data only)
X1 = [tr for tr, lab in zip(X_train, y_train) if lab == 1]
X2 = [tr for tr, lab in zip(X_train, y_train) if lab == 2]

csp_pairs = 4 # the number of features we want to keep, CSP features comes in pairs so there will be 2*csp_pairs features
W = csp_fit(X1, X2, reg=1e-6, m_pairs=csp_pairs) # obtain the filters

# get the features
f_train = csp_transform(W, X_train)
f_test = csp_transform(W, X_test)

# visualize the training data features
fig, (X_raw, X_csp) = plot_csp_feature_extraction(
    epochs=X_train,
    y=y_train,
    W=W,               # CSP filters
    log_power=True
)
fig.savefig('fig_train.png', dpi=300) # save and report the figure
fig.show()
# %% LDA
w, b, c1, c2 = lda_fit(f_train, y_train, reg=1e-4) # fit the classifier
y_hat = lda_predict(w, b, c1, c2, f_test) # perform prediction

acc = (y_hat == y_test).mean()
online_acc = (online_pred_test == y_test).mean() 
# here we calculated percent total correct (PTC) instead of percent valid correct (PVC); 
# feel free to play around with different metrics to see how the performance change
print(f"CSP+LDA accuracy: {acc*100:.2f}%; Online accuracy: {online_acc*100:.2f}%") # report the accuracies
# %%
# visualize the test data features
fig, (X_raw, X_csp) = plot_csp_feature_extraction(
    epochs=X_test,
    y=y_test,
    W=W,               # CSP filters
    log_power=True
)
fig.savefig('fig_test.png', dpi=300) # save and report the figure
fig.show()