# Homework Notes — Sine Wave Denoising

- Build 4 sine waves.
- Each signal is sampled for 10 seconds at 1000 Hz, so each signal has 10,000 samples.
- Add random noise to the signals, then create a mixed signal by summing the noisy signals.
- The model input is basically: one-hot selector C + context window from the noisy mixed signal.
- The model target is: the aligned clean window from the chosen pure sine wave.
- The context window should have 10 samples.
- Example: choose one of the 4 target sine waves, pick a random location, take 10 noisy mixed samples as input, and the corresponding 10 clean samples from the chosen sine as output.
- We need to implement 3 models: fully connected, RNN, and LSTM.
- Train them with MSE loss.
- The point is to compare the models and explain the results.
