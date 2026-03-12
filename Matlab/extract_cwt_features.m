%% extract_cwt_features.m
% CWT-based feature extraction for COP classification
%
% This script:
%   1) loads COP data from .mat files,
%   2) applies low-pass filtering,
%   3) extracts continuous wavelet transform (CWT) features from COPx and COPy,
%   4) assigns subject IDs and labels,
%   5) saves the output matrix for classification.
%
% Output columns in "data":
%   1   = Subject
%   2:62   = CWT features of COPx
%   63:123 = CWT features of COPy
%   124 = Label
%
% Output file:
%   data.mat
%
% Author: Negar Rahimi

clear all
clc

%% Settings
subj = 22;
cond = 3;
Trial = 2;

folder_path = 'D:\Articles\Published\Published_2024_Distinguishing Among Standing Postures with Machine Learning-based Classification Algorithms\US\Finalized_DATA\22S_0102';
file_list = dir(fullfile(folder_path, '*.mat'));

data = zeros(501 * subj * cond * Trial, 124);

%% Load all trials
for i = 1:length(file_list)
    file_name = file_list(i).name;
    file_path = fullfile(folder_path, file_name);
    original_data(((i - 1) * 501) + 1:i * 501, :) = cell2mat(struct2cell(load(file_path)));
end

%% Low-pass filter COPx and COPy
fs = 25;
[b, a] = butter(3, 8 / (fs / 2), 'low');

% Optional: inspect filter response
% figure()
% freqz(b, a)

original_data(:, 7) = filter(b, a, original_data(:, 7));
original_data(:, 8) = filter(b, a, original_data(:, 8));

%% Frequency settings
Fs = 25;
L = length(original_data(1:501, 1));
fs = (Fs * (-(L / 2):(L / 2) - 1) / L)';

%% Extract CWT features
% Output format:
% data = [Subj, CWT(COPx), CWT(COPy), Label]

for i = 1:subj * cond * Trial
    data(((i - 1) * 501) + 1:i * 501, 2:62) = (abs(cwt(original_data(((i - 1) * 501) + 1:i * 501, 7))))';
    data(((i - 1) * 501) + 1:i * 501, 63:123) = (abs(cwt(original_data(((i - 1) * 501) + 1:i * 501, 8))))';
end

%% Optional alternative transforms
% Uncomment if you want to explore FFT-based features instead of CWT
%
% for i = 1:subj*cond*Trial
%     data(((i-1)*501)+1:i*501,3) = abs(fftshift(fft(original_data(((i-1)*501)+1:i*501,7))));
%     data(((i-1)*501)+1:i*501,4) = abs(fftshift(fft(original_data(((i-1)*501)+1:i*501,8))));
% end

% Uncomment if you want to explore DWT-based features instead of CWT
%
% [LoD, HiD] = wfilters('bior3.5', 'd');
% Ax = zeros(501, 1);
% Dx = zeros(501, 1);
% Ay = zeros(501, 1);
% Dy = zeros(501, 1);
%
% for i = 1:subj*cond*Trial
%     [Ax(1:256,:), Dx(1:256,:)] = dwt(original_data(((i-1)*501)+1:i*501,7), LoD, HiD);
%     data(((i-1)*501)+1:i*501,2) = Ax;
%     data(((i-1)*501)+1:i*501,3) = Dx;
%
%     [Ay(1:256,:), Dy(1:256,:)] = dwt(original_data(((i-1)*501)+1:i*501,8), LoD, HiD);
%     data(((i-1)*501)+1:i*501,4) = Ay;
%     data(((i-1)*501)+1:i*501,5) = Dy;
% end

%% Organize output matrix
data(:, 1) = original_data(:, 1);     % Subject
data(:, 124) = original_data(:, 4);   % Label

%% Relabel classes
data(data(:, 124) == 5, 124) = 2;
data(data(:, 124) == 6, 124) = 3;

%% Save output
save("data.mat", "data");

%% Optional: display frequency range of CWT scales
numScales = 61;
[~, frequencies] = cwt(original_data(1:501, 8), Fs);
bandwidthFactor = 1 / 6;

disp('Scale\tCenter Frequency (Hz)\tFrequency Range (Hz)');
for i = 1:numScales
    center_freq = frequencies(i);
    lower_freq = center_freq * (1 - bandwidthFactor);
    upper_freq = center_freq * (1 + bandwidthFactor);
    fprintf('%d\t%.2f Hz\t\t\t%.2f - %.2f Hz\n', i, center_freq, lower_freq, upper_freq);
end