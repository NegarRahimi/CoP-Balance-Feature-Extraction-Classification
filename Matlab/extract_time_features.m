%% extract_time_features.m
% Time-domain feature extraction for baseline-condition COP classification
%
% This script:
%   1) loads COP data from .mat files,
%   2) applies low-pass filtering,
%   3) extracts time-domain features,
%   4) relabels classes for 4-class classification,
%   5) saves the output matrix for classification.
%
% Output columns in "data":
%   1 = Subject
%   2 = Time
%   3 = COPx
%   4 = COPy
%   5 = Displacement
%   6 = Area
%   7 = Angle
%   8 = Label
%
% Output file:
%   data_Time_Condition_Baseline.mat
%
% Author: Negar Rahimi

clear all
clc

%% Settings
subj = 22;
cond = 4;
Trial = 2;

folder_path = 'D:\Articles\2024_Calasification of Tens using Balance\US\Finalized_DATA\BaselineConditions\BaselineConditions';
file_list = dir(fullfile(folder_path, '*.mat'));

data = zeros(501 * subj * cond * Trial, 8);

%% Load all trials
for i = 1:length(file_list)
    file_name = file_list(i).name;
    file_path = fullfile(folder_path, file_name);
    original_data(((i - 1) * 501) + 1 : i * 501, :) = cell2mat(struct2cell(load(file_path)));
end

%% Low-pass filter COPx and COPy
fs = 25;
[b, a] = butter(3, 8 / (fs / 2), 'low');

% Optional: inspect filter response
% figure()
% freqz(b, a)

original_data(:, 7) = filter(b, a, original_data(:, 7));
original_data(:, 8) = filter(b, a, original_data(:, 8));

%% Extract time-domain features
% Output format:
% data = [Subj, Time, COPx, COPy, Distance, Area, Angle, Label]

data(1, 5) = 0;

% Displacement between consecutive samples
j = 1;
for i = 1:length(original_data) - 1
    if mod(j, 501) == 0
        data(i + 1, 5) = 0;
    else
        data(i + 1, 5) = sqrt((original_data(i + 1, 7) - original_data(i, 7))^2 + ...
                              (original_data(i + 1, 8) - original_data(i, 8))^2);
    end
    j = j + 1;
end

% Triangle area formed by consecutive COP points
k = 1;
for i = 1:length(original_data) - 2
    if mod(k, 501) == 0
        data(i + 1:i + 2, 6) = 0;
    else
        data(i + 2, 6) = 0.5 * abs((original_data(i + 1, 7) * original_data(i + 2, 8)) - ...
                                    (original_data(i + 2, 7) * original_data(i + 1, 8)));
    end
    k = k + 1;
end

% Angle between consecutive COP vectors
k = 1;
for i = 1:length(original_data) - 2
    if mod(k, 501) == 0
        data(i + 1:i + 2, 7) = 0;
    else
        data(i + 2, 7) = acos((original_data(i + 1, 7) * (original_data(i + 2, 7) - original_data(i + 1, 7)) + ...
                                original_data(i + 1, 8) * (original_data(i + 2, 8) - original_data(i + 1, 8))) / ...
                               ((sqrt(original_data(i + 1, 7)^2 + original_data(i + 1, 8)^2)) * ...
                                (sqrt((original_data(i + 2, 7) - original_data(i + 1, 7))^2 + ...
                                      (original_data(i + 2, 8) - original_data(i + 1, 8))^2))));
    end
    k = k + 1;
end

%% Organize output matrix
data(:, 1) = original_data(:, 1);   % Subject
data(:, 2) = original_data(:, 6);   % Time
data(:, 3) = original_data(:, 7);   % COPx
data(:, 4) = original_data(:, 8);   % COPy
data(:, 8) = original_data(:, 5);   % Label

%% Relabel classes for 4-class classification
% Original labels:
%   2 -> 1
%   3 -> 2
%   4 -> 2
%   5 -> 3
%   6 -> 3
%   7 -> 4
%   8 -> 4

data(data(:, 8) == 2, 8) = 1;
data(data(:, 8) == 3, 8) = 2;
data(data(:, 8) == 4, 8) = 2;
data(data(:, 8) == 5, 8) = 3;
data(data(:, 8) == 6, 8) = 3;
data(data(:, 8) == 7, 8) = 4;
data(data(:, 8) == 8, 8) = 4;

%% Save output
save("data_Time_Condition_Baseline.mat", "data");

%% Optional: inspect distribution of a selected feature
% figure()
% h = histogram(data(:, 3), 30);
% hold on;
%
% pd = fitdist(data(:, 3), 'Normal');
% x_values = linspace(min(data(:, 3)), max(data(:, 3)), 100);
% pdf_values = pdf(pd, x_values);
% scale = h.BinWidth * sum(h.Values);
%
% plot(x_values, pdf_values * scale, 'r-', 'LineWidth', 2);
% xlabel('Feature value');
% ylabel('Frequency');
% title('Histogram with fitted normal distribution');
% legend('Histogram', 'Normal PDF');
% hold off;