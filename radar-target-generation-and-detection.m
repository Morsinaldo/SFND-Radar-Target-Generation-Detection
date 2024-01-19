clear variables global;
close all;
clc;

%% Radar Specifications 
%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Frequency of operation = 77GHz
% Max Range = 200m
% Range Resolution = 1 m
% Max Velocity = 100 m/s
%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Radar parameters
radar_frequency = 77e9;             % Hz (radar carrier frequency)
max_range = 200;                    % m
range_resolution = 1;               % m
max_velocity = 100;                 % m/s

c = 3e8;                            % m/s (speed of light)

%% User-Defined Target Range and Velocity
% Define the target's initial position and velocity.
% Note: Velocity remains constant.

initial_position = 100;             % m, cannot exceed 200m
constant_velocity = 37;             % m/s, range -70 .. 70 m/s

%% FMCW Waveform Generation

% Design the FMCW waveform by specifying the parameters.
% Calculate the Bandwidth (B), Chirp Time (Tchirp), and Slope (slope) 
% of the FMCW chirp using the given requirements.

sweep_to_roundtrip_factor = 5.5;    % sweep-to-roundtrip factor,
                                     % given by project description
                                     % as a typical value for an
                                     % FMCW radar system.

bandwidth = c / (2 * range_resolution);       % Hz, bandwidth
chirp_time = sweep_to_roundtrip_factor * 2 * max_range / c; % s, chirp time
slope = bandwidth / chirp_time;                 % Hz/s

% Operating carrier frequency of the radar 
fc = radar_frequency;
                                                          
% The number of chirps in one sequence. It's ideal to have 2^ value for 
% the ease of running the FFT for Doppler Estimation. 
num_doppler_cells = 128;            % # of Doppler cells OR 
                                    % # of sent periods % number of chirps

% The number of samples on each chirp. 
num_range_cells = 1024;             % for the length of time OR # of range cells

% Timestamp for running the displacement scenario for every sample on each
% chirp
time = linspace(0, num_doppler_cells * chirp_time, num_range_cells * num_doppler_cells);

% Creating the vectors for Tx, Rx, and Mix based on the total samples input.
transmitted_signal = zeros(1, length(time)); 
received_signal = zeros(1, length(time));
beat_signal = zeros(1, length(time));

% Similar vectors for range_covered and time_delay.
range_covered = zeros(1, length(time));
time_delay = zeros(1, length(time));

%% Signal Generation and Moving Target Simulation
% Running the radar scenario over time. 

for idx = 1:length(time)         
    
    % For each time step, update the Range of the Target
    % for constant velocity. 
    range_covered(idx) = initial_position + constant_velocity * time(idx);         
    time_delay(idx)  = 2 * range_covered(idx) / c;   
    
    % For each time step, update the transmitted and
    % received signal. 
    t_tx = time(idx);
    t_rx = time(idx) - time_delay(idx);
    
    tx_phase = 2 * pi * (fc * t_tx + 0.5 * slope * t_tx^2 );   
    rx_phase = 2 * pi * (fc * t_rx + 0.5 * slope * t_rx^2 );   
    
    transmitted_signal(idx) = cos(tx_phase);
    received_signal(idx) = cos(rx_phase);
    
    % Now, by mixing the Transmit and Receive signals, generate the beat signal.
    beat_signal(idx) = transmitted_signal(idx) .* received_signal(idx);
    
end

%% Range Measurement

% Reshape the vector into Nr*Nd array. Nr and Nd here would also define
% the size of Range and Doppler FFT, respectively.
beat_signal_reshaped = reshape(beat_signal, [num_range_cells, num_doppler_cells]);

% Run the FFT on the beat signal along the range bins dimension (Nr)
% and normalize.
range_fft = fft(beat_signal_reshaped, num_range_cells) ./ length(beat_signal_reshaped);

% Take the absolute value of FFT output
range_fft = abs(range_fft);

% Output of FFT is double-sided signal, but we are interested in only
% one side of the spectrum. Hence we throw out half of the samples.
range_fft = range_fft(1:(num_range_cells/2));

% Find the index of the maximum amplitude in the FFT output.
[max_amplitude, index] = max(range_fft);
estimated_distance_to_target = index

% Plotting the range
figure('Name', 'Range from First FFT')

% Plot FFT output 
plot(range_fft); 
axis([0, max_range, 0, 1]);
ylim([0, 0.5])
grid minor;
xlabel('measured range [m]');
ylabel('amplitude');

%% RANGE DOPPLER RESPONSE
% The 2D FFT implementation is already provided here. This will run a 2DFFT
% on the mixed signal (beat signal) output and generate a range doppler
% map. You will implement CFAR on the generated RDM.

% Range Doppler Map Generation.

% The output of the 2D FFT is an image that has a response in the range and
% doppler FFT bins. So, it is important to convert the axis from bin sizes
% to range and doppler based on their Max values.

beat_signal_reshaped = reshape(beat_signal, [num_range_cells, num_doppler_cells]);

% 2D FFT using the FFT size for both dimensions.
range_doppler_map = fft2(beat_signal_reshaped, num_range_cells, num_doppler_cells);

% Taking just one side of the signal from the Range dimension.
range_doppler_map = range_doppler_map(1:num_range_cells/2, 1:num_doppler_cells);
range_doppler_map = fftshift(range_doppler_map);
RDM = abs(range_doppler_map);
RDM = 10*log10(RDM);

% Use the surf function to plot the output of 2DFFT and to show axes
% in both dimensions.
doppler_axis = linspace(-100, 100, num_doppler_cells);
range_axis = linspace(-200, 200, num_range_cells/2) * ((num_range_cells/2)/400);

figure;
surf(doppler_axis, range_axis, RDM);

%% CFAR implementation

% Slide Window through the complete Range Doppler Map

% Select the number of Training Cells in both dimensions.
Tr = 8;
Td = 8;

% Select the number of Guard Cells in both dimensions around the Cell
% under test (CUT) for accurate estimation
Gr = 4;
Gd = 8;

% Offset the threshold by SNR value in dB
offset = 8;

% Create a vector to store noise_level for each iteration on training cells
radius_doppler = Td + Gd;  % no. of doppler cells on either side of CUT
radius_range = Tr + Gr;  % no. of range cells on either side of CUT

Nrange_cells = num_range_cells/2 - 2 * radius_doppler; % no. of range dimension cells
Ndoppler_cells = num_doppler_cells - 2 * radius_range;   % no. of doppler dim. cells

grid_size = (2*Tr + 2*Gr + 1) * (2*Td + 2*Gd + 1);
Nguard_cut_cells = (2*Gr + 1) * (2*Gd + 1);     % no. guards + cell-under-test
Ntrain_cells = grid_size - Nguard_cut_cells;  % no. of training cells

noise_level = zeros(Nrange_cells, Ndoppler_cells);

% Design a loop such that it slides the CUT across the range-doppler map by
% giving margins at the edges for Training and Guard Cells.
%
% For every iteration, sum the signal level within all the training
% cells. To sum, convert the value from logarithmic to linear using db2pow
% function. Average the summed values for all the training
% cells used. After averaging, convert it back to logarithmic using pow2db.
%
% Further, add the offset to it to determine the threshold. Next, compare the
% signal under CUT with this threshold. If the CUT level > threshold, assign
% it a value of 1, else equate it to 0.

% Use RDM[x, y] as the matrix from the output of 2D FFT for 
% implementing CFAR.
cfar_signal = zeros(size(RDM));

r_min = radius_range + 1;
r_max = Nrange_cells - radius_range;

d_min = radius_doppler + 1;
d_max = Ndoppler_cells - radius_doppler;

for r = r_min : r_max
    for d = d_min : d_max
        cell_under_test = RDM(r, d);
        
        cell_count = 0;
        for delta_r = -radius_range : radius_range
            for delta_d = -radius_doppler : radius_doppler
                
                cr = r + delta_r;
                cd = d + delta_d;
                
                in_valid_range = (cr >= 1) && (cd >= 1) && (cr <= Nrange_cells) && (cd <= Ndoppler_cells);
                in_train_cell = abs(delta_r) > Gr || abs(delta_d) > Gd;
                
                if in_valid_range && in_train_cell
                    noise = db2pow(RDM(cr, cd));
                    noise_level(r, d) = noise_level(r, d) + noise;
                    cell_count = cell_count + 1;
                end
               
            end
        end

        % If the signal in the cell under test (CUT) exceeds the
        % threshold, we mark the cell as hot by setting it to 1.
        % We don't need to set it to zero, since the array
        % is already zeroed out.
        threshold = pow2db(noise_level(r, d) / cell_count) + offset;

        if (cell_under_test >= threshold)
            cfar_signal(r, d) = RDM(r, d); % ... or set to 1
        end
        
    end
end

% Display the CFAR output using the Surf function like we did for Range
% Doppler Response output.
figure('Name', 'CA-CFAR Filtered RDM');
ax1 = subplot(1, 2, 1);
surfc(doppler_axis, range_axis, RDM, 'LineStyle', 'none');
alpha 0.75;
zlim([0 50]);
xlabel('velocity [m/s]');
ylabel('range [m]');
zlabel('signal strength [dB]')
title('Range Doppler Response')
colorbar;

ax2 = subplot(1, 2, 2);
surf(doppler_axis, range_axis, cfar_signal, 'LineStyle', 'none');
alpha 0.75;
grid minor;
zlim([0, 50]);
xlabel('velocity [m/s]');
ylabel('range [m]');
zlabel('signal strength [dB]')
title(sprintf('CA-CFAR filtered Range Doppler Response (threshold=%d dB)', offset))
colorbar;

linkprop([ax1, ax2],{'CameraUpVector', 'CameraPosition', 'CameraTarget', 'XLim', 'YLim', 'ZLim'});
