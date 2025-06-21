function filtered_seismogram = applyLowPassFilter(seismogram, Fs, Fc, n)
    % applyLowPassFilter - Designs and applies a low-pass Butterworth filter.
    %
    % Syntax:  filtered_seismogram = applyLowPassFilter(seismogram, Fs, Fc, n)
    %
    % Inputs:
    %    seismogram - Array containing the seismogram trace data
    %    Fs - Sampling rate in Hz
    %    Fc - Cutoff frequency of the low-pass filter in Hz
    %    n - Order of the Butterworth filter
    %
    % Outputs:
    %    filtered_seismogram - The filtered seismogram trace
    
    % Normalize the frequency with respect to the Nyquist frequency
    Wn = Fc / (Fs / 2);
    
    % Design a low-pass Butterworth filter
    [b, a] = butter(n, Wn, 'low');
    
    % Apply the filter to the seismogram trace with zero-phase distortion
    filtered_seismogram = filtfilt(b, a, seismogram);
end
