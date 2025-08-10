%PLOT_GAUSSIAN_RANDOM_FIELD Demo script to visualize a 2-D Gaussian random field.
%
%   This script calls GENERATE_GAUSSIAN_RANDOM_FIELD to create a random
%   field and displays it using a pseudocolor plot. Modify the parameters
%   below to experiment with different field sizes and roughness values.
%
%   Usage (from MATLAB/Octave command window):
%       >> plot_gaussian_random_field
%
%   ---------------------------------------------------------------------

% ------------------------ User-adjustable parameters ---------------------
fieldSize = 256;   % Side length of the square field
alpha      = 3.0;  % Roughness parameter (larger -> smoother)
normalize  = true; % Whether to rescale field to [0,1]

% ---------------------- Generate the random field ------------------------
field = generate_gaussian_random_field(fieldSize, alpha, normalize);

% --------------------------- Plot the field ------------------------------
figure('Name', 'Gaussian Random Field');
imagesc(field);
axis image off;
colormap jet;
colorbar;

title(sprintf('Gaussian Random Field (size = %d, \alpha = %.2f)', fieldSize, alpha));
