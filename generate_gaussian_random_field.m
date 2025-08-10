function field = generate_gaussian_random_field(sz, alpha, normalize)
% GENERATE_GAUSSIAN_RANDOM_FIELD Generate a 2-D Gaussian random field using a power-law spectrum.
%
%   field = GENERATE_GAUSSIAN_RANDOM_FIELD(sz) generates a sz-by-sz Gaussian
%   random field with default parameters (alpha = 3, normalize = true).
%
%   field = GENERATE_GAUSSIAN_RANDOM_FIELD(sz, alpha) allows you to specify
%   the roughness parameter ALPHA. Larger ALPHA produces smoother fields.
%
%   field = GENERATE_GAUSSIAN_RANDOM_FIELD(sz, alpha, normalize) additionally
%   controls whether the output FIELD is linearly rescaled to the interval
%   [0, 1].
%
%   Inputs
%   ------
%   sz        : Scalar specifying the size of the square field (sz × sz).
%   alpha     : (Optional) Roughness parameter |alpha| > 0 (default: 3.0).
%   normalize : (Optional) Logical flag indicating whether to scale the
%               output field to [0, 1] (default: true).
%
%   Output
%   -------
%   field : sz-by-sz double matrix representing the Gaussian random field.
%
%   The implementation mirrors the reference Python code that uses an
%   inverse FFT of a complex Gaussian field weighted by a power-law power
%   spectrum.
%
%   Example
%   -------
%   f = generate_gaussian_random_field(256, 2.5);
%   imagesc(f), axis image off, colormap jet, colorbar;
%
%   Author: ChatGPT
%   ---------------------------------------------------------------------

    % Set default arguments ------------------------------------------------
    if nargin < 2 || isempty(alpha)
        alpha = 3.0;         % Default roughness parameter
    end
    if nargin < 3 || isempty(normalize)
        normalize = true;    % Default: rescale output to [0, 1]
    end

    % Validate inputs ------------------------------------------------------
    validateattributes(sz, {'numeric'}, {'scalar', 'integer', 'positive'}, mfilename, 'sz');
    validateattributes(alpha, {'numeric'}, {'scalar', 'real', 'positive'}, mfilename, 'alpha');
    validateattributes(normalize, {'logical', 'numeric'}, {'scalar'}, mfilename, 'normalize');

    % ---------------------------------------------------------------------
    % 1. Frequency grid construction (equivalent to numpy.fft.fftfreq)
    % ---------------------------------------------------------------------
    % Create frequency indices ranging from 0 … sz-1, then shift so that
    % indices greater than sz/2 represent negative frequencies.
    [kx, ky] = meshgrid(0:sz-1, 0:sz-1);
    kx(kx > sz/2) = kx(kx > sz/2) - sz;
    ky(ky > sz/2) = ky(ky > sz/2) - sz;

    % Normalise by the field size to obtain frequency in cycles per pixel
    kx = kx / sz;
    ky = ky / sz;

    % Square of the wavenumber magnitude |k|^2 --------------------------------
    k_squared = kx.^2 + ky.^2;
    k_squared(1, 1) = 1;   % Avoid divide-by-zero at the DC component

    % ---------------------------------------------------------------------
    % 2. Power-law power spectrum: P(|k|) ∝ |k|^{-alpha}
    % ---------------------------------------------------------------------
    power_spectrum = k_squared .^ (-alpha / 2.0);
    power_spectrum(1, 1) = 0;   % Set the DC component to zero (zero mean)

    % ---------------------------------------------------------------------
    % 3. Complex Gaussian white noise in frequency domain ------------------
    % ---------------------------------------------------------------------
    random_complex = randn(sz) + 1i * randn(sz);

    % ---------------------------------------------------------------------
    % 4. Apply power spectrum and transform back to spatial domain ----------
    % ---------------------------------------------------------------------
    random_field_freq = random_complex .* sqrt(power_spectrum);
    field = real(ifft2(random_field_freq));

    % ---------------------------------------------------------------------
    % 5. Optional linear rescaling to [0, 1] --------------------------------
    % ---------------------------------------------------------------------
    if normalize
        field = field - min(field(:));
        field = field / max(field(:));
    end
end
