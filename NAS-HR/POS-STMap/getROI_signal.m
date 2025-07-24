function [signals] = getROI_signal(frame, landmarks)
%getROI_signal Extract ROI signals from 468-landmark data.

% Reshape landmarks => 468x2
landmarks = reshape(landmarks, 2, [])';

% Initialize output: 1x45
signals = zeros(1,45);

% Forehead polygon (example)
forehead_idx = [10, 338, 297, 332];
x_poly = landmarks(forehead_idx,1);
y_poly = landmarks(forehead_idx,2);
BW = uint8(roipoly(frame, x_poly, y_poly));
numpix = sum(BW(:));
if numpix > 0
    % Store R,G,B in columns 15,30,45
    for i = 1:3
        imgROI = double(frame(:,:,i)) .* double(BW);
        signals(1, i*15) = sum(imgROI(:)) / numpix;
    end
end

% Additional 14 ROIs
roi_indices = {
    [61, 291, 375];
    [122, 398, 362];
    [1, 10, 338];
    [151, 379, 361];
    [234, 454, 356];
    [93, 132, 58];
    [324, 361, 288];
    [57, 200, 172];
    [78, 308, 291];
    [202, 317, 402];
    [150, 152, 148];
    [226, 400, 377];
    [330, 361, 288];
    [400, 378, 356];
};

% ROI i => columns i, i+15, i+30
for roi_index = 1:length(roi_indices)
    inds = roi_indices{roi_index};
    x_poly = landmarks(inds, 1);
    y_poly = landmarks(inds, 2);
    BW_roi = uint8(roipoly(frame, x_poly, y_poly));
    npix = sum(BW_roi(:));
    if npix == 0
        roi_signal = [0, 0, 0];
    else
        roi_signal = zeros(1,3);
        for c = 1:3
            imgROI = double(frame(:,:,c)) .* double(BW_roi);
            roi_signal(c) = sum(imgROI(:)) / npix;
        end
    end
    for c = 1:3
        signals(1, roi_index + (c-1)*15) = roi_signal(c);
    end
end

end
