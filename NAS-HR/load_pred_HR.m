% Load the .mat files
data_pr = load("C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\NAS-HR\Search&Train\augments\HR6_Augment\0HR_pr.mat");  % Replace 'path_to_file' with the actual path
data_rel = load("C:\Users\User\Documents\Monash\FYP\health-check-with-human-face-scanner\NAS-HR\Search&Train\augments\HR6_Augment\0HR_rel.mat");

% Combine HR_pr and HR_rel side by side
combined_data = [data_pr.HR_pr(:), data_rel.HR_rel(:)];  % Ensure both are column vectors

% Display the combined data
disp('Predicted and Relative Heart Rates (side by side):');
disp('   Predicted HR       Relative HR');
disp(combined_data);

% Add headers
headers = {'Predicted HR', 'Relative HR'};
output_data = [headers; num2cell(combined_data)];

% Save the combined data with headers to a CSV file
writecell(output_data, 'HR_comparison.csv');