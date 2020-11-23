function predicted_detection = predict_detection_vgg(reference_image,test_image, eccentricity)
% Predict the detection of difference between two given images at specific eccentricity.
% reference_image, test_image - images used for prediction; need to of at least 256x256 size.
% eccentricity - the eccentricity in degrees, at which the image is presented. Has to be one of the following: 8, 14, 20

    eccentricities = [8, 14, 20];

    [width, height, ~] = size(reference_image);
    if any(size(reference_image) ~= size(test_image))
        error('Images do not have equal dimesions.');
    end
    
    if width < 256 || height < 256
        error('Images have to be at least of size 256x256.');
    end
    
    if ~any(eccentricity == eccentricities)
        error('Specify one of the following eccentricities: 8, 14, 20.');
    end
    
    matname = sprintf("vgg_%d", eccentricity);
    fitting_parameters = matfile(matname).fitting_parameters;
    logistic_parameters = matfile(matname).logistic_parameters;
    counter = 0;
    prediction = 0;
    layers = ["conv1_1", "conv1_2", "pool1", "conv2_1", "conv2_2", "pool2", ...
        "conv3_1", "conv3_2", "conv3_3", "conv3_4", "pool3", ...
        "conv4_1", "conv4_2", "conv4_3", "conv4_4", "pool4", ...
        "conv5_1", "conv5_2", "conv5_3", "conv5_4", "pool5"];
    fitting_function = @(w, x) w(1) + x * w(2:end);
    logistic_function = @(w, x) w(1) + (w(2) - w(1)) ./ ((w(3) + w(4) * exp(-x * w(5))) .^ (1 ./ w(6)));
    net = vgg19();
    for i = 1 : 256 : height - 255
        for j = 1 : 256 : width - 255
            for k = 1 : length(layers)
                current_original_features = double(activations(net, reference_image(i : i + 255, j : j + 255, :), layers(k)));
                current_distorted_features = double(activations(net, test_image(i : i + 255, j : j + 255, :), layers(k)));
                total_features(k) = mean(abs(current_distorted_features - current_original_features), 'all');
            end
            flatten_predictions = fitting_function(fitting_parameters, total_features);
            prediction = prediction + logistic_function(logistic_parameters, flatten_predictions);
            counter = counter + 1;
        end
    end
    predicted_detection = prediction / counter;
end

