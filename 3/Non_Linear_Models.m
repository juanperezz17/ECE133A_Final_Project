% Loading in data
data = readtable ("OnlineNewsPopularity.csv");
articleData = data{:,3:61};

% Separating into folds
X_stand = articleData;
X_features = X_stand(:,1:58);
target = X_stand(:,59);
folds = 5;
cv = cvpartition(size(X_features,1), 'KFold', folds);
rmse = zeros(folds,1);


%%
% Set number of parameters, i.e. features
num_param = 58;

for k = 1:folds

    % Training and test sets for each fold
    train_index = cv.training(k);
    test_index = cv.test(k); 
    X_train = X_features(train_index,:);
    y_train = target(train_index,:);
    X_test = X_features(test_index,:);
    y_test = target(test_index,:);

    % Setting initial guess for parameters
    x0 = ones(1, num_param);
    for i = 1:58
        x0(i) = 0.001;
    end

    % Running non-linear fitting algorithm
    parameters = lsqcurvefit(@fun, x0, X_train, y_train);

    % Predicted y value
    y_pred = fun(parameters, X_test);

    % Calculate RMS error
    rmse(k) = sqrt(mean((y_test - y_pred).^2));

end
%%

% Non-linear prediction function for part 3d
% Minimize (fun(x, xdata(i)) - y(i))^2
% x is parameter vector, xdata(i) is feature vector for each data point i
function F = fun(x, xdata)
    F = zeros(size(xdata, 1), 1);
    for i = 1:size(xdata, 1)
        F(i, 1) = atan(xdata(i,:)*transpose(x));
    end
end