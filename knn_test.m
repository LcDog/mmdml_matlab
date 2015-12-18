params = struct();
params = SetDefaultParams(params);

[train_text, train_sift, test_text, test_sift, same_pair, dif_pair, train_label, test_label] = readTeTrFile(params);

[theta_init, eta_init, W_init, U_init] = initdata(train_text, train_sift, params);

[theta_, eta_, W, U] = minimize_step(theta_init, eta_init, W_init, U_init, ...
							         train_text, train_sift, same_pair, dif_pair, params);

train = get_h(W, U, train_text, train_sift);
test = get_h(W, U, test_text, test_sift);

A = eye(size(params.latent_dim));
k = 1;

preds = KNN(train_label, train, A, k, test);
acc = sum(preds == test_label)/length(test_label);