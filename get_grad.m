function [theta_grad, eta_grad, W_grad, U_grad] = get_grad (train_text, train_sift, same_pair, dif_pair, ...
														    theta_, eta_, W, U, params)
% ---------------------------------------------------------------------
% USEAGE:
% calculate the sub-gradients defined in Eq.(14-16) for mmdml algorithm 
% ---------------------------------------------------------------------
% INPUT:
% train_text, train_sift, same_pair, dif_pair:
% 		data generate from readTeTrFile function
% t: hidden topics calculated from get_h, (double)train_num * hidden_topics_dim
% params: parameters of the algorithm
% ---------------------------------------------------------------------
% OUTPUT:
% theta_grad, eta_grad, W_grad, U_grad:
% 		sub-gradients of theta, eta, W, U
% ---------------------------------------------------------------------

	lamda_1 = params.lamda_1;
	lamda_2 = params.lamda_2;

	t = get_h(W, U, train_text, train_sift);
	t_dim = size(t,2);
	text_dim = size(train_text, 2);
	sift_dim = size(train_sift, 2);
	train_num = size(train_sift, 1);

	theta_grad = zeros(text_dim,1);
	eta_grad = zeros(sift_dim,1);
	W_grad = zeros(text_dim, t_dim);
	U_grad = zeros(sift_dim, t_dim);

	% S: set of same pairs
	S_size = size(same_pair,1);
	% D: set of dif pairs
	D_size = size(dif_pair,1);

	% t(m) - t(n)
	t_same = zeros(S_size, t_dim);
	t_dif = zeros(D_size, t_dim);

	% x(m) - x(n)
	text_same = zeros(S_size, text_dim);
	text_dif = zeros(D_size, text_dim);

	% z(m) - z(n)
	sift_same = zeros(S_size, sift_dim);
	sift_dif = zeros(D_size, sift_dim);


	% ********************************************************************************************
	% ******************************    ABSOLETE    VERSION         ******************************
	% ********************************************************************************************
	% for i = 1:S_size
	% 	m = same_pair(i,1);
	% 	n = same_pair(i,2);
	% 	t_same(i,:) = t(m,:) - t(n,:);
	% 	text_same(i,:) = train_text(m,:) - train_text(n,:);
	% 	sift_same(i,:) = train_sift(m,:) - train_sift(n,:);
	% end
	% 
	% for i = 1:D_size
	% 	m = dif_pair(i,1);
	% 	n = dif_pair(i,2);
	% 	t_dif(i,:) = t(m,:) - t(n,:);
	% 	text_dif(i,:) = train_text(m,:) - train_text(n,:);
	% 	sift_dif(i,:) = train_sift(m,:) - train_sift(n,:);
	% end
	% 
	% % calculate metric based grad for W, U
	% for k = 1:t_dim
	% 	for i = 1:text_dim
	% 		W_grad(i,k) = W_grad(i,k) + t_same(:,k)' * text_same(:,k)*2*lamda_1/S_size;
	% 		for tem = 1:D_size
	% 			if norm(t_dif(i,:),2) >= 1
	% 				continue;
	% 			end
	% 			W_grad(i,k) = W_grad(i,k) + t_dif(tem,k) * text_dif(tem,k)*2*lamda_2/D_size;
	% 		end
	% 	end
	% 	for i = 1:sift_dim
	% 		U_grad(i,k) = U_grad(i,k) + t_same(:,k)' * sift_same(:,k)*2*lamda_1/S_size;
	% 		for tem = 1:D_size
	% 			if norm(t_dif(i,:),2) >= 1
	% 				continue;
	% 			end
	% 			U_grad(i,k) = U_grad(i,k) + t_dif(tem,k) * sift_dif(tem,k)*2*lamda_2/D_size;
	% 		end
	% 	end
	% end
	% ********************************************************************************************


	% a faster version
	% calculate metric based grad for W, U
	for i = 1:S_size
		m = same_pair(i,1);
		n = same_pair(i,2);
		t_same(i,:) = t(m,:) - t(n,:);
		text_same(i,:) = train_text(m,:) - train_text(n,:);
		sift_same(i,:) = train_sift(m,:) - train_sift(n,:);
	end

	for i = 1:D_size
		m = dif_pair(i,1);
		n = dif_pair(i,2);
		t_dif(i,:) = t(m,:) - t(n,:);
		text_dif(i,:) = train_text(m,:) - train_text(n,:);
		sift_dif(i,:) = train_sift(m,:) - train_sift(n,:);
	end

	t_valid = sum(t_dif.^2,2) >= 1;
	t_dif = t_dif(t_valid,:);
	text_dif = text_dif(t_valid,:);
	sift_dif = sift_dif(t_valid,:);

	for k = 1:t_dim
		for i = 1:text_dim
			W_grad(i,k) = W_grad(i,k) + t_same(:,k)' * text_same(:,k)*2*lamda_1/S_size;
			W_grad(i,k) = W_grad(i,k) + t_dif(:,k)' * text_dif(:,k)*2*lamda_2/D_size;
		end
		for i = 1:sift_dim
			U_grad(i,k) = U_grad(i,k) + t_same(:,k)' * sift_same(:,k)*2*lamda_1/S_size;
			U_grad(i,k) = U_grad(i,k) + t_dif(:,k)' * sift_dif(:,k)*2*lamda_2/D_size;
		end
	end

	% contrastive divergence
	[gibbs_text, gibbs_sift, gibbs_t] = Gibbs_sampling (train_text, train_sift, theta_, eta_, W, U);
	theta_grad = mean(gibbs_text - train_text)';
	eta_grad = mean(gibbs_sift - train_sift)';
	for k = 1:t_dim
		for i = 1:text_dim
			W_grad(i,k) = W_grad(i,k) + (gibbs_text(:,i)'*gibbs_t(:,k) - train_text(:,i)'*t(:,k))/train_num;
		end
		for i = 1:sift_dim
			U_grad(i,k) = U_grad(i,k) + (gibbs_sift(:,i)'*gibbs_t(:,k) - train_sift(:,i)'*t(:,k))/train_num;
		end
	end

end